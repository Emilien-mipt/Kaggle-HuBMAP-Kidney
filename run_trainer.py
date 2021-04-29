import logging
import os
import time

import hydra
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from omegaconf import DictConfig
from pytorch_toolbelt.utils import count_parameters
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from augmentations import get_aug
from dataset import HuBMAPDataset
from model import HuBMAP
from train_val import train_one_epoch, valid_one_epoch
from utils.get_loss import get_loss
from utils.get_optimizer import get_optimizer
from utils.get_scheduler import get_scheduler
from utils.utils import save_model, save_plot, seed_torch


def run_trainer(cfg):
    # Create dir for saving logs and weights
    print("Creating dir for saving weights and tensorboard logs")
    os.makedirs("weights")
    os.makedirs("logs")
    print("Dir has been created!")

    # Define logger to save train logs
    LOGGER = logging.getLogger(__name__)

    # Set seed
    seed = cfg.train_params.seed
    seed_torch(seed=seed)

    # Define paths
    img_path = hydra.utils.to_absolute_path(cfg.dataset.path)
    directory_list = os.listdir(img_path)
    LOGGER.info(f"Choose dataset {cfg.dataset.name}")
    LOGGER.info(f"Tile size: {cfg.dataset.tile_size}")

    directory_list = [fnames.split("_")[0] for fnames in directory_list]
    dir_df = pd.DataFrame(directory_list, columns=["id"])
    print(dir_df.shape)

    n_splits = cfg.train_params.n_splits
    LOGGER.info(f"Choose cross validation strategy with {n_splits} folds")

    if cfg.model.name == "Unet":
        base_model = smp.Unet(cfg.model.encoder, encoder_weights="imagenet", classes=1)
    if cfg.model.name == "FPN":
        base_model = smp.FPN(cfg.model.encoder, encoder_weights="imagenet", classes=1)
    LOGGER.info(f"Model: {cfg.model.name}")
    LOGGER.info(f"Encoder: {cfg.model.encoder}")

    device = torch.device(f"cuda:{cfg.train_params.gpu_id}")
    LOGGER.info(f"Device: {cfg.train_params.gpu_id}")

    model = HuBMAP(base_model).to(device)

    LOGGER.info(f"Number of parameters in the model: {count_parameters(model)}")

    # optimizer
    optimizer = get_optimizer(cfg, model)
    LOGGER.info(f"Optimizer: {cfg.optimizer.type}")

    # scheduler setting
    scheduler = get_scheduler(cfg, optimizer)
    LOGGER.info(f"Scheduler: {cfg.scheduler.type}")

    criterion = get_loss(cfg)
    LOGGER.info(f"Criterion: {cfg.criterion}")

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler()

    # Train params
    batch_size = cfg.train_params.batch_size
    LOGGER.info(f"Batch size: {batch_size}")

    # Train params
    num_workers = cfg.train_params.num_workers
    mask_path = hydra.utils.to_absolute_path(cfg.dataset.mask_path)
    label_path = hydra.utils.to_absolute_path(cfg.labels)
    mean = np.array(cfg.dataset.mean)
    std = np.array(cfg.dataset.std)

    fold_scores = np.zeros(shape=(n_splits, 1))

    # Start training
    for fold in range(n_splits):
        # Create dirs for corresponding folds
        os.makedirs(os.path.join("weights", f"fold-{fold}"))
        os.makedirs(os.path.join("logs", f"fold-{fold}"))

        # Write to tensorboard
        tb = SummaryWriter(os.path.join("logs", f"fold-{fold}"))

        train_ds = HuBMAPDataset(
            img_path,
            mask_path,
            label_path,
            fold=fold,
            mean=mean,
            std=std,
            n_splits=n_splits,
            seed=seed,
            train=True,
            tfms=get_aug(),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_ds = HuBMAPDataset(
            img_path,
            mask_path,
            label_path,
            fold=fold,
            mean=mean,
            std=std,
            n_splits=n_splits,
            seed=seed,
            train=False,
        )
        valid_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        save_plot(train_ds, 10, mean, std, f"train_fold{fold}.png")
        print(f"Saved train images for fold {fold}")

        best_epoch = 0
        best_dice_score = 0.0
        count_bad_epochs = 0

        LOGGER.info("-" * 100)
        LOGGER.info(f"Start training FOLD {fold}...")

        for epoch in range(cfg.train_params.epochs):

            start_time = time.time()

            # train
            avg_train_loss, train_dice = train_one_epoch(
                cfg, train_loader, model, criterion, optimizer, scaler, epoch, device, scheduler
            )

            # eval
            avg_val_loss, val_dice_score = valid_one_epoch(cfg, valid_loader, model, criterion, device)

            cur_lr = optimizer.param_groups[0]["lr"]

            LOGGER.info(f"Current learning rate: {cur_lr}")

            tb.add_scalar("Learning rate", cur_lr, epoch + 1)
            tb.add_scalar("Train Loss", avg_train_loss, epoch + 1)
            tb.add_scalar("Train Dice", train_dice, epoch + 1)
            tb.add_scalar("Val Loss", avg_val_loss, epoch + 1)
            tb.add_scalar("Val Dice", val_dice_score, epoch + 1)

            elapsed = time.time() - start_time

            LOGGER.info(
                f"Epoch {epoch + 1} - avg_train_loss: {avg_train_loss:.4f} \
                    avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
            )
            LOGGER.info(f"Epoch: {epoch + 1} - Val_loss: {avg_val_loss} - Dice: {val_dice_score}")

            # Update best score
            if val_dice_score >= best_dice_score:
                best_dice_score = val_dice_score
                fold_scores[fold, 0] = best_dice_score
                LOGGER.info(f"Epoch {epoch + 1} - Save Best Dice: {best_dice_score:.4f}")
                save_model(
                    model,
                    epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
                    val_dice_score,
                    os.path.join(f"fold-{fold}", f"fold-{fold}_best.pt"),
                )
                best_epoch = epoch + 1
                count_bad_epochs = 0
            else:
                count_bad_epochs += 1
            print(count_bad_epochs)
            LOGGER.info(f"Number of bad epochs {count_bad_epochs}")
            # Early stopping
            if count_bad_epochs > cfg.train_params.early_stop:
                LOGGER.info(
                    f"Stop the training, since the score has not improved for "
                    f"{cfg.train_params.early_stop} epochs!"
                )
                save_model(
                    model,
                    epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
                    val_dice_score,
                    os.path.join(f"fold-{fold}", f"fold-{fold}_last.pt"),
                )
                break
            elif epoch + 1 == cfg.train_params.epochs:
                LOGGER.info(f"Reached the final {epoch + 1} epoch!")
                save_model(
                    model,
                    epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
                    val_dice_score,
                    os.path.join(f"fold-{fold}", f"fold-{fold}_final.pt"),
                )
        LOGGER.info(f"AFTER TRAINING FOLD {fold}: Epoch {best_epoch}: Best Dice score: {best_dice_score:.4f}")
        tb.close()
    LOGGER.info(f"Array with best scores for corresponding folds: {fold_scores}")
    avg_score = np.mean(fold_scores)
    LOGGER.info(f"Average score: {avg_score:.4f}")


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    run_trainer(cfg)


if __name__ == "__main__":
    run()
