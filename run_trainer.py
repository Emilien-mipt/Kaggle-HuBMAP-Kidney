import os
import time

import hydra
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold
from torch.utils.tensorboard import SummaryWriter

from config import CFG
from model import HuBMAP
from prepare_dataloaders import prepare_train_valid_dataloader
from train_val import train_one_epoch, valid_one_epoch
from utils.get_loss import get_loss
from utils.get_optimizer import get_optimizer
from utils.get_scheduler import get_scheduler
from utils.utils import init_logger, plot, save_model, seed_torch


def run_trainer(cfg):
    # Create dir for saving logs and weights
    print("Creating dir for saving weights")
    os.makedirs("weights")
    print("Dir has been created!")

    # Define logger to save train logs
    LOGGER = init_logger("train.log")

    # Set seed
    seed_torch(seed=cfg.train_params.seed)

    # Write to tensorboard
    tb = SummaryWriter(os.getcwd())

    # Define paths
    directory_list = os.listdir(hydra.utils.to_absolute_path(cfg.dataset.path))
    LOGGER.info(f"Choose dataset {cfg.dataset.name}")
    LOGGER.info(f"Tile size: {cfg.dataset.tile_size}")

    directory_list = [fnames.split("_")[0] for fnames in directory_list]
    dir_df = pd.DataFrame(directory_list, columns=["id"])
    print(dir_df.shape)

    FOLDS = cfg.train_params.n_splits
    gkf = GroupKFold(FOLDS)
    dir_df["Folds"] = 0
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(dir_df, groups=dir_df[dir_df.columns[0]].values)):
        dir_df.loc[val_idx, "Folds"] = fold
    LOGGER.info(f"Choose cross validation strategy with {cfg.train_params.n_splits} folds")

    if cfg.model == "Unet":
        base_model = smp.Unet(cfg.model.encoder, encoder_weights="imagenet", classes=1)
    if CFG.base_model == "FPN":
        base_model = smp.FPN(cfg.model.encoder, encoder_weights="imagenet", classes=1)
    LOGGER.info(f"Model: {cfg.model.name}")

    device = torch.device(f"cuda:{cfg.train_params.gpu_id}")
    LOGGER.info(f"Device: {cfg.train_params.gpu_id}")

    model = HuBMAP(base_model).to(device)

    # optimizer
    optimizer = get_optimizer(cfg, model)
    LOGGER.info(f"Optimizer: {cfg.optimizer.type}")

    # scheduler setting
    # scheduler = get_scheduler(cfg, optimizer)
    scheduler = None
    # LOGGER.info(f"Scheduler: {cfg.scheduler.type}")

    criterion = get_loss(cfg)
    LOGGER.info(f"Criterion: {cfg.criterion}")

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler()

    for fold, (train_idx, val_idx) in enumerate(gkf.split(dir_df, groups=dir_df[dir_df.columns[0]].values)):
        if fold > 1:
            break
        train_loader, valid_loader = prepare_train_valid_dataloader(cfg, dir_df, [fold])

        best_epoch = 0
        best_dice_score = 0.0
        count_bad_epochs = 0

        LOGGER.info("-" * 50)
        LOGGER.info(f"Start training FOLD {fold}...")

        for epoch in range(cfg.train_params.epoch):

            start_time = time.time()

            # train
            avg_train_loss, train_dice = train_one_epoch(
                cfg, train_loader, model, criterion, optimizer, scaler, epoch, device, scheduler
            )

            # eval
            avg_val_loss, val_dice_score = valid_one_epoch(valid_loader, model, criterion, device)

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
            LOGGER.info(f"Epoch {epoch + 1} - Dice: {val_dice_score}")

            # Update best score
            if val_dice_score >= best_dice_score:
                best_dice_score = val_dice_score
                LOGGER.info(f"Epoch {epoch + 1} - Save Best Dice: {best_dice_score:.4f}")
                save_model(
                    model,
                    epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
                    val_dice_score,
                    os.path.join("weights", f"fold-{fold}_best.pt"),
                )
                best_epoch = epoch + 1
                count_bad_epochs = 0
            else:
                count_bad_epochs += 1
            print(count_bad_epochs)
            LOGGER.info(f"Number of bad epochs {count_bad_epochs}")
            # Early stopping
            if count_bad_epochs > cfg.train_params.early_stopping:
                LOGGER.info(
                    f"Stop the training, since the score has not improved for "
                    f"{cfg.train_params.early_stopping} epochs!"
                )
                save_model(
                    model,
                    epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
                    val_dice_score,
                    os.path.join("weights", f"fold-{fold}_epoch{epoch + 1}_last.pth"),
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
                    os.path.join("weights", f"fold-{fold}_epoch{epoch + 1}_final.pth"),
                )
        LOGGER.info(f"AFTER TRAINING: Epoch {best_epoch}: Best Dice score: {best_dice_score:.4f}")
        tb.close()


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    run_trainer(cfg)


if __name__ == "__main__":
    run()
