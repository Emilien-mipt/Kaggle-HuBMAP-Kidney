import os

import pandas as pd
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch import FPN, Unet
from sklearn.model_selection import GroupKFold
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader

from config import CFG
from dataset import HuBMAPDataset
from model import HuBMAP
from train_val import train_one_epoch, valid_one_epoch
from utils.loss_functions import DiceBCELoss, DiceLoss, Hausdorff_loss, Lovasz_loss
from utils.utils import plot


def prepare_train_valid_dataloader(df, fold):
    train_ids = df[~df.Folds.isin(fold)]
    val_ids = df[df.Folds.isin(fold)]

    train_ds = HuBMAPDataset(train_ids, mode="train", augment="base", transform=True)
    val_ds = HuBMAPDataset(val_ids, mode="val", augment="base", transform=True)
    train_loader = DataLoader(train_ds, batch_size=4, pin_memory=True, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4, pin_memory=True, shuffle=False, num_workers=4)
    return train_loader, val_loader


def main():
    if CFG.data == 512:
        directory_list = os.listdir("./data/hubmap-512x512/train")
    elif CFG.data == 256:
        directory_list = os.listdir("./data/hubmap-256x256/train")
    directory_list = [fnames.split("_")[0] for fnames in directory_list]
    dir_df = pd.DataFrame(directory_list, columns=["id"])

    if CFG.base_model == "Unet":
        base_model = smp.Unet(CFG.encoder, encoder_weights="imagenet", classes=1)
    if CFG.base_model == "FPN":
        base_model = smp.FPN(CFG.encoder, encoder_weights="imagenet", classes=1)
    print(base_model)

    FOLDS = 5
    gkf = GroupKFold(FOLDS)
    dir_df["Folds"] = 0
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(dir_df, groups=dir_df[dir_df.columns[0]].values)):
        dir_df.loc[val_idx, "Folds"] = fold

    if CFG.criterion == "DiceBCELoss":
        criterion = DiceBCELoss()
    elif CFG.criterion == "DiceLoss":
        criterion = DiceLoss()
    elif CFG.criterion == "Hausdorff":
        criterion = Hausdorff_loss()
    elif CFG.criterion == "Lovasz":
        criterion = Lovasz_loss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HuBMAP(base_model).to(device)
    # optimizer
    optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)

    # scheduler setting
    if CFG.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
    elif CFG.scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps
        )
    elif CFG.scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(dir_df, groups=dir_df[dir_df.columns[0]].values)):
        if fold > 1:
            break
        trainloader, validloader = prepare_train_valid_dataloader(dir_df, [fold])

        num_epochs = CFG.epoch
        # num_epochs = 2
        for epoch in range(num_epochs):
            train_one_epoch(epoch, model, device, optimizer, scheduler, trainloader, criterion)
            with torch.no_grad():
                valid_one_epoch(epoch, model, device, optimizer, scheduler, validloader, criterion)
        torch.save(model.state_dict(), f"FOLD-{fold}-model.pth")


if __name__ == "__main__":
    main()
