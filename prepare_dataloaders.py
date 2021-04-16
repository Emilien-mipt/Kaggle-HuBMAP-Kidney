from torch.utils.data import DataLoader

from dataset import HuBMAPDataset


def prepare_train_valid_dataloader(cfg, df, fold):
    train_ids = df[~df.Folds.isin(fold)]
    val_ids = df[df.Folds.isin(fold)]

    train_ds = HuBMAPDataset(cfg, train_ids, mode="train", augment=cfg.train_params.aug_type, transform=True)
    val_ds = HuBMAPDataset(cfg, val_ids, mode="val", augment=cfg.train_params.aug_type, transform=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train_params.batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=cfg.train_params.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train_params.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=cfg.train_params.num_workers,
    )
    return train_loader, val_loader
