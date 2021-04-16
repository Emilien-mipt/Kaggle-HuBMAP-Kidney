from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)


def get_scheduler(cfg, optimizer):
    if cfg.scheduler.type == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.scheduler.T_0, T_mult=cfg.scheduler.T_1, eta_min=cfg.scheduler.eta_min, last_epoch=-1
        )
    elif cfg.scheduler.type == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=cfg.scheduler.mode,
            factor=cfg.scheduler.factor,
            patience=cfg.scheduler.patience,
            verbose=True,
            eps=cfg.scheduler.eps,
        )
    elif cfg.scheduler.type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=cfg.scheduler.T_max, eta_min=cfg.scheduler.min_lr, last_epoch=-1
        )
    else:
        assert False and "WTF optimizer?"
        raise ValueError
    return scheduler
