import torch.nn as nn

from .loss_functions import DiceBCELoss, DiceLoss, Hausdorff_loss, Lovasz_loss


def get_loss(cfg):
    if cfg.criterion == "DiceBCELoss":
        criterion = DiceBCELoss()
    elif cfg.criterion == "DiceLoss":
        criterion = DiceLoss()
    elif cfg.criterion == "Hausdorff":
        criterion = Hausdorff_loss()
    elif cfg.criterion == "Lovasz":
        criterion = Lovasz_loss()
    elif cfg.criterion == "BCELoss":
        criterion = nn.BCEWithLogitsLoss()
    else:
        assert False and "WTF loss?"
        raise ValueError
    return criterion
