import torch.nn as nn
from pytorch_toolbelt.losses import DiceLoss, JaccardLoss, LovaszLoss

from .loss_functions import DiceBCELoss, Hausdorff_loss


def get_loss(cfg):
    if cfg.criterion == "DiceBCELoss":
        criterion = DiceBCELoss()
    elif cfg.criterion == "DiceLoss":
        criterion = DiceLoss(mode="binary")
    elif cfg.criterion == "JaccardLoss":
        criterion = JaccardLoss(mode="binary")
    elif cfg.criterion == "Hausdorff":
        criterion = Hausdorff_loss()
    elif cfg.criterion == "Lovasz":
        criterion = LovaszLoss()
    elif cfg.criterion == "BCELoss":
        criterion = nn.BCEWithLogitsLoss()
    else:
        assert False and "WTF loss?"
        raise ValueError
    return criterion
