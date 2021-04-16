import torch.nn as nn
from torch.nn import functional as F

from config import CFG

from .losses_pytorch.hausdorff import HausdorffDTLoss
from .losses_pytorch.lovasz_loss import LovaszSoftmax


def get_dice_coeff(pred, targs):
    """
    Calculates the dice coeff of a single or batch of predicted mask and true masks.

    Args:
        pred : Batch of Predicted masks (b, w, h) or single predicted mask (w, h)
        targs : Batch of true masks (b, w, h) or single true mask (w, h)

    Returns: Dice coeff over a batch or over a single pair.
    """

    pred = (pred > 0).float()
    return 2.0 * (pred * targs).sum() / ((pred + targs).sum() + 1.0)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=CFG.smoothing):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    # Formula Given above.
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=CFG.smoothing):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # BCE part of the loss
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        intersection = (inputs * targets).mean()
        dice_loss = 1 - (2.0 * intersection + smooth) / (inputs.mean() + targets.mean() + smooth)

        Dice_BCE = BCE + (1 - dice_loss)

        return Dice_BCE.mean()


class Hausdorff_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        return HausdorffDTLoss()(inputs, targets)


class Lovasz_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        return LovaszSoftmax()(inputs, targets)
