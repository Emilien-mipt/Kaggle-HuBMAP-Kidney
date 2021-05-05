import torch
import torch.nn as nn
from torch.nn import functional as F

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


class DiceBCELoss:
    def __init__(self, dice_weight=1):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.dice_weight:
            eps = 1e-15
            dice_target = (targets == 1).float()
            dice_output = outputs
            intersection = (dice_output * dice_target).sum()
            union = dice_output.sum() + dice_target.sum() + eps

            loss -= torch.log(2 * intersection / union)

        return loss


class Hausdorff_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        return HausdorffDTLoss()(inputs, targets)
