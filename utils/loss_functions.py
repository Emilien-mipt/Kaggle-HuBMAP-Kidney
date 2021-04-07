import torch.nn as nn
from losses_pytorch.hausdorff import HausdorffDTLoss
from losses_pytorch.lovasz_loss import LovaszSoftmax
from torch.nn import functional as F

from config import CFG


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

        return dice


class DiceBCELoss(nn.Module):
    # Formula Given above.
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=CFG.smoothing):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).mean()
        dice_loss = 1 - (2.0 * intersection + smooth) / (inputs.mean() + targets.mean() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        Dice_BCE = BCE + dice_loss

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
