import math
import os
import random
import time
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

import matplotlib.pyplot as plt
import numpy as np
import torch


def init_logger(log_file_name):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file_name)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision=10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "{} (remain {})".format(asMinutes(s), asMinutes(rs))


def save_model(model, epoch, trainloss, valloss, metric, name):
    """Saves PyTorch model."""
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "train_loss": trainloss,
            "val_loss": valloss,
            "metric_loss": metric,
        },
        os.path.join("weights", name),
    )


def save_plot(imgs, masks, name):
    for i, (img, mask) in enumerate(zip(imgs, masks)):
        img = ((img.permute(1, 2, 0)) * 255.0).numpy().astype(np.uint8)
        plt.subplot(8, 8, i + 1)
        plt.imshow(img, vmin=0, vmax=255)
        plt.imshow(mask.squeeze().numpy(), alpha=0.2)
        plt.axis("off")
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.savefig(name)
