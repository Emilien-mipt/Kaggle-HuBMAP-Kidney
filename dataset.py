import os

import cv2
import hydra
from torch.utils.data import Dataset

from augmentations import (
    base_transform,
    strong_transform,
    val_transform,
    weak_transform,
)
from config import CFG


class HuBMAPDataset(Dataset):
    def __init__(self, cfg, df, mode="train", augment="weak", transform=True):
        ids = df.id.values
        self.data_path = hydra.utils.to_absolute_path(cfg.dataset.path)
        self.mask_path = hydra.utils.to_absolute_path(cfg.dataset.mask_path)
        self.fnames = [fname for fname in os.listdir(self.data_path) if fname.split("_")[0] in ids]
        self.mode = mode
        self.augment = augment
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_path, fname), cv2.IMREAD_GRAYSCALE)

        if self.mode == "train":
            if self.transform is True:
                if self.augment == "base":
                    augmented = base_transform(image=img, mask=mask)
                    img, mask = augmented["image"], augmented["mask"]
                elif self.augment == "weak":
                    augmented = weak_transform(image=img, mask=mask)
                    img, mask = augmented["image"], augmented["mask"]
                elif self.augment == "strong":
                    augmented = strong_transform(image=img, mask=mask)
                    img, mask = augmented["image"], augmented["mask"]

        elif self.mode == "val":
            transformed = val_transform(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]

        img = img.type("torch.FloatTensor")
        img = img / 255
        mask = mask.type("torch.FloatTensor")

        return img, mask
