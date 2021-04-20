import os

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset


def img2tensor(img, dtype=np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class HuBMAPDataset(Dataset):
    def __init__(self, data_path, mask_path, label_path, fold, mean, std, n_splits, seed, train=True, tfms=None):
        self.data_path = data_path
        self.mask_path = mask_path
        self.labels = label_path
        ids = pd.read_csv(self.labels).id.values
        kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
        ids = set(ids[list(kf.split(ids))[fold][0 if train else 1]])
        self.fnames = [fname for fname in os.listdir(self.data_path) if fname.split("_")[0] in ids]
        self.train = train
        self.tfms = tfms
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_path, fname), cv2.IMREAD_GRAYSCALE)
        if self.tfms is not None:
            augmented = self.tfms(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]
        return img2tensor((img / 255.0 - self.mean) / self.std), img2tensor(mask)
