import torch.nn as nn


class HuBMAP(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.cnn_model = model

    def forward(self, imgs):
        img_segs = self.cnn_model(imgs)
        return img_segs
