import torch
import torch.nn as nn

from .util import sinusodial
from .Conv import DownConv, UpConv


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels):
        super(UNet, self).__init__()
        self.emb_channels = emb_channels
        self.inconv = nn.Conv2d(in_channels, 64, 1, 1, 0)

        self.down1 = DownConv(64, 128, self.emb_channels)
        self.down2 = DownConv(128, 256, self.emb_channels)

        self.bottle1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bottle2 = nn.Conv2d(256, 256, 3, 1, 1)

        self.up1 = UpConv(256, 128, self.emb_channels)
        self.up2 = UpConv(128, 64, self.emb_channels)

        self.outconv = nn.Conv2d(64, out_channels, 1, 1, 0)

    def forward(self, x, t, labels=None):
        pos_emb = sinusodial(t, self.emb_channels)
        if labels is None:
            class_emb = torch.zeros_like(pos_emb)
        else:
            class_emb = sinusodial(labels, self.emb_channels)

        x = self.inconv(x)

        x1 = self.down1(x, pos_emb, class_emb)
        x2 = self.down2(x1, pos_emb, class_emb)
        x = self.bottle1(x2)
        x = self.bottle2(x)
        x = self.up1(x, x2, pos_emb, class_emb)
        x = self.up2(x, x1, pos_emb, class_emb)

        x = self.outconv(x)
        return x
