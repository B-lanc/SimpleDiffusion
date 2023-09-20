import torch
import torch.nn as nn
import lightning as L

from .util import sinusodial
from .Conv import DownConv, UpConv, MASKUpConv
from .Attention import AttentionBlock


class UNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, emb_channels, MASKING=False, ATTENTION=False
    ):
        super(UNet, self).__init__()
        UP = MASKUpConv if MASKING else UpConv
        att = AttentionBlock if ATTENTION else nn.Identity

        self.emb_channels = emb_channels
        self.inconv = nn.Conv2d(in_channels, 64, 1, 1, 0)

        self.down1 = DownConv(64, 128, self.emb_channels)
        self.attn1 = att(128)
        self.down2 = DownConv(128, 256, self.emb_channels)
        self.attn2 = att(256)
        self.down3 = DownConv(256, 512, self.emb_channels)

        self.bottle1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bottle2 = nn.Conv2d(512, 512, 3, 1, 1)

        self.up1 = UP(512, 256, self.emb_channels)
        self.attn3 = att(256)
        self.up2 = UP(256, 128, self.emb_channels)
        self.attn4 = att(128)
        self.up3 = UP(128, 64, self.emb_channels)

        self.outconv = nn.Conv2d(64, out_channels, 1, 1, 0)

    def forward(self, x, t, labels=None):
        device = x.device
        pos_emb = sinusodial(t, self.emb_channels, device)
        if labels is None:
            class_emb = torch.zeros_like(pos_emb)
        else:
            class_emb = sinusodial(labels, self.emb_channels, device)

        x = self.inconv(x)

        x1 = self.down1(x, pos_emb, class_emb)
        x2 = self.attn1(x1)
        x2 = self.down2(x2, pos_emb, class_emb)
        x3 = self.attn2(x2)
        x3 = self.down3(x3, pos_emb, class_emb)
        x = self.bottle1(x3)
        x = self.bottle2(x)
        x = self.up1(x, x3, pos_emb, class_emb)
        x = self.attn3(x)
        x = self.up2(x, x2, pos_emb, class_emb)
        x = self.attn4(x)
        x = self.up3(x, x1, pos_emb, class_emb)

        x = self.outconv(x)
        return x
