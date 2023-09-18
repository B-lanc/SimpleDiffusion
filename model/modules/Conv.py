import torch
import torch.nn as nn


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels):
        super(DownConv, self).__init__()
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.pos_emb_linear = nn.Linear(emb_channels, out_channels)
        self.class_emb_linear = nn.Linear(emb_channels, out_channels)

    def forward(self, x, pos_emb, class_emb):
        x = self.norm1(self.conv1(x))
        x = self.activation(x)
        x = self.norm2(self.conv2(x))

        pos_emb = self.activation(self.pos_emb_linear(pos_emb))
        pos_emb = pos_emb[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])

        class_emb = self.activation(self.class_emb_linear(class_emb))
        class_emb = class_emb[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])

        return x + pos_emb + class_emb


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels):
        super(UpConv, self).__init__()
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv2 = nn.ConvTranspose2d(2 * in_channels, out_channels, 2, 2, 0)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.pos_emb_linear = nn.Linear(emb_channels, out_channels)
        self.class_emb_linear = nn.Linear(emb_channels, out_channels)

    def forward(self, x, short, pos_emb, class_emb):
        x = self.norm1(self.conv1(x))
        x = self.activation(x)
        x = torch.cat((x, short), dim=1)
        x = self.norm2(self.conv2(x))

        pos_emb = self.activation(self.pos_emb_linear(pos_emb))
        pos_emb = pos_emb[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])

        class_emb = self.activation(self.class_emb_linear(class_emb))
        class_emb = class_emb[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])

        return x + pos_emb + class_emb

class MASKUpConv(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels):
        super(UpConv, self).__init__()
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, 2, 2, 0)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.pos_emb_linear = nn.Linear(emb_channels, out_channels)
        self.class_emb_linear = nn.Linear(emb_channels, out_channels)

    def forward(self, x, short, pos_emb, class_emb):
        x = self.norm1(self.conv1(x))
        x = self.activation(x)
        x = x * short
        x = self.norm2(self.conv2(x))

        pos_emb = self.activation(self.pos_emb_linear(pos_emb))
        pos_emb = pos_emb[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])

        class_emb = self.activation(self.class_emb_linear(class_emb))
        class_emb = class_emb[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])

        return x + pos_emb + class_emb
