import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    """
    Stolen from stable diffusion :>
    """
    def __init__(self, in_channel):
        super(AttentionBlock, self).__init__()

        self.norm = nn.GroupNorm(8, in_channel)
        self.q = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.k = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.v = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.out = nn.Conv2d(in_channel, in_channel, 1, 1, 0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, w * h)
        q = q.permute(0, 2, 1)      # (b, wh, c)
        k = k.reshape(b, c, w * h)  # (b, c, wh)

        w_ = torch.bmm(q, k)  # (b, qwh, kwh)
        w_ = w_ * (int(c) ** 0.5)
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = v.reshape(b, c, w * h)  # (b, c, wh)
        w_ = w_.permute(0, 2, 1)    # (b, kwh, qwh)
        h_ = torch.bmm(v, w_)       # (b, c, qwh)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.out(h_)

        return x + h_
