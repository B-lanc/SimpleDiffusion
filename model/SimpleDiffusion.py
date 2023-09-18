import lightning as L
import torch

from .modules.UNet import UNet, MASKINGUNet
from .modules.Diffuser import Diffuser

import random


class SimpleDiffusion(L.LightningModule):
    def __init__(self, timesteps=1000, class_rate=0.9):
        super(SimpleDiffusion, self).__init__()
        self.timesteps = timesteps
        self.cr = class_rate

        self.model = MASKINGUNet(3, 3, 256)
        self.diff = Diffuser(timesteps=timesteps)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        labels, imgs = batch
        timesteps = self.gen_steps(labels.shape[0])
        noised_img, noise = self.diff(imgs, timesteps)

        if random.random() > self.cr:
            labels = None
        preds = self.model(noised_img, timesteps, labels)
        loss = torch.nn.functional.mse_loss(preds, noise)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def gen_steps(self, batch_size):
        return torch.randint(1, self.timesteps, size=(batch_size,), device=self.device)

    def sample(self, labels, n, cfg_scale=3):
        """
        labels: either 1D tensor of the label numbers (0 to 9) or None, if None, then uses unconditional, if tensor, then the amount of images determined by the length of the tensor
        n: only if labels is None, the amount of images to generate
        """
        with torch.no_grad():
            if labels is None:
                x = torch.randn((n, 3, 32, 32)).to(self.device)
            else:
                n = labels.shape[0]
                x = torch.randn((n, 3, 32, 32)).to(self.device)

            for i in reversed(range(1, self.timesteps)):
                t = (torch.ones((n)) * i).long().to(self.device)
                pred = self.model(x, t, labels)
                if labels is not None:
                    unlabeled = self.model(x, t, None)
                    pred = unlabeled + (pred - unlabeled) * cfg_scale
                temp = self.diff.sample_step(x, pred, t)
                x = temp

        return x
