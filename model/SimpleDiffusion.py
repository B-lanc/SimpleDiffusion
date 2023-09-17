import lightning as L
import torch

from .modules.UNet import UNet
from .modules.Diffuser import Diffuser

import random

class SimpleDiffusion(L.LightningModule):
    def __init__(self, timesteps=1000, class_rate=0.9):
        super(SimpleDiffusion, self).__init__()
        self.timesteps = timesteps
        self.cr = class_rate

        self.model = UNet(3, 3, 256)
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
        return loss

    def gen_steps(self, batch_size):
        return torch.randint(1, self.timesteps, size=(batch_size,), device=self.device)