import lightning as L
import torch


class SimpleDiffusion(L.LightningModule):
    def __init__(self):
        super(SimpleDiffusion, self).__init__()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
