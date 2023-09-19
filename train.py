import lightning as L
from torch.utils.data import DataLoader
import torch

from model.SimpleDiffusion import SimpleDiffusion
from dataset import load_cifar_train
import settings

import os


def main():
    TAG = "MASKINGUNet"
    model = SimpleDiffusion(1000, 0.9)
    save_dir = os.path.join(settings.save_dir, TAG)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ds = load_cifar_train(settings.dataset_dir)
    dataloader = DataLoader(
        ds, batch_size=settings.batch_size, shuffle=True, num_workers=12
    )

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=500,
        min_epochs=200,
        default_root_dir=save_dir,
    )

    torch.set_float32_matmul_precision("high")
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
