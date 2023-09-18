import torch

from model.SimpleDiffusion import SimpleDiffusion
from dataset import save_image, label2num
import settings

import os


def main():
    prompt = ["bird"] * 32
    checkpoint_dir = os.path.join(
        settings.save_dir,
        "lightning_logs",
        "version_17",
        "checkpoints",
        "epoch=499-step=391000.ckpt",
    )

    labels = [label2num(pro) for pro in prompt]
    labels = torch.Tensor(labels).to(settings.device)
    model = (
        SimpleDiffusion(1000, 0.9)
        .load_from_checkpoint(checkpoint_dir)
        .to(settings.device)
    )
    save_dir = os.path.join(settings.save_dir, "samples")

    results = model.sample(labels, 32, 1).detach().cpu().numpy()
    results = (results.clip(-1, 1) + 1) / 2
    for idx, res in enumerate(results):
        save_image(os.path.join(save_dir, f"{idx}.png"), res.transpose(1, 2, 0))


if __name__ == "__main__":
    main()
