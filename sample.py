import torch

from model.SimpleDiffusion import SimpleDiffusion
from dataset import save_image, label2num
import settings

import os


def main():
    prompt = ["frog"] * 64
    checkpoint_dir = os.path.join(
        settings.save_dir,
        "MASKINGUNetNoAtt",
        "lightning_logs",
        "version_0",
        "checkpoints",
        "epoch=99-step=78200.ckpt",
    )
    labels = [label2num(pro) for pro in prompt]
    labels = torch.Tensor(labels).to(settings.device)

    model = (
        SimpleDiffusion.load_from_checkpoint(
            checkpoint_dir,
            timesteps=1000,
            class_rate=0.9,
            MASKING=True,
            ATTENTION=False,
        )
        .to(settings.device)
        .eval()
    )

    save_dir = os.path.join(settings.save_dir, "samples")

    results = model.sample(labels, 32, 3).detach().cpu().numpy()
    # results = (results.clip(-1, 1) + 1) / 2
    results = results.clip(0, 1)
    for idx, res in enumerate(results):
        save_image(os.path.join(save_dir, f"{idx}.png"), res.transpose(1, 2, 0))


if __name__ == "__main__":
    main()
