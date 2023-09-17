import torch


class Diffuser():
    def __init__(
        self,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device = "cuda",
    ):
        self.t = timesteps
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def __call__(self, img, timesteps):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[timesteps])[:, None, None, None]
        minus_sqrt_alpha_hat = torch.sqrt(1 - self.alpha_hat[timesteps])[:, None, None, None]
        noise = torch.randn_like(img)
        return sqrt_alpha_hat * img + minus_sqrt_alpha_hat * noise, noise
