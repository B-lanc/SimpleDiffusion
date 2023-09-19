import torch
import torch.nn as nn


class Diffuser(nn.Module):
    def __init__(
        self,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
    ):
        super(Diffuser, self).__init__()
        self.t = timesteps

        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.beta = nn.Parameter(self.beta, requires_grad=False)
        self.alpha = nn.Parameter(self.alpha, requires_grad=False)
        self.alpha_hat = nn.Parameter(self.alpha_hat, requires_grad=False)

    def forward(self, img, timesteps):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[timesteps])[:, None, None, None]
        minus_sqrt_alpha_hat = torch.sqrt(1 - self.alpha_hat[timesteps])[
            :, None, None, None
        ]
        noise = torch.randn_like(img)
        return sqrt_alpha_hat * img + minus_sqrt_alpha_hat * noise, noise

    def sample_step(self, x, predicted, timesteps):
        if type(timesteps) == int:
            timesteps = [timesteps]
        if timesteps[0] > 1:
            noise = torch.randn_like(predicted)
        else:
            noise = torch.zeros_like(predicted)
        a = self.alpha[timesteps[0]]
        a_h = self.alpha_hat[timesteps[0]]
        b = self.beta[timesteps[0]]
        return (
            1 / torch.sqrt(a) * (x - ((1 - a) / (torch.sqrt(1 - a_h))) * predicted)
            + torch.sqrt(b) * noise
        )


# class DiffuserTemp():
#     def __init__(
#         self,
#         timesteps=1000,
#         beta_start=1e-4,
#         beta_end=0.02,
#         device = "cuda"
#     ):
#         self.t = timesteps

#         self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
#         self.alpha = (1 - self.beta).to(device)
#         self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(device)

#     def forward(self, img, timesteps):
#         sqrt_alpha_hat = torch.sqrt(self.alpha_hat[timesteps])[:, None, None, None]
#         minus_sqrt_alpha_hat = torch.sqrt(1 - self.alpha_hat[timesteps])[
#             :, None, None, None
#         ]
#         noise = torch.randn_like(img)
#         return sqrt_alpha_hat * img + minus_sqrt_alpha_hat * noise, noise

#     def sample_step(self, x, predicted, timesteps):
#         if type(timesteps) == int:
#             timesteps = [timesteps]
#         if timesteps[0] > 1:
#             noise = torch.randn_like(predicted)
#         else:
#             noise = torch.zeros_like(predicted)
#         a = self.alpha[timesteps[0]]
#         a_h = self.alpha_hat[timesteps[0]]
#         b = self.beta[timesteps[0]]
#         return (
#             1
#             / torch.sqrt(a)
#             * (x - ((1 - a) / (torch.sqrt(1 - a_h))) * predicted)
#             + torch.sqrt(b) * noise
#         )
