"""Small MLP variational autoencoder for the physics stream (Stream A).

Architecture (defaults): encoder 57 -> 32 -> 16 -> (mu[8], logvar[8]);
decoder 8 -> 16 -> 32 -> 57. The anomaly score is the sum of squared latent
means (Sum mu^2), following the AXOL1TL / CICADA convention, so the training
objective must keep mu informative (avoid posterior collapse) -- we use a small
KL weight with optional warm-up.

The model is intentionally tiny (a few thousand parameters), so it trains
comfortably on a 4 GB card at batch 1024 with no gradient checkpointing.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(dims: List[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class VAE(nn.Module):
    def __init__(self, input_dim: int = 57, latent_dim: int = 8,
                 hidden: Tuple[int, ...] = (32, 16)) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        enc_dims = [input_dim, *hidden]
        self.encoder = _mlp(enc_dims)
        self.fc_mu = nn.Linear(hidden[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden[-1], latent_dim)
        dec_dims = [latent_dim, *reversed(hidden), input_dim]
        self.decoder = _mlp(dec_dims)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon: torch.Tensor, target: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
    """Return (total, recon_mse, kl) with mean reduction over the batch."""
    recon_mse = F.mse_loss(recon, target, reduction="mean")
    # KL(q(z|x) || N(0, I)) averaged over the batch.
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total = recon_mse + beta * kl
    return total, recon_mse, kl
