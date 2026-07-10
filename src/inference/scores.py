"""Anomaly scores for the two PHAROS detectors.

Stream A (VAE): score = sum of squared latent means (Sum mu^2), the AXOL1TL
convention. Higher = more anomalous.

Stream B (conv AE): score = per-sample reconstruction MSE. Higher = more
anomalous.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def vae_anomaly_score(model, x: torch.Tensor, device: torch.device,
                      batch_size: int = 8192) -> np.ndarray:
    """Sum mu^2 anomaly score for each row of ``x`` (a normalized (N, D) tensor)."""
    model.eval()
    scores = []
    for start in range(0, x.shape[0], batch_size):
        batch = x[start:start + batch_size].to(device)
        mu, _ = model.encode(batch)
        scores.append(torch.sum(mu * mu, dim=1).cpu().numpy())
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)


@torch.no_grad()
def ae_recon_error(model, x: torch.Tensor, device: torch.device,
                   batch_size: int = 256) -> np.ndarray:
    """Mean squared reconstruction error per sample for a conv AE.

    ``x`` has shape (N, C, L). Error is averaged over channels and length.
    """
    model.eval()
    scores = []
    for start in range(0, x.shape[0], batch_size):
        batch = x[start:start + batch_size].to(device)
        recon = model(batch)
        err = torch.mean((recon - batch) ** 2, dim=(1, 2))
        scores.append(err.cpu().numpy())
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)
