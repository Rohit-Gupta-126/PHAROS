"""1D convolutional autoencoder for HVCM waveforms (Stream B).

Input: (batch, 14, L) normalized waveforms (default L = 500). The encoder
compresses along the time axis with strided Conv1d blocks to a small bottleneck;
the decoder mirrors it with ConvTranspose1d. Anomaly score is the per-sample
reconstruction MSE (see ``src.inference.scores.ae_recon_error``).

The model is small enough to train on a 4 GB card at a modest batch size.
``L`` must be divisible by 8 (three stride-2 stages); 500 -> pad to 504 is
avoided by using L=500 with kernel/padding chosen so lengths stay aligned via
``output_padding``. We keep it simple and require L % 8 == 0-friendly handling by
cropping the decoder output back to L.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvAE(nn.Module):
    def __init__(self, n_channels: int = 14, seq_len: int = 500,
                 base: int = 16, latent_channels: int = 8) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len

        # Encoder: 3 stride-2 downsampling stages.
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, base, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(base, base * 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(base * 2, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # Decoder: mirror with transposed convs.
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, base * 2, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(base * 2, base, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(base, n_channels, kernel_size=7, stride=2,
                               padding=3, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        # Crop or pad back to the exact input length.
        if out.shape[-1] != x.shape[-1]:
            out = out[..., : x.shape[-1]]
        return out
