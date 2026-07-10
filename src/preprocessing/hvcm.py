"""SNS HVCM (high-voltage converter modulator) waveform loading (Stream B).

Each system (RFQ / DTL / CCL / SCL) is stored as::

    {system}.npy         X: (pulses, 4500, 14) float32  -- 14 expert waveforms
    {system}_labels.npy  Y: (pulses, 3) object -- [file, state, fault_type]

``state`` is "Run" (normal) or "Fault"; ``fault_type`` is "Normal" for run
pulses or a specific fault-class string for faults.

We downsample the 4500-sample waveforms to ``target_len`` (default 500) by
average pooling, done in row chunks so the full (up to 4598x4500x14 ~ 3.6 GB)
array is never materialized at once. Two views are produced:

* ``waveforms`` (N, 14, target_len) channels-first, for the conv autoencoder;
* ``features``  (N, 14*n_stats) per-channel summary statistics, for the
  IsolationForest baseline.

Per-channel normalization statistics are fit on the *normal training* pulses
only and applied to everything else.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SYSTEMS = ["RFQ", "DTL", "CCL", "SCL"]
N_CHANNELS = 14
RAW_LEN = 4500
CHANNEL_NAMES = [
    "A+IGBT-I", "A+*IGBT-I", "B+IGBT-I", "B+*IGBT-I", "C+IGBT-I", "C+*IGBT-I",
    "A-FLUX", "B-FLUX", "C-FLUX", "MOD-V", "MOD-I", "CB-I", "CB-V", "DV/DT",
]
# Per-channel summary statistics used for the IsolationForest baseline.
STAT_NAMES = ["mean", "std", "min", "max", "ptp", "abs_energy"]


def system_path(data_dir: str | Path, system: str) -> Tuple[Path, Path]:
    d = Path(data_dir)
    return d / f"{system}.npy", d / f"{system}_labels.npy"


def _avg_pool_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    """Average-pool along axis=1 of (n, RAW_LEN, C) -> (n, target_len, C)."""
    n, length, c = x.shape
    factor = length // target_len
    usable = factor * target_len
    x = x[:, :usable, :]
    x = x.reshape(n, target_len, factor, c).mean(axis=2)
    return x


def _summary_features(wave_nlc: np.ndarray) -> np.ndarray:
    """Per-channel summary stats for (n, L, C) -> (n, C*len(STAT_NAMES))."""
    mean = wave_nlc.mean(axis=1)
    std = wave_nlc.std(axis=1)
    mn = wave_nlc.min(axis=1)
    mx = wave_nlc.max(axis=1)
    ptp = mx - mn
    energy = np.mean(wave_nlc ** 2, axis=1)
    feats = np.stack([mean, std, mn, mx, ptp, energy], axis=2)  # (n, C, n_stats)
    return feats.reshape(wave_nlc.shape[0], -1)


@dataclass
class HVCMSystem:
    """Loaded, downsampled data for one HVCM system."""

    system: str
    waves: np.ndarray          # (N, target_len, C), downsampled, unnormalized
    features: np.ndarray       # (N, C*n_stats)
    is_fault: np.ndarray       # (N,) bool
    fault_type: np.ndarray     # (N,) str
    target_len: int

    @property
    def normal_mask(self) -> np.ndarray:
        return ~self.is_fault

    def fault_classes(self) -> List[str]:
        return sorted({t for t, f in zip(self.fault_type, self.is_fault) if f})


def load_system(data_dir: str | Path, system: str, target_len: int = 500,
                chunk: int = 256, max_pulses: int | None = None) -> HVCMSystem:
    xpath, ypath = system_path(data_dir, system)
    y = np.load(ypath, allow_pickle=True)
    state = y[:, 1].astype(str)
    ftype = y[:, 2].astype(str)
    is_fault = state == "Fault"

    x = np.load(xpath, mmap_mode="r")
    n = x.shape[0]
    if max_pulses is not None and max_pulses < n:
        n = max_pulses
        is_fault = is_fault[:n]
        ftype = ftype[:n]

    waves_ds: List[np.ndarray] = []
    feats: List[np.ndarray] = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = np.asarray(x[start:end], dtype=np.float32)  # (b, 4500, 14)
        ds = _avg_pool_1d(block, target_len)                # (b, target_len, 14)
        waves_ds.append(ds)
        feats.append(_summary_features(ds))
    waves = np.concatenate(waves_ds, axis=0)
    features = np.concatenate(feats, axis=0)
    return HVCMSystem(system=system, waves=waves, features=features,
                      is_fault=is_fault, fault_type=ftype, target_len=target_len)


@dataclass
class ChannelNormalizer:
    """Per-channel z-score over the (L) axis, fit on normal-train waveforms."""

    mean: np.ndarray   # (C,)
    std: np.ndarray    # (C,)

    @classmethod
    def fit(cls, waves_nlc: np.ndarray, eps: float = 1e-6) -> "ChannelNormalizer":
        mean = waves_nlc.mean(axis=(0, 1))
        std = waves_nlc.std(axis=(0, 1))
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform_channels_first(self, waves_nlc: np.ndarray) -> np.ndarray:
        """Normalize (N, L, C) and return channels-first (N, C, L) float32."""
        norm = (waves_nlc - self.mean) / self.std
        return np.transpose(norm, (0, 2, 1)).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str | Path) -> "ChannelNormalizer":
        d = np.load(Path(path))
        return cls(mean=d["mean"], std=d["std"])


def split_normal_train_val(sys_data: HVCMSystem, val_frac: float = 0.15,
                           seed: int = 1337) -> Tuple[np.ndarray, np.ndarray]:
    """Row indices into ``sys_data`` for normal train / normal val pulses."""
    rng = np.random.default_rng(seed)
    normal_idx = np.where(sys_data.normal_mask)[0]
    rng.shuffle(normal_idx)
    n_val = int(round(len(normal_idx) * val_frac))
    return normal_idx[n_val:], normal_idx[:n_val]
