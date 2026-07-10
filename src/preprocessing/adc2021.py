"""ADC2021 (Delphes L1 anomaly-detection challenge) data loading.

Each HDF5 file holds a dataset ``Particles`` of shape ``(N, 19, 4)`` where the
four columns are ``[pT, eta, phi, class]`` and the 19 object slots are a fixed
layout::

    row  0        : MET               (class 1)
    rows 1..4     : up to 4 electrons  (class 2)
    rows 5..8     : up to 4 muons      (class 3)
    rows 9..18    : up to 10 jets      (class 4)

Absent objects are zero-padded (class 0). We drop the class column and flatten
to a 57-dim feature vector (19 slots x [pT, eta, phi]), matching the standard
AXOL1TL / CICADA flat event representation.

Normalization: pT is heavy-tailed, so we ``log1p`` it before standardizing;
eta/phi are standardized directly. Statistics are fit on the training split only
and saved alongside the model checkpoint. Constant slots (MET-eta is always 0,
padded objects are all-zero) have ~0 std and are guarded to map to 0.

See ``docs/data_schema_adc2021.md`` for the full inferred schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np

N_SLOTS = 19          # 1 MET + 4 e + 4 mu + 10 jet
N_KIN = 3             # pT, eta, phi (class column dropped)
N_FEATURES = N_SLOTS * N_KIN  # 57

# Human-readable feature names, in flattened order.
_SLOT_NAMES = (
    ["MET"]
    + [f"e{i}" for i in range(1, 5)]
    + [f"mu{i}" for i in range(1, 5)]
    + [f"jet{i}" for i in range(1, 11)]
)
FEATURE_NAMES = [f"{s}_{k}" for s in _SLOT_NAMES for k in ("pt", "eta", "phi")]

# Column index of pT within each (pt, eta, phi) triple, for the log transform.
_PT_COLUMNS = np.arange(0, N_FEATURES, N_KIN)


def load_events(path: str | Path, max_events: int | None = None,
                seed: int = 1337,
                region: Tuple[float, float] = (0.0, 1.0),
                chunk_rows: int = 1_000_000) -> np.ndarray:
    """Load raw kinematics as ``(n, 57)`` float32 (class column dropped).

    ``region`` restricts sampling to a fractional slice of the file, e.g.
    ``(0.0, 0.9)`` for the first 90% of events; this lets training and
    evaluation draw from disjoint regions. Within the region, if ``max_events``
    is smaller than what is available, an evenly-spaced (strided) subsample is
    taken.

    The file is read in contiguous row chunks (each cast to float32 immediately),
    so peak memory stays ~1 GB regardless of ``max_events`` -- important on the
    memory-constrained (8 GB) WSL target. ``seed`` is accepted for API symmetry;
    strided sampling is deterministic.
    """
    path = Path(path)
    with h5py.File(path, "r") as h:
        particles = h["Particles"]
        total = particles.shape[0]
        lo = int(total * region[0])
        hi = int(total * region[1])
        avail = hi - lo
        step = max(1, avail // max_events) if max_events else 1

        parts: list[np.ndarray] = []
        collected = 0
        for cstart in range(lo, hi, chunk_rows):
            if max_events is not None and collected >= max_events:
                break
            cend = min(cstart + chunk_rows, hi)
            block = np.asarray(particles[cstart:cend, :, :N_KIN],
                               dtype=np.float32)
            if step > 1:
                # Keep global stride alignment across chunk boundaries.
                offset = (-(cstart - lo)) % step
                block = block[offset::step]
            block = block.reshape(-1, N_FEATURES)
            if max_events is not None:
                take = min(len(block), max_events - collected)
                block = block[:take]
            parts.append(block)
            collected += len(block)

    return np.concatenate(parts, axis=0) if parts else np.empty((0, N_FEATURES),
                                                                np.float32)


@dataclass
class Normalizer:
    """Per-feature log1p(pT) + z-score standardizer, fit on training data."""

    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    @staticmethod
    def _log_pt(x: np.ndarray) -> np.ndarray:
        out = x.copy()
        out[:, _PT_COLUMNS] = np.log1p(np.clip(out[:, _PT_COLUMNS], 0, None))
        return out

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-6) -> "Normalizer":
        xl = cls._log_pt(x)
        mean = xl.mean(axis=0)
        std = xl.std(axis=0)
        # Guard constant/near-constant features (MET-eta, padded slots) -> map to 0.
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32), eps=eps)

    def transform(self, x: np.ndarray) -> np.ndarray:
        xl = self._log_pt(np.asarray(x, dtype=np.float32))
        return ((xl - self.mean) / self.std).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std, eps=np.float32(self.eps))

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        d = np.load(Path(path))
        return cls(mean=d["mean"], std=d["std"], eps=float(d["eps"]))


def train_val_split(x: np.ndarray, val_frac: float = 0.1,
                    seed: int = 1337) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic row split into (train, val)."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return x[train_idx], x[val_idx]


def make_datasets(background_path: str | Path, max_events: int | None,
                  val_frac: float = 0.1, seed: int = 1337,
                  region: Tuple[float, float] = (0.0, 0.9)
                  ) -> Tuple[np.ndarray, np.ndarray, Normalizer]:
    """Load background, split, fit normalizer on train, return normalized arrays.

    Training samples the first 90% of the file by default (``region``), leaving
    the tail for an approximately held-out evaluation background. Returns
    ``(x_train, x_val, normalizer)`` as normalized float32 arrays.
    """
    raw = load_events(background_path, max_events=max_events, seed=seed,
                      region=region)
    raw_train, raw_val = train_val_split(raw, val_frac=val_frac, seed=seed)
    del raw
    normalizer = Normalizer.fit(raw_train)
    return normalizer.transform(raw_train), normalizer.transform(raw_val), normalizer


def load_signal(path: str | Path, normalizer: Normalizer,
                max_events: int | None = None, seed: int = 1337) -> np.ndarray:
    """Load and normalize a signal file using an existing (background) normalizer."""
    raw = load_events(path, max_events=max_events, seed=seed)
    return normalizer.transform(raw)
