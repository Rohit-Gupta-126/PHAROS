"""Device selection, deterministic seeding, and AMP autocast helper.

Everything is GPU-accelerated when CUDA is available and falls back to CPU so
the same code runs unchanged in CI. ``get_device`` accepts an explicit override
(used by the smoke tests to force CPU).
"""
from __future__ import annotations

import contextlib
import os
import random

import numpy as np
import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return a torch device.

    ``prefer`` may be "cpu", "cuda", or None (auto: cuda if available else cpu).
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 2)
        return f"cuda:{idx} ({name}, {total:.0f} MiB)"
    return "cpu"


def seed_everything(seed: int = 1337) -> None:
    """Seed python, numpy, and torch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def autocast_ctx(device: torch.device, enabled: bool = True):
    """Autocast only on CUDA; a no-op context elsewhere."""
    if device.type == "cuda" and enabled:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield
