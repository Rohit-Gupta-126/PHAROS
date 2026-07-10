"""Shared utilities: config loading, device selection, seeding, paths."""
from .config import load_config, PROJECT_ROOT
from .device import get_device, autocast_ctx, seed_everything

__all__ = [
    "load_config",
    "PROJECT_ROOT",
    "get_device",
    "autocast_ctx",
    "seed_everything",
]
