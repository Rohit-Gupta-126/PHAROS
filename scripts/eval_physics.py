"""Evaluate the Stream A physics VAE: ROC/AUC of background vs a signal file.

Runnable via ``python -m scripts.eval_physics --config configs/physics_vae.yaml``
(Makefile target ``eval-physics``). Loads the trained checkpoint + normalizer,
scores a held-out background sample and a signal sample (default A->4l) with the
Sum mu^2 anomaly score, and writes to ``reports/phase0/``:

    physics_roc.png         - ROC curve with AUC
    physics_score_hist.png  - score distributions (background vs signal)
    physics_metrics.json    - AUC and summary stats
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve

from src.common.config import load_config, resolve_path
from src.common.device import describe_device, get_device, seed_everything
from src.inference.scores import vae_anomaly_score
from src.preprocessing.adc2021 import Normalizer, load_events
from src.training.vae import VAE


DEFAULTS: Dict[str, Any] = {
    "background_path": "data/raw/adc2021/background_for_training.h5",
    "signal_path": "data/raw/adc2021/Ato4l_lepFilter_13TeV_filtered.h5",
    "signal_name": "A->4l",
    "model_dir": "models/physics_vae",
    "reports_dir": "reports/phase0",
    "eval_events": 100_000,   # per class, subsampled
    "seed": 1337,
    "device": None,
}


def _merge(cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = dict(DEFAULTS)
    if cfg:
        merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged


def _load_model(model_dir: Path, device: torch.device) -> VAE:
    ckpt = torch.load(model_dir / "ckpt.pt", map_location=device)
    model = VAE(input_dim=ckpt["input_dim"], latent_dim=ckpt["latent_dim"],
                hidden=tuple(ckpt["hidden"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def run(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    c = _merge(cfg)
    seed_everything(c["seed"])
    device = get_device(c["device"])
    model_dir = resolve_path(c["model_dir"])
    reports_dir = resolve_path(c["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval-physics] device = {describe_device(device)}")
    normalizer = Normalizer.load(model_dir / "norm.npz")
    model = _load_model(model_dir, device)

    n = c["eval_events"]
    # Background from the held-out tail region (training uses the first 90%).
    bg_raw = load_events(c["background_path"], max_events=n, seed=c["seed"] + 1,
                         region=(0.9, 1.0))
    sig_raw = load_events(c["signal_path"], max_events=n, seed=c["seed"] + 2)
    bg = torch.from_numpy(normalizer.transform(bg_raw))
    sig = torch.from_numpy(normalizer.transform(sig_raw))

    bg_scores = vae_anomaly_score(model, bg, device)
    sig_scores = vae_anomaly_score(model, sig, device)

    y_true = np.concatenate([np.zeros(len(bg_scores)), np.ones(len(sig_scores))])
    y_score = np.concatenate([bg_scores, sig_scores])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))

    sig_name = c["signal_name"]
    print(f"[eval-physics] bg={len(bg_scores)} sig={len(sig_scores)} "
          f"AUC({sig_name})={roc_auc:.4f}")

    _plot_roc(fpr, tpr, roc_auc, sig_name, reports_dir / "physics_roc.png")
    _plot_hist(bg_scores, sig_scores, sig_name,
               reports_dir / "physics_score_hist.png")

    metrics = {
        "auc": roc_auc, "signal": sig_name,
        "n_background": int(len(bg_scores)), "n_signal": int(len(sig_scores)),
        "background_score_mean": float(bg_scores.mean()),
        "signal_score_mean": float(sig_scores.mean()),
        "background_score_median": float(np.median(bg_scores)),
        "signal_score_median": float(np.median(sig_scores)),
    }
    (reports_dir / "physics_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[eval-physics] wrote plots + metrics -> {reports_dir}")
    return metrics


def _plot_roc(fpr, tpr, roc_auc, sig_name, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{sig_name}  (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="chance")
    ax.set_xlabel("False positive rate (background)")
    ax.set_ylabel("True positive rate (signal)")
    ax.set_title("PHAROS Stream A VAE - ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_hist(bg_scores, sig_scores, sig_name, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    lo = min(bg_scores.min(), sig_scores.min())
    hi = max(np.quantile(bg_scores, 0.999), np.quantile(sig_scores, 0.999))
    bins = np.linspace(lo, hi, 80)
    ax.hist(bg_scores, bins=bins, density=True, alpha=0.6, label="background")
    ax.hist(sig_scores, bins=bins, density=True, alpha=0.6, label=sig_name)
    ax.set_xlabel(r"anomaly score  $\sum \mu^2$")
    ax.set_ylabel("density")
    ax.set_title("PHAROS Stream A VAE - score distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the Stream A physics VAE.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--eval-events", type=int, default=None)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    cfg = load_config(args.config) if args.config else {}
    if args.eval_events is not None:
        cfg["eval_events"] = args.eval_events
    if args.device is not None:
        cfg["device"] = args.device
    run(cfg)


if __name__ == "__main__":
    main()
