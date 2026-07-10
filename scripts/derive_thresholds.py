"""Derive Phase 1 scorer thresholds from the Phase 0 background distributions.

Thresholds are NOT arbitrary: for each detector we score a background/normal
sample with the frozen Phase 0 artifacts and take a configurable percentile of
that score distribution, so the expected background keep-rate is
``(100 - percentile) / 100`` by construction.

* Stream A (physics): Sum mu^2 on held-out background events (file tail region
  0.9-1.0, disjoint from the training region) -- same sample the Phase 0 eval
  uses, so scorer output is directly comparable to ``physics_metrics.json``.
* Stream B (PDM): conv-AE reconstruction MSE on the normal-validation pulses of
  each system (train/val split re-derived with the Phase 0 seed, so the
  threshold is not biased by pulses the AE was fit on).

Writes ``configs/thresholds.json``. Run via ``make thresholds``.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import torch

from src.common.config import load_config, resolve_path
from src.common.device import describe_device, get_device, seed_everything
from src.inference.scores import ae_recon_error, vae_anomaly_score
from src.preprocessing import adc2021, hvcm

from scripts.eval_pdm import _load_ae
from scripts.eval_physics import _load_model as _load_vae


DEFAULTS: Dict[str, Any] = {
    "percentile": 99.0,
    "out_path": "configs/thresholds.json",
    "physics_config": "configs/physics_vae.yaml",
    "pdm_config": "configs/pdm_ae.yaml",
    "eval_events": 100_000,
    "seed": 1337,
    "device": None,
}


def derive_physics(cfg: Dict[str, Any], percentile: float, eval_events: int,
                   seed: int, device: torch.device) -> Dict[str, Any]:
    model_dir = resolve_path(cfg["model_dir"])
    normalizer = adc2021.Normalizer.load(model_dir / "norm.npz")
    model = _load_vae(model_dir, device)
    # Held-out tail region, matching scripts/eval_physics.py exactly.
    bg_raw = adc2021.load_events(cfg["background_path"], max_events=eval_events,
                                 seed=seed + 1, region=(0.9, 1.0))
    scores = vae_anomaly_score(model, torch.from_numpy(
        normalizer.transform(bg_raw)), device)
    thr = float(np.percentile(scores, percentile))
    print(f"[thresholds] physics: n={len(scores)} p{percentile:g}(Sum mu^2)={thr:.4f}")
    return {
        "score": "sum_mu2",
        "threshold": thr,
        "percentile": percentile,
        "n_background": int(len(scores)),
        "background_region": [0.9, 1.0],
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
    }


def derive_pdm(cfg: Dict[str, Any], percentile: float, seed: int,
               device: torch.device) -> Dict[str, Any]:
    data_dir = resolve_path(cfg["data_dir"])
    model_root = resolve_path(cfg["model_dir"])
    out: Dict[str, Any] = {}
    for system in cfg["systems"]:
        model_dir = model_root / system
        sys_data = hvcm.load_system(data_dir, system,
                                    target_len=cfg["target_len"],
                                    max_pulses=cfg.get("max_pulses"))
        # Normal-val pulses only (Phase 0 split, same seed) -> unbiased threshold.
        _, val_idx = hvcm.split_normal_train_val(
            sys_data, val_frac=cfg["val_frac"], seed=seed)
        norm = hvcm.ChannelNormalizer.load(model_dir / "channel_norm.npz")
        ae = _load_ae(model_dir, device)
        waves_cf = norm.transform_channels_first(sys_data.waves[val_idx])
        scores = ae_recon_error(ae, torch.from_numpy(waves_cf), device)
        thr = float(np.percentile(scores, percentile))
        print(f"[thresholds] pdm/{system}: n_normal_val={len(scores)} "
              f"p{percentile:g}(recon MSE)={thr:.6f}")
        out[system] = {
            "score": "ae_recon_mse",
            "threshold": thr,
            "percentile": percentile,
            "n_normal_val": int(len(scores)),
            "score_mean": float(scores.mean()),
            "score_median": float(np.median(scores)),
        }
    return out


def run(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    c = dict(DEFAULTS)
    if cfg:
        c.update({k: v for k, v in cfg.items() if v is not None})
    seed_everything(c["seed"])
    device = get_device(c["device"])
    print(f"[thresholds] device = {describe_device(device)}")

    phys_cfg = load_config(c["physics_config"])
    pdm_cfg = load_config(c["pdm_config"])
    pct = float(c["percentile"])

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "percentile": pct,
        "expected_background_keep_rate": (100.0 - pct) / 100.0,
        "physics": derive_physics(phys_cfg, pct, c["eval_events"], c["seed"],
                                  device),
        "pdm": derive_pdm(pdm_cfg, pct, c["seed"], device),
    }
    out_path = resolve_path(c["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[thresholds] wrote {out_path}")
    return result


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--percentile", type=float, default=None,
                   help="background percentile for the keep threshold (default 99)")
    p.add_argument("--eval-events", type=int, default=None)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--out", dest="out_path", type=str, default=None)
    args = p.parse_args(argv)
    run({k: v for k, v in vars(args).items()})


if __name__ == "__main__":
    main()
