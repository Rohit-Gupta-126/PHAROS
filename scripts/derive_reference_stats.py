"""Derive the Phase 3 drift-monitor REFERENCE distributions from Phase 0.

For each stream the reference is built from the same frozen artifacts and the
same held-out samples the Phase 1 thresholds were derived on -- the monitor
then compares live sliding windows against these frozen references.

* Physics: Sum mu^2 score distribution on held-out background (file region
  0.9-1.0, seed-matched to derive_thresholds) + the raw (pre-normalization)
  input-feature distributions for the indices in ``configs/monitor.yaml``.
  -> ``models/physics_vae/reference_stats.json``
* PDM: per-system conv-AE recon-MSE score distribution on normal-validation
  pulses + per-channel-mean distributions for the tracked system/channels.
  -> ``models/pdm/reference_stats.json``

Each entry is a quantile-binned histogram (for PSI) plus a downsampled raw
sample (for KS / dashboard overlay). Run via ``make reference-stats``.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import torch
import yaml

from src.common.config import load_config, resolve_path
from src.common.device import describe_device, get_device, seed_everything
from src.inference.scores import ae_recon_error, vae_anomaly_score
from src.preprocessing import adc2021, hvcm

from scripts.eval_pdm import _load_ae
from scripts.eval_physics import _load_model as _load_vae
from services.monitor.drift_stats import ReferenceDist

SCHEMA = "pharos.reference_stats.v1"
SEED = 1337          # must match derive_thresholds (same held-out samples)
EVAL_EVENTS = 100_000


def _entry(name: str, samples: np.ndarray, sample_size: int) -> Dict[str, Any]:
    ref = ReferenceDist.from_samples(name, samples)
    keep = np.asarray(samples, dtype=np.float64).ravel()
    if keep.size > sample_size:
        keep = np.random.default_rng(SEED).choice(keep, sample_size,
                                                  replace=False)
    return {**ref.to_dict(), "samples": np.round(keep, 8).tolist()}


def derive_physics(phys_cfg: Dict[str, Any], mon: Dict[str, Any],
                   device: torch.device) -> Dict[str, Any]:
    model_dir = resolve_path(phys_cfg["model_dir"])
    normalizer = adc2021.Normalizer.load(model_dir / "norm.npz")
    model = _load_vae(model_dir, device)
    bg_raw = adc2021.load_events(phys_cfg["background_path"],
                                 max_events=EVAL_EVENTS, seed=SEED + 1,
                                 region=(0.9, 1.0))
    scores = vae_anomaly_score(model, torch.from_numpy(
        normalizer.transform(bg_raw)), device)
    n_keep = int(mon["reference_sample_size"])
    features = {}
    for idx in mon["physics_feature_indices"]:
        features[f"f{idx:02d}"] = _entry(f"f{idx:02d}", bg_raw[:, idx], n_keep)
    print(f"[ref-stats] physics: n={len(scores)} score_mean={scores.mean():.6g} "
          f"features={list(features)}")
    out = {"score": _entry("score", scores, n_keep), "features": features,
           "n_background": int(len(scores))}
    path = model_dir / "reference_stats.json"
    path.write_text(json.dumps({"schema": SCHEMA, "stream": "physics",
                                "generated_at": _now(), **out}, indent=2),
                    encoding="utf-8")
    print(f"[ref-stats] wrote {path}")
    return out


def derive_pdm(pdm_cfg: Dict[str, Any], mon: Dict[str, Any],
               device: torch.device) -> Dict[str, Any]:
    data_dir = resolve_path(pdm_cfg["data_dir"])
    model_root = resolve_path(pdm_cfg["model_dir"])
    n_keep = int(mon["reference_sample_size"])
    tracked_system = mon["pdm_system"]
    systems: Dict[str, Any] = {}
    for system in pdm_cfg["systems"]:
        model_dir = model_root / system
        sys_data = hvcm.load_system(data_dir, system,
                                    target_len=pdm_cfg["target_len"],
                                    max_pulses=pdm_cfg.get("max_pulses"))
        _, val_idx = hvcm.split_normal_train_val(
            sys_data, val_frac=pdm_cfg["val_frac"], seed=SEED)
        norm = hvcm.ChannelNormalizer.load(model_dir / "channel_norm.npz")
        ae = _load_ae(model_dir, device)
        waves = sys_data.waves[val_idx]                    # (n, L, C) raw
        scores = ae_recon_error(ae, torch.from_numpy(
            norm.transform_channels_first(waves)), device)
        entry: Dict[str, Any] = {"score": _entry("score", scores, n_keep),
                                 "n_normal_val": int(len(scores))}
        if system == tracked_system:
            ch_means = waves.mean(axis=1)                  # (n, C) raw means
            entry["channel_means"] = {
                f"ch{j:02d}": _entry(f"ch{j:02d}", ch_means[:, j], n_keep)
                for j in mon["pdm_channel_indices"]}
        systems[system] = entry
        print(f"[ref-stats] pdm/{system}: n_normal_val={len(scores)}"
              + (f" channels={mon['pdm_channel_indices']}"
                 if system == tracked_system else ""))
    path = model_root / "reference_stats.json"
    path.write_text(json.dumps({"schema": SCHEMA, "stream": "pdm",
                                "generated_at": _now(),
                                "tracked_system": tracked_system,
                                "systems": systems}, indent=2),
                    encoding="utf-8")
    print(f"[ref-stats] wrote {path}")
    return systems


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--monitor-config", type=str, default="configs/monitor.yaml")
    p.add_argument("--physics-config", type=str, default="configs/physics_vae.yaml")
    p.add_argument("--pdm-config", type=str, default="configs/pdm_ae.yaml")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--only", choices=["physics", "pdm"], default=None,
                   help="derive just one stream's reference")
    args = p.parse_args(argv)

    seed_everything(SEED)
    device = get_device(args.device)
    print(f"[ref-stats] device = {describe_device(device)}")
    mon = yaml.safe_load(resolve_path(args.monitor_config).read_text())

    if args.only in (None, "physics"):
        derive_physics(load_config(args.physics_config), mon, device)
    if args.only in (None, "pdm"):
        derive_pdm(load_config(args.pdm_config), mon, device)


if __name__ == "__main__":
    main()
