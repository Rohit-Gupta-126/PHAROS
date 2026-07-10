"""Train the Stream B predictive-maintenance detectors on HVCM waveforms.

For each requested system (RFQ / DTL / CCL / SCL) we train, on *normal* pulses
only:

* a 1D conv autoencoder (primary) -> anomaly score = reconstruction MSE;
* an IsolationForest (baseline) on per-channel summary features.

Runnable via ``python -m src.training.train_pdm --config configs/pdm_ae.yaml``
(Makefile target ``train-pdm``). Exposes ``run(cfg)`` for the smoke tests.

Artifacts per system (default ``models/pdm/{system}/``):
    ae.pt          - conv AE weights + shape metadata
    channel_norm.npz - per-channel normalizer (fit on normal-train)
    iso.joblib     - IsolationForest + feature scaler
    history.json   - AE per-epoch losses
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.common.config import load_config, resolve_path
from src.common.device import (autocast_ctx, describe_device, get_device,
                               seed_everything)
from src.preprocessing.hvcm import (SYSTEMS, ChannelNormalizer, load_system,
                                    split_normal_train_val)
from src.training.conv_ae import ConvAE


DEFAULTS: Dict[str, Any] = {
    "data_dir": "data/raw/hvcm",
    "out_dir": "models/pdm",
    "systems": SYSTEMS,
    "target_len": 500,
    "max_pulses": None,
    "val_frac": 0.15,
    "base": 16,
    "latent_channels": 8,
    "epochs": 30,
    "batch_size": 32,
    "lr": 1e-3,
    "iso_estimators": 200,
    "iso_contamination": "auto",
    "seed": 1337,
    "device": None,
    "amp": True,
    "log_every": 5,
}


def _merge(cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = dict(DEFAULTS)
    if cfg:
        merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged


def _train_conv_ae(waves_cf: np.ndarray, val_cf: np.ndarray, c: Dict[str, Any],
                   device: torch.device) -> tuple[ConvAE, List[dict]]:
    seq_len = waves_cf.shape[-1]
    model = ConvAE(n_channels=waves_cf.shape[1], seq_len=seq_len,
                   base=c["base"], latent_channels=c["latent_channels"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=c["lr"])
    loss_fn = torch.nn.MSELoss()
    use_amp = bool(c["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(waves_cf)),
        batch_size=c["batch_size"], shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    val_t = torch.from_numpy(val_cf).to(device) if len(val_cf) else None

    history = []
    for epoch in range(1, c["epochs"] + 1):
        model.train()
        run_loss = 0.0
        nb = 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with autocast_ctx(device, enabled=use_amp):
                recon = model(batch)
                loss = loss_fn(recon, batch)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            run_loss += loss.item()
            nb += 1
        train_loss = run_loss / max(nb, 1)
        val_loss = float("nan")
        if val_t is not None and len(val_t):
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(val_t), val_t))
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss})
        if epoch % c["log_every"] == 0 or epoch == c["epochs"]:
            print(f"    AE epoch {epoch:3d}/{c['epochs']} "
                  f"train={train_loss:.5f} val={val_loss:.5f}")
    return model, history


def run(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    c = _merge(cfg)
    seed_everything(c["seed"])
    device = get_device(c["device"])
    data_dir = resolve_path(c["data_dir"])
    out_root = resolve_path(c["out_dir"])
    print(f"[train-pdm] device = {describe_device(device)}")

    results: Dict[str, Any] = {}
    for system in c["systems"]:
        print(f"[train-pdm] === {system} ===")
        t0 = time.time()
        sys_data = load_system(data_dir, system, target_len=c["target_len"],
                               max_pulses=c["max_pulses"])
        train_idx, val_idx = split_normal_train_val(
            sys_data, val_frac=c["val_frac"], seed=c["seed"])
        print(f"[train-pdm] {system}: {len(sys_data.is_fault)} pulses, "
              f"{len(train_idx)} normal-train, {len(val_idx)} normal-val, "
              f"{int(sys_data.is_fault.sum())} faults")

        # --- Conv AE (channel-normalized waveforms) ---
        norm = ChannelNormalizer.fit(sys_data.waves[train_idx])
        train_cf = norm.transform_channels_first(sys_data.waves[train_idx])
        val_cf = norm.transform_channels_first(sys_data.waves[val_idx])
        model, history = _train_conv_ae(train_cf, val_cf, c, device)

        # --- IsolationForest baseline (summary features) ---
        scaler = StandardScaler().fit(sys_data.features[train_idx])
        iso = IsolationForest(
            n_estimators=c["iso_estimators"],
            contamination=c["iso_contamination"], random_state=c["seed"],
        ).fit(scaler.transform(sys_data.features[train_idx]))

        # --- Save artifacts ---
        out_dir = out_root / system
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "n_channels": sys_data.waves.shape[2],
            "seq_len": c["target_len"], "base": c["base"],
            "latent_channels": c["latent_channels"],
        }, out_dir / "ae.pt")
        norm.save(out_dir / "channel_norm.npz")
        joblib.dump({"iso": iso, "scaler": scaler}, out_dir / "iso.joblib")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2),
                                              encoding="utf-8")

        dt = time.time() - t0
        results[system] = {
            "out_dir": str(out_dir),
            "final_train_loss": history[-1]["train_loss"],
            "first_train_loss": history[0]["train_loss"],
            "n_faults": int(sys_data.is_fault.sum()), "sec": round(dt, 1),
        }
        print(f"[train-pdm] {system}: saved -> {out_dir}  ({dt:.1f}s)")
    return results


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stream B PDM detectors.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--systems", type=str, default=None,
                   help="comma-separated subset, e.g. RFQ,DTL")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-pulses", type=int, default=None)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    cfg = load_config(args.config) if args.config else {}
    if args.systems:
        cfg["systems"] = [s.strip() for s in args.systems.split(",") if s.strip()]
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.max_pulses is not None:
        cfg["max_pulses"] = args.max_pulses
    if args.device is not None:
        cfg["device"] = args.device
    run(cfg)


if __name__ == "__main__":
    main()
