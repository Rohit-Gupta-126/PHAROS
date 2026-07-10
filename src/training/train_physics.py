"""Train the Stream A VAE unsupervised on ADC2021 background events.

Runnable via ``python -m src.training.train_physics --config configs/physics_vae.yaml``
(Makefile target ``train-physics``). Exposes ``run(cfg)`` so the smoke tests can
drive a tiny CPU configuration directly.

Outputs (default ``models/physics_vae/``):
    ckpt.pt      - model weights + config
    norm.npz     - normalizer statistics (fit on the training split)
    config.json  - resolved config
    history.json - per-epoch train/val losses
    model.onnx   - optional ONNX export (--onnx)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.common.config import load_config, resolve_path
from src.common.device import (autocast_ctx, describe_device, get_device,
                               seed_everything)
from src.preprocessing.adc2021 import N_FEATURES, make_datasets
from src.training.vae import VAE, vae_loss


DEFAULTS: Dict[str, Any] = {
    "background_path": "data/raw/adc2021/background_for_training.h5",
    "out_dir": "models/physics_vae",
    "max_events": 2_000_000,
    "val_frac": 0.1,
    "latent_dim": 8,
    "hidden": [32, 16],
    "epochs": 20,
    "batch_size": 1024,
    "lr": 1e-3,
    "beta": 1.0,
    "kl_warmup_epochs": 5,
    "seed": 1337,
    "device": None,       # None -> auto (cuda if available)
    "amp": True,
    "num_workers": 0,
    "export_onnx": False,
    "log_every": 1,
}


def _merge(cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = dict(DEFAULTS)
    if cfg:
        merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged


def run(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    c = _merge(cfg)
    seed_everything(c["seed"])
    device = get_device(c["device"])
    out_dir = resolve_path(c["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train-physics] device = {describe_device(device)}")
    print(f"[train-physics] loading background (max_events={c['max_events']}) ...")
    x_train, x_val, normalizer = make_datasets(
        resolve_path(c["background_path"]), max_events=c["max_events"],
        val_frac=c["val_frac"], seed=c["seed"],
    )
    print(f"[train-physics] train={x_train.shape} val={x_val.shape}")
    normalizer.save(out_dir / "norm.npz")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train)),
        batch_size=c["batch_size"], shuffle=True,
        num_workers=c["num_workers"], drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    x_val_t = torch.from_numpy(x_val)

    model = VAE(input_dim=N_FEATURES, latent_dim=c["latent_dim"],
                hidden=tuple(c["hidden"])).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=c["lr"])
    use_amp = bool(c["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    for epoch in range(1, c["epochs"] + 1):
        # Linear KL warm-up avoids early posterior collapse (Sum mu^2 -> 0).
        warm = c["kl_warmup_epochs"]
        beta = c["beta"] * (min(epoch, warm) / warm if warm > 0 else 1.0)

        model.train()
        t0 = time.time()
        run_tot = run_rec = run_kl = 0.0
        n_batches = 0
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with autocast_ctx(device, enabled=use_amp):
                recon, mu, logvar = model(batch)
                total, rec, kl = vae_loss(recon, batch, mu, logvar, beta=beta)
            scaler.scale(total).backward()
            scaler.step(optim)
            scaler.update()
            run_tot += total.item(); run_rec += rec.item(); run_kl += kl.item()
            n_batches += 1
        train_loss = run_tot / max(n_batches, 1)

        # Validation (recon + KL at full beta for a comparable number).
        model.eval()
        with torch.no_grad():
            vb = x_val_t.to(device)
            recon, mu, logvar = model(vb)
            val_total, val_rec, val_kl = vae_loss(recon, vb, mu, logvar,
                                                  beta=c["beta"])
            val_mu2 = float(torch.mean(torch.sum(mu * mu, dim=1)))
        dt = time.time() - t0

        rec_entry = {
            "epoch": epoch, "beta": round(beta, 4),
            "train_loss": train_loss,
            "train_recon": run_rec / max(n_batches, 1),
            "train_kl": run_kl / max(n_batches, 1),
            "val_loss": float(val_total), "val_recon": float(val_rec),
            "val_kl": float(val_kl), "val_mean_mu2": val_mu2, "sec": round(dt, 2),
        }
        history.append(rec_entry)
        if epoch % c["log_every"] == 0 or epoch == c["epochs"]:
            print(f"[train-physics] epoch {epoch:3d}/{c['epochs']} "
                  f"train={train_loss:.4f} val={float(val_total):.4f} "
                  f"(rec={float(val_rec):.4f} kl={float(val_kl):.4f}) "
                  f"mean_mu2={val_mu2:.3f} beta={beta:.3f} {dt:.1f}s")

    # Save artifacts.
    ckpt = {
        "state_dict": model.state_dict(),
        "input_dim": N_FEATURES,
        "latent_dim": c["latent_dim"],
        "hidden": list(c["hidden"]),
    }
    torch.save(ckpt, out_dir / "ckpt.pt")
    (out_dir / "config.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2),
                                          encoding="utf-8")

    if c["export_onnx"]:
        _export_onnx(model, device, out_dir / "model.onnx")

    print(f"[train-physics] saved checkpoint -> {out_dir / 'ckpt.pt'}")
    first, last = history[0]["train_loss"], history[-1]["train_loss"]
    print(f"[train-physics] train loss {first:.4f} -> {last:.4f}")
    return {"out_dir": str(out_dir), "history": history,
            "final_train_loss": last, "first_train_loss": first}


def _export_onnx(model: VAE, device: torch.device, path: Path) -> None:
    model.eval()
    dummy = torch.zeros(1, model.input_dim, device=device)
    torch.onnx.export(
        model, dummy, str(path), input_names=["event"],
        output_names=["recon", "mu", "logvar"],
        dynamic_axes={"event": {0: "batch"}, "recon": {0: "batch"},
                      "mu": {0: "batch"}, "logvar": {0: "batch"}},
        opset_version=17,
    )
    print(f"[train-physics] exported ONNX -> {path}")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Stream A physics VAE.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--onnx", action="store_true", help="export ONNX after training")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    cfg = load_config(args.config) if args.config else {}
    for key in ("max_events", "epochs", "batch_size", "device"):
        val = getattr(args, key.replace("-", "_"))
        if val is not None:
            cfg[key] = val
    if args.onnx:
        cfg["export_onnx"] = True
    run(cfg)


if __name__ == "__main__":
    main()
