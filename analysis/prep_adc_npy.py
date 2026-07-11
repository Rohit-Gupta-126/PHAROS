"""Prep ADC2021 observables + VAE scores for the RDataFrame analysis (Phase 4).

Runs on the HOST (pharos env: torch + h5py + sklearn). It scores background vs
the A->4l signal with the frozen VAE -- reusing the exact Phase 0 scoring path
(``scripts.eval_physics``) -- and extracts a handful of physics observables
straight from the 57-vector, then writes a columnar ``.npz`` that the ROOT
container's ``analysis/physics_rdf.py`` turns into RDataFrame histograms.

It also computes the per-signal ROC/AUC table for the README here (AUC needs
sklearn/torch, which live on the host, not in the ROOT image), quoting MEDIANS
for the heavy-tailed scores (recon-MSE means are outlier-dominated).

Outputs:
    data/interim/adc_obs.npz            (columns for RDataFrame)
    reports/phase4/physics_auc_table.json / .md

Run: ``python -m analysis.prep_adc_npy`` (uses the active model pointer).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.config import resolve_path
from src.common.device import get_device, seed_everything
from src.inference.scores import vae_anomaly_score, vae_recon_error
from src.preprocessing.adc2021 import Normalizer, load_events
from scripts.eval_physics import _auc, _load_model

# 57-vector indices (slot * 3 + kin). See src/preprocessing/nanoaod.py.
MET_PT = 0
E1_PT = 1 * 3
MU1_PT = 5 * 3
JET_PT_IDX = [s * 3 for s in range(9, 19)]   # 10 jet pT slots


def _observables(raw: np.ndarray) -> dict:
    """Physics observables straight from the raw (N,57) kinematics."""
    jet_pt = raw[:, JET_PT_IDX]
    return {
        "met_pt": raw[:, MET_PT].astype(np.float64),
        "lead_ele_pt": raw[:, E1_PT].astype(np.float64),
        "lead_mu_pt": raw[:, MU1_PT].astype(np.float64),
        "n_jet": (jet_pt > 0).sum(axis=1).astype(np.float64),
        "ht": jet_pt.sum(axis=1).astype(np.float64),
    }


def _active_model_dir(default: str) -> Path:
    """Follow the hot-swap pointer if present, else the default dir."""
    ptr = resolve_path("models/physics_vae/current.json")
    if ptr.exists():
        d = json.loads(ptr.read_text())
        md = resolve_path(d["model_dir"])
        if (md / "ckpt.pt").exists():
            return md
    return resolve_path(default)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--background", default="data/raw/adc2021/background_for_training.h5")
    p.add_argument("--signal", default="data/raw/adc2021/Ato4l_lepFilter_13TeV_filtered.h5")
    p.add_argument("--signal-name", default="A->4l")
    p.add_argument("--model-dir", default="models/physics_vae")
    p.add_argument("--eval-events", type=int, default=100_000)
    p.add_argument("--out-npz", default="data/interim/adc_obs.npz")
    p.add_argument("--reports-dir", default="reports/phase4")
    args = p.parse_args(argv)

    seed_everything(1337)
    device = get_device(None)
    model_dir = _active_model_dir(args.model_dir)
    print(f"[prep-adc] scoring with {model_dir} on {device}")
    normalizer = Normalizer.load(model_dir / "norm.npz")
    model = _load_model(model_dir, device)

    n = args.eval_events
    bg_raw = load_events(args.background, max_events=n, seed=1338, region=(0.9, 1.0))
    sig_raw = load_events(args.signal, max_events=n, seed=1339)

    bg = torch.from_numpy(normalizer.transform(bg_raw))
    sig = torch.from_numpy(normalizer.transform(sig_raw))
    bg_summu2 = vae_anomaly_score(model, bg, device)
    sig_summu2 = vae_anomaly_score(model, sig, device)
    bg_recon = vae_recon_error(model, bg, device)
    sig_recon = vae_recon_error(model, sig, device)

    auc_latent = _auc(bg_summu2, sig_summu2)
    auc_recon = _auc(bg_recon, sig_recon)

    # Columnar dump for RDataFrame (label 0 = background, 1 = signal).
    bg_obs, sig_obs = _observables(bg_raw), _observables(sig_raw)
    cols = {"label": np.concatenate([np.zeros(len(bg_raw)),
                                     np.ones(len(sig_raw))])}
    for k in bg_obs:
        cols[k] = np.concatenate([bg_obs[k], sig_obs[k]])
    cols["summu2"] = np.concatenate([bg_summu2, sig_summu2]).astype(np.float64)
    cols["recon_mse"] = np.concatenate([bg_recon, sig_recon]).astype(np.float64)
    out_npz = resolve_path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **cols)
    print(f"[prep-adc] wrote {out_npz} ({len(cols['label'])} rows)")

    # AUC table -- medians over means for the heavy-tailed scores.
    def stats(a):
        return {"median": float(np.median(a)), "mean": float(np.mean(a)),
                "p99": float(np.quantile(a, 0.99))}
    table = {
        "signal": args.signal_name,
        "model_dir": str(model_dir),
        "n_background": int(len(bg_raw)), "n_signal": int(len(sig_raw)),
        "scores": {
            "latent_summu2": {
                "auc": auc_latent, "note": "FPGA-cheap trigger score (Sum mu^2)",
                "background": stats(bg_summu2), "signal": stats(sig_summu2)},
            "recon_mse": {
                "auc": auc_recon, "note": "full-VAE reconstruction MSE (offline)",
                "background": stats(bg_recon), "signal": stats(sig_recon)},
        },
        "note": ("Medians quoted alongside means: both scores are heavy-tailed "
                 "and non-negative, so recon-MSE means are outlier-dominated."),
    }
    reports = resolve_path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "physics_auc_table.json").write_text(
        json.dumps(table, indent=2), encoding="utf-8")
    _write_md(table, reports / "physics_auc_table.md")
    print(f"[prep-adc] AUC latent(Sum mu^2)={auc_latent:.4f} "
          f"recon(MSE)={auc_recon:.4f} -> {reports}")


def _write_md(table: dict, path: Path) -> None:
    s = table["scores"]
    lines = [
        f"# Stream A physics ROC/AUC — background vs {table['signal']}",
        "",
        f"Model: `{table['model_dir']}` · "
        f"n_bg={table['n_background']} n_sig={table['n_signal']}",
        "",
        "| Score | AUC | bg median | sig median | bg mean | sig mean |",
        "|-------|-----|-----------|------------|---------|----------|",
    ]
    for name, key in (("Σμ² (trigger)", "latent_summu2"),
                      ("recon MSE (offline)", "recon_mse")):
        d = s[key]
        lines.append(
            f"| {name} | **{d['auc']:.3f}** | {d['background']['median']:.3g} | "
            f"{d['signal']['median']:.3g} | {d['background']['mean']:.3g} | "
            f"{d['signal']['mean']:.3g} |")
    lines += ["", f"_{table['note']}_", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
