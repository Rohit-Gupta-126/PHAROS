"""Evaluate the Stream B PDM detectors: per-fault-class detection AUC.

For each system we score every pulse with both detectors (conv AE reconstruction
error and IsolationForest) and, for each fault class, compute the AUC of
(normal vs that fault class). An overall "ALL faults" AUC is also reported.

Runnable via ``python -m scripts.eval_pdm --config configs/pdm_ae.yaml``
(Makefile target ``eval-pdm``). Writes to ``reports/phase0/``:

    pdm_auc.csv            - long table: system, fault_class, n, n_below_floor,
                             ae_auc, iso_auc (labels merged by case/whitespace)
    pdm_metrics.json       - {headline (median AUC over n>=5 classes), per_system}
    pdm_{system}_auc.png   - per-system bar chart of per-fault AE vs IsoForest AUC
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
import torch
from sklearn.metrics import roc_auc_score

from src.common.config import load_config, resolve_path
from src.common.device import describe_device, get_device, seed_everything
from src.inference.scores import ae_recon_error
from src.preprocessing.hvcm import ChannelNormalizer, load_system
from src.training.conv_ae import ConvAE


DEFAULTS: Dict[str, Any] = {
    "data_dir": "data/raw/hvcm",
    "model_dir": "models/pdm",
    "reports_dir": "reports/phase0",
    "systems": None,          # None -> whatever models exist under model_dir
    "target_len": 500,
    "max_pulses": None,
    "min_fault_samples": 3,   # skip fault classes with fewer samples
    "headline_min_n": 5,      # floor for the median-AUC headline stat
    "seed": 1337,
    "device": None,
}


def _canon_fault(s: str) -> str:
    """Canonicalize a fault-class label so case/whitespace variants merge.

    e.g. "B FLUX Low Fault" and "B Flux Low Fault" both map to the same string,
    as do "C FLUX Low Fault" / "C Flux Low Fault". We collapse internal
    whitespace and title-case, which normalizes the FLUX/Flux and spacing
    differences seen in the raw SNS labels without altering their meaning.
    """
    return " ".join(str(s).split()).title()


def _merge(cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = dict(DEFAULTS)
    if cfg:
        merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged


def _load_ae(model_dir: Path, device: torch.device) -> ConvAE:
    ckpt = torch.load(model_dir / "ae.pt", map_location=device)
    model = ConvAE(n_channels=ckpt["n_channels"], seq_len=ckpt["seq_len"],
                   base=ckpt["base"], latent_channels=ckpt["latent_channels"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def _auc(neg_scores: np.ndarray, pos_scores: np.ndarray) -> float:
    """AUC with negatives (normal) labeled 0 and positives (fault) labeled 1."""
    y = np.concatenate([np.zeros(len(neg_scores)), np.ones(len(pos_scores))])
    s = np.concatenate([neg_scores, pos_scores])
    return float(roc_auc_score(y, s))


def run(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    c = _merge(cfg)
    seed_everything(c["seed"])
    device = get_device(c["device"])
    data_dir = resolve_path(c["data_dir"])
    model_root = resolve_path(c["model_dir"])
    reports_dir = resolve_path(c["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval-pdm] device = {describe_device(device)}")

    systems = c["systems"]
    if not systems:
        systems = sorted(p.name for p in model_root.iterdir()
                         if (p / "ae.pt").exists()) if model_root.exists() else []
    if not systems:
        raise FileNotFoundError(f"No trained PDM systems under {model_root}")

    rows: List[Dict[str, Any]] = []
    per_system: Dict[str, Any] = {}
    for system in systems:
        model_dir = model_root / system
        print(f"[eval-pdm] === {system} ===")
        sys_data = load_system(data_dir, system, target_len=c["target_len"],
                               max_pulses=c["max_pulses"])
        # Merge case/whitespace-variant fault labels before aggregation.
        sys_data.fault_type = np.array(
            [_canon_fault(t) for t in sys_data.fault_type], dtype=object)
        norm = ChannelNormalizer.load(model_dir / "channel_norm.npz")
        ae = _load_ae(model_dir, device)
        iso_bundle = joblib.load(model_dir / "iso.joblib")
        iso, scaler = iso_bundle["iso"], iso_bundle["scaler"]

        waves_cf = norm.transform_channels_first(sys_data.waves)
        ae_scores = ae_recon_error(ae, torch.from_numpy(waves_cf), device)
        # Higher iso score = more anomalous.
        iso_scores = -iso.decision_function(scaler.transform(sys_data.features))

        normal = ~sys_data.is_fault
        ae_norm, iso_norm = ae_scores[normal], iso_scores[normal]

        floor = c["headline_min_n"]
        system_rows = []
        # Overall: all faults vs normal.
        fault = sys_data.is_fault
        row_all = {
            "system": system, "fault_class": "ALL",
            "n": int(fault.sum()), "n_below_floor": False,
            "ae_auc": _auc(ae_norm, ae_scores[fault]),
            "iso_auc": _auc(iso_norm, iso_scores[fault]),
        }
        rows.append(row_all); system_rows.append(row_all)

        for fclass in sys_data.fault_classes():
            mask = fault & (sys_data.fault_type == fclass)
            n = int(mask.sum())
            if n < c["min_fault_samples"]:
                continue
            row = {
                "system": system, "fault_class": fclass, "n": n,
                "n_below_floor": n < floor,
                "ae_auc": _auc(ae_norm, ae_scores[mask]),
                "iso_auc": _auc(iso_norm, iso_scores[mask]),
            }
            rows.append(row); system_rows.append(row)

        per_system[system] = system_rows
        _plot_system(system, system_rows, reports_dir / f"pdm_{system}_auc.png")
        print(f"[eval-pdm] {system}: ALL ae_auc={row_all['ae_auc']:.3f} "
              f"iso_auc={row_all['iso_auc']:.3f} "
              f"({len(system_rows) - 1} classes >= {c['min_fault_samples']})")

    # Headline aggregate: median per-class AUC over classes with n >= floor
    # (excludes the ALL row and noisy small-n classes).
    floor = c["headline_min_n"]
    headline_classes = [r for r in rows
                        if r["fault_class"] != "ALL" and not r["n_below_floor"]]
    headline = {
        "headline_min_n": floor,
        "n_classes_total": sum(1 for r in rows if r["fault_class"] != "ALL"),
        "n_classes_in_headline": len(headline_classes),
        "median_ae_auc": (float(np.median([r["ae_auc"] for r in headline_classes]))
                          if headline_classes else None),
        "median_iso_auc": (float(np.median([r["iso_auc"] for r in headline_classes]))
                           if headline_classes else None),
    }
    print(f"[eval-pdm] headline (n>={floor}, {headline['n_classes_in_headline']}"
          f"/{headline['n_classes_total']} classes): "
          f"median AE={headline['median_ae_auc']} "
          f"IsoForest={headline['median_iso_auc']}")

    _write_csv(rows, reports_dir / "pdm_auc.csv")
    (reports_dir / "pdm_metrics.json").write_text(
        json.dumps({"headline": headline, "per_system": per_system}, indent=2),
        encoding="utf-8")
    print(f"[eval-pdm] wrote table + plots -> {reports_dir}")
    return {"rows": rows, "headline": headline}


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    import csv
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["system", "fault_class", "n",
                                          "n_below_floor", "ae_auc", "iso_auc"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "ae_auc": f"{r['ae_auc']:.4f}",
                        "iso_auc": f"{r['iso_auc']:.4f}"})


def _plot_system(system: str, rows: List[Dict[str, Any]], path: Path) -> None:
    labels = [r["fault_class"] if r["fault_class"] != "ALL"
              else "ALL" for r in rows]
    ae = [r["ae_auc"] for r in rows]
    iso = [r["iso_auc"] for r in rows]
    y = np.arange(len(labels))
    h = 0.4
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(labels) + 1)))
    ax.barh(y - h / 2, ae, height=h, label="Conv AE", color="#3b7dd8")
    ax.barh(y + h / 2, iso, height=h, label="IsolationForest", color="#e08a3c")
    ax.axvline(0.5, ls="--", color="grey", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("detection AUC (normal vs fault class)")
    ax.set_title(f"PHAROS Stream B PDM - {system}")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Stream B PDM detectors.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--systems", type=str, default=None)
    p.add_argument("--max-pulses", type=int, default=None)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    cfg = load_config(args.config) if args.config else {}
    if args.systems:
        cfg["systems"] = [s.strip() for s in args.systems.split(",") if s.strip()]
    if args.max_pulses is not None:
        cfg["max_pulses"] = args.max_pulses
    if args.device is not None:
        cfg["device"] = args.device
    run(cfg)


if __name__ == "__main__":
    main()
