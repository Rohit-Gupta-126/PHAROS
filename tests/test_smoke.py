"""Tiny-subset smoke tests for both PHAROS Phase 0 pipelines.

These run each train + eval pipeline end-to-end on a very small subset, forced
onto CPU, so they are fast and CI-friendly. They require the raw datasets to be
present under ``data/raw/`` (skipped otherwise).
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.common.config import PROJECT_ROOT

BG = PROJECT_ROOT / "data/raw/adc2021/background_for_training.h5"
SIGNAL = PROJECT_ROOT / "data/raw/adc2021/Ato4l_lepFilter_13TeV_filtered.h5"
HVCM_RFQ = PROJECT_ROOT / "data/raw/hvcm/RFQ.npy"

requires_adc = pytest.mark.skipif(
    not (BG.exists() and SIGNAL.exists()), reason="ADC2021 data not present")
requires_hvcm = pytest.mark.skipif(
    not HVCM_RFQ.exists(), reason="HVCM data not present")


@requires_adc
def test_physics_train_and_eval(tmp_path: Path):
    from src.training import train_physics
    from scripts import eval_physics

    model_dir = tmp_path / "physics_vae"
    reports_dir = tmp_path / "reports"

    train_cfg = {
        "background_path": str(BG), "out_dir": str(model_dir),
        "max_events": 2000, "val_frac": 0.1, "epochs": 1,
        "batch_size": 256, "kl_warmup_epochs": 1, "device": "cpu",
        "amp": False, "latent_dim": 8, "hidden": [16, 8],
    }
    result = train_physics.run(train_cfg)
    assert (model_dir / "ckpt.pt").exists()
    assert (model_dir / "norm.npz").exists()
    assert math.isfinite(result["final_train_loss"])

    eval_cfg = {
        "background_path": str(BG), "signal_path": str(SIGNAL),
        "model_dir": str(model_dir), "reports_dir": str(reports_dir),
        "eval_events": 1000, "device": "cpu",
    }
    metrics = eval_physics.run(eval_cfg)
    assert (reports_dir / "physics_roc.png").exists()
    assert (reports_dir / "physics_score_hist.png").exists()
    assert 0.0 <= metrics["auc"] <= 1.0


@requires_hvcm
def test_pdm_train_and_eval(tmp_path: Path):
    from src.training import train_pdm
    from scripts import eval_pdm

    model_dir = tmp_path / "pdm"
    reports_dir = tmp_path / "reports"

    train_cfg = {
        "data_dir": str(HVCM_RFQ.parent), "out_dir": str(model_dir),
        "systems": ["RFQ"], "epochs": 1, "batch_size": 32, "device": "cpu",
        "amp": False, "base": 8, "latent_channels": 4, "iso_estimators": 50,
    }
    result = train_pdm.run(train_cfg)
    assert (model_dir / "RFQ" / "ae.pt").exists()
    assert (model_dir / "RFQ" / "iso.joblib").exists()
    assert math.isfinite(result["RFQ"]["final_train_loss"])

    eval_cfg = {
        "data_dir": str(HVCM_RFQ.parent), "model_dir": str(model_dir),
        "reports_dir": str(reports_dir), "systems": ["RFQ"], "device": "cpu",
    }
    out = eval_pdm.run(eval_cfg)
    assert (reports_dir / "pdm_auc.csv").exists()
    assert (reports_dir / "pdm_RFQ_auc.png").exists()
    # ALL-faults row must exist and be a valid AUC.
    all_row = next(r for r in out["rows"]
                   if r["system"] == "RFQ" and r["fault_class"] == "ALL")
    assert 0.0 <= all_row["ae_auc"] <= 1.0
    assert 0.0 <= all_row["iso_auc"] <= 1.0
