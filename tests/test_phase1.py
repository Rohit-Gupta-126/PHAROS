"""Phase 1 smoke tests: wire format + scorer/Phase-0 score consistency.

No broker required -- these exercise the JSON round-trip and the frozen-model
scoring path directly, asserting that an event serialized the way the producer
does and scored the way the scorer does reproduces the Phase 0 score.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.common.config import PROJECT_ROOT
from src.common.device import get_device
from src.inference.scores import ae_recon_error, vae_anomaly_score
from src.preprocessing import adc2021, hvcm

PHYS_MODEL = PROJECT_ROOT / "models" / "physics_vae"
PDM_MODEL = PROJECT_ROOT / "models" / "pdm"
THRESHOLDS = PROJECT_ROOT / "configs" / "thresholds.json"


def test_rate_limiter_unthrottled_is_noop():
    from services.common import RateLimiter
    RateLimiter(0).wait()
    RateLimiter(None).wait()


def test_thresholds_file_schema():
    if not THRESHOLDS.exists():
        pytest.skip("configs/thresholds.json not generated (run make thresholds)")
    cfg = json.loads(THRESHOLDS.read_text())
    assert 0 < cfg["expected_background_keep_rate"] < 1
    assert cfg["physics"]["score"] == "sum_mu2"
    assert cfg["physics"]["threshold"] > 0
    for system, entry in cfg["pdm"].items():
        assert entry["score"] == "ae_recon_mse"
        assert entry["threshold"] > 0


@pytest.mark.skipif(not (PHYS_MODEL / "ckpt.pt").exists(),
                    reason="Phase 0 physics artifacts missing")
def test_physics_score_survives_json_roundtrip():
    """Producer JSON-serializes raw features; scorer must reproduce the score
    computed directly on the raw array (Phase 0 path) exactly-ish."""
    from scripts.eval_physics import _load_model

    device = get_device("cpu")
    normalizer = adc2021.Normalizer.load(PHYS_MODEL / "norm.npz")
    model = _load_model(PHYS_MODEL, device)

    rng = np.random.default_rng(0)
    raw = np.abs(rng.normal(size=(16, adc2021.N_FEATURES))).astype(np.float32)
    direct = vae_anomaly_score(model, torch.from_numpy(
        normalizer.transform(raw)), device)

    # Producer -> JSON -> scorer path.
    payload = [json.loads(json.dumps({"features": row.tolist()}))
               for row in raw]
    feats = np.asarray([r["features"] for r in payload], dtype=np.float32)
    via_wire = vae_anomaly_score(model, torch.from_numpy(
        normalizer.transform(feats)), device)

    np.testing.assert_allclose(via_wire, direct, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not (PDM_MODEL / "RFQ" / "ae.pt").exists(),
                    reason="Phase 0 PDM artifacts missing")
def test_pdm_score_survives_json_roundtrip():
    from scripts.eval_pdm import _load_ae

    device = get_device("cpu")
    norm = hvcm.ChannelNormalizer.load(PDM_MODEL / "RFQ" / "channel_norm.npz")
    ae = _load_ae(PDM_MODEL / "RFQ", device)

    rng = np.random.default_rng(0)
    waves = rng.normal(size=(4, 500, hvcm.N_CHANNELS)).astype(np.float32)
    direct = ae_recon_error(ae, torch.from_numpy(
        norm.transform_channels_first(waves)), device)

    wire = [np.asarray(json.loads(json.dumps({"wave": w.tolist()}))["wave"],
                       dtype=np.float32) for w in waves]
    via_wire = ae_recon_error(ae, torch.from_numpy(
        norm.transform_channels_first(np.stack(wire))), device)

    np.testing.assert_allclose(via_wire, direct, rtol=1e-5, atol=1e-7)
