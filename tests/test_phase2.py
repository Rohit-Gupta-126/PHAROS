"""Phase 2 smoke tests: ONNX backend parity + windowed-budget decision logic.

No broker required. The ONNX tests skip if the export has not been run yet
(``make export-onnx``).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.common.config import PROJECT_ROOT

PHYS_MODEL = PROJECT_ROOT / "models" / "physics_vae"
ONNX = PHYS_MODEL / "encoder_mu.onnx"


def _record(i: int, score: float, ts_ns: int) -> dict:
    return {"event_id": f"ev-{i:04d}", "score": score, "backend": "ort",
            "producer_ts_ns": ts_ns, "scored_ts_ns": ts_ns}


def test_ort_backend_matches_pytorch():
    if not ONNX.exists():
        pytest.skip("encoder_mu.onnx not exported (run make export-onnx)")
    from services.scorers.trigger_backends import OrtBackend, TorchBackend
    rng = np.random.default_rng(7)
    x = rng.normal(size=(64, 57)).astype(np.float32)
    s_ref = TorchBackend(PHYS_MODEL).score(x)
    s_ort = OrtBackend(PHYS_MODEL).score(x)
    assert np.abs(s_ref - s_ort).max() <= 1e-5


def test_windowed_budget_threshold_and_rate_limit():
    from services.decision.physics_decision import WindowedBudget
    wb = WindowedBudget(window_s=1.0, budget_fraction=0.01, threshold=0.5)
    t0 = 1_000_000_000_000
    # Window 1: 100 events, 1 passer -> budget 1, not binding.
    out = []
    for i in range(100):
        score = 0.9 if i == 3 else 0.1
        out += wb.add(_record(i, score, t0 + i * int(1e6)))
    # Window 2 trigger: event past the 1 s boundary flushes window 1.
    out += wb.add(_record(100, 0.1, t0 + int(1.5e9)))
    assert [d["event_id"] for d in out] == ["ev-0003"]
    assert out[0]["decision_reason"] == "threshold_pass"

    # Window 2: 100 events, 5 passers, budget 1 -> binding, keep the top score.
    for i in range(101, 200):
        score = 0.6 + i / 1000 if i < 106 else 0.1
        wb.add(_record(i, score, t0 + int(1.5e9) + (i - 100) * int(1e6)))
    # Window 2 holds 100 events (ev-0100 + 99 more): budget = ceil(0.01*100)
    # = 1, so only the highest-scoring passer survives, marked rate_limited.
    out2 = wb.flush()
    assert [d["event_id"] for d in out2] == ["ev-0105"]
    assert out2[0]["decision_reason"] == "rate_limited"


def test_windowed_budget_stats_reduction():
    from services.decision.physics_decision import WindowedBudget
    wb = WindowedBudget(window_s=1.0, budget_fraction=0.01, threshold=0.5)
    t0 = 0
    for i in range(1000):
        score = 0.9 if i % 100 == 0 else 0.0  # 1% passers exactly
        wb.add(_record(i, score, t0 + i * int(1e6)))
    wb.flush()
    s = wb.stats()
    assert s["n_input"] == 1000
    assert s["n_kept"] == 10
    assert s["achieved_keep_rate"] == pytest.approx(0.01)
    assert s["reduction_factor"] == pytest.approx(100.0)
