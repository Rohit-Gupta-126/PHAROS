"""Phase 3 smoke tests: drift statistics (pure logic, no broker/model)."""
from __future__ import annotations

import numpy as np
import pytest

from services.monitor.drift_stats import (
    ReferenceDist, RollingMoments, SlidingWindow, evaluate_window, psi,
    ks_stat, severity_from_psi,
)

RNG = np.random.default_rng(42)


def _ref(samples=None) -> ReferenceDist:
    if samples is None:
        samples = RNG.normal(0.0, 1.0, 5000)
    return ReferenceDist.from_samples("score", samples)


class TestPsi:
    def test_same_distribution_is_stable(self):
        ref = _ref()
        cur = RNG.normal(0.0, 1.0, 2000)
        assert psi(ref, cur) < 0.05
        assert severity_from_psi(psi(ref, cur)) == "ok"

    def test_shifted_distribution_alerts(self):
        ref = _ref()
        cur = RNG.normal(2.0, 1.0, 2000)  # 2-sigma mean shift
        v = psi(ref, cur)
        assert v > 0.25
        assert severity_from_psi(v) == "alert"

    def test_mild_shift_warns_between_thresholds(self):
        assert severity_from_psi(0.15) == "warn"

    def test_open_ended_bins_capture_outliers(self):
        ref = _ref()
        # All mass far outside the reference support must still be counted.
        v = psi(ref, np.full(500, 100.0))
        assert np.isfinite(v) and v > 0.25

    def test_roundtrip_serialization(self):
        ref = _ref()
        ref2 = ReferenceDist.from_dict(ref.to_dict())
        cur = RNG.normal(0.5, 1.0, 1000)
        assert psi(ref, cur) == pytest.approx(psi(ref2, cur))

    def test_degenerate_reference_raises(self):
        with pytest.raises(ValueError):
            ReferenceDist.from_samples("const", np.zeros(1000))


class TestKs:
    def test_ks_detects_shift(self):
        a = RNG.normal(0, 1, 2000)
        assert ks_stat(a, RNG.normal(0, 1, 2000))["pvalue"] > 0.01
        assert ks_stat(a, RNG.normal(1, 1, 2000))["pvalue"] < 1e-6


class TestSlidingWindow:
    def test_evaluates_only_when_full_then_every_step(self):
        w = SlidingWindow(size=10, step=5)
        fires = [w.add(float(i), ts_ns=i) for i in range(25)]
        assert [i for i, f in enumerate(fires) if f] == [9, 14, 19, 24]

    def test_window_bounds_track_timestamps(self):
        w = SlidingWindow(size=3, step=1)
        for i in range(5):
            w.add(float(i), ts_ns=1000 + i)
        assert (w.window_start_ns, w.window_end_ns) == (1002, 1004)
        assert w.values().tolist() == [2.0, 3.0, 4.0]

    def test_evaluate_window_emits_alert_fields(self):
        ref = _ref()
        w = SlidingWindow(size=500, step=500)
        fired = False
        for i, v in enumerate(RNG.normal(3.0, 1.0, 500)):
            fired = w.add(v, ts_ns=i) or fired
        assert fired
        ev = evaluate_window(ref, w)
        assert ev["severity"] == "alert"
        assert ev["metric"] == "score_psi"
        assert ev["window_n"] == 500
        assert ev["window_end_ns"] == 499


class TestRollingMoments:
    def test_summary(self):
        m = RollingMoments(4)
        for v in (1, 2, 3, 4, 5):
            m.add(v)
        s = m.summary()
        assert s["n"] == 4 and s["mean"] == pytest.approx(3.5)
