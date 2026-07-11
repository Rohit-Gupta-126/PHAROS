"""Drift statistics for the Phase 3 monitor: PSI, KS, sliding windows.

Hand-rolled PSI against a *frozen* reference histogram plus scipy's two-sample
KS test -- deliberately no alibi-detect/river dependency (footprint + Windows
reliability; see docs/design_log.md Phase 3).

PSI convention (banking / model-monitoring standard, used here as a heuristic,
not a calibrated test): < 0.1 stable, 0.1-0.25 warn, > 0.25 alert.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Sequence

import numpy as np
from scipy.stats import ks_2samp

PSI_WARN_DEFAULT = 0.10
PSI_ALERT_DEFAULT = 0.25

_EPS = 1e-6  # floor for empty bins so PSI stays finite


@dataclass(frozen=True)
class ReferenceDist:
    """Frozen reference distribution: decile-style histogram + moments."""

    name: str
    bin_edges: Sequence[float]   # len = n_bins + 1; open-ended outer bins
    bin_probs: Sequence[float]   # len = n_bins; sums to ~1
    mean: float
    std: float
    quantiles: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_samples(cls, name: str, samples: np.ndarray,
                     n_bins: int = 10) -> "ReferenceDist":
        """Build a reference from Phase 0 samples using quantile bin edges."""
        samples = np.asarray(samples, dtype=np.float64).ravel()
        if samples.size < n_bins * 2:
            raise ValueError(f"{name}: need >= {n_bins * 2} samples, "
                             f"got {samples.size}")
        # Quantile edges give ~equal-mass bins; drop duplicate edges that a
        # spiky distribution can produce.
        edges = np.unique(np.quantile(samples, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            raise ValueError(f"{name}: distribution too degenerate to bin")
        probs = _bin_probs(samples, edges)
        qs = {f"p{int(q * 100):02d}": float(np.quantile(samples, q))
              for q in (0.01, 0.25, 0.50, 0.75, 0.99)}
        return cls(name=name, bin_edges=edges.tolist(),
                   bin_probs=probs.tolist(),
                   mean=float(samples.mean()), std=float(samples.std()),
                   quantiles=qs)

    def to_dict(self) -> Dict:
        return {"name": self.name, "bin_edges": list(self.bin_edges),
                "bin_probs": list(self.bin_probs), "mean": self.mean,
                "std": self.std, "quantiles": dict(self.quantiles)}

    @classmethod
    def from_dict(cls, d: Dict) -> "ReferenceDist":
        return cls(name=d["name"], bin_edges=d["bin_edges"],
                   bin_probs=d["bin_probs"], mean=d["mean"], std=d["std"],
                   quantiles=d.get("quantiles", {}))


def _bin_probs(samples: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Histogram probabilities with open-ended outer bins (no mass lost)."""
    inner = np.asarray(edges, dtype=np.float64)[1:-1]
    idx = np.searchsorted(inner, samples, side="right")
    counts = np.bincount(idx, minlength=len(inner) + 1).astype(np.float64)
    return counts / counts.sum()


def psi(reference: ReferenceDist, current: np.ndarray) -> float:
    """Population stability index of ``current`` vs the frozen reference."""
    cur = _bin_probs(np.asarray(current, dtype=np.float64).ravel(),
                     np.asarray(reference.bin_edges))
    ref = np.clip(np.asarray(reference.bin_probs, dtype=np.float64), _EPS, None)
    cur = np.clip(cur, _EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def ks_stat(reference_samples: np.ndarray, current: np.ndarray) -> Dict[str, float]:
    """Two-sample KS statistic + p-value (needs raw reference samples)."""
    stat, p = ks_2samp(np.asarray(reference_samples).ravel(),
                       np.asarray(current).ravel())
    return {"stat": float(stat), "pvalue": float(p)}


def severity_from_psi(value: float,
                      warn: float = PSI_WARN_DEFAULT,
                      alert: float = PSI_ALERT_DEFAULT) -> str:
    if value >= alert:
        return "alert"
    if value >= warn:
        return "warn"
    return "ok"


class SlidingWindow:
    """Fixed-size sliding sample window with step-based evaluation points.

    ``add`` returns True every ``step`` samples once the window is full,
    signalling "evaluate drift now".
    """

    def __init__(self, size: int, step: int | None = None) -> None:
        if size < 2:
            raise ValueError("window size must be >= 2")
        self.size = size
        self.step = step or size
        self._buf: Deque[float] = deque(maxlen=size)
        self._since_eval = 0
        self.first_ts_ns: int | None = None
        self.last_ts_ns: int | None = None
        self._ts: Deque[int] = deque(maxlen=size)

    def add(self, value: float, ts_ns: int | None = None) -> bool:
        self._buf.append(float(value))
        self._ts.append(int(ts_ns) if ts_ns is not None else 0)
        self._since_eval += 1
        if len(self._buf) < self.size:
            return False
        if self._since_eval >= self.step:
            self._since_eval = 0
            return True
        return False

    def values(self) -> np.ndarray:
        return np.asarray(self._buf, dtype=np.float64)

    @property
    def window_start_ns(self) -> int:
        return self._ts[0] if self._ts else 0

    @property
    def window_end_ns(self) -> int:
        return self._ts[-1] if self._ts else 0

    def __len__(self) -> int:
        return len(self._buf)


class RollingMoments:
    """Cheap rolling mean/std tracker over the last N samples."""

    def __init__(self, size: int) -> None:
        self._buf: Deque[float] = deque(maxlen=size)

    def add(self, value: float) -> None:
        self._buf.append(float(value))

    def summary(self) -> Dict[str, float]:
        a = np.asarray(self._buf, dtype=np.float64)
        if a.size == 0:
            return {"n": 0, "mean": float("nan"), "std": float("nan"),
                    "p50": float("nan")}
        return {"n": int(a.size), "mean": float(a.mean()),
                "std": float(a.std()), "p50": float(np.median(a))}


def evaluate_window(reference: ReferenceDist, window: SlidingWindow,
                    warn: float = PSI_WARN_DEFAULT,
                    alert: float = PSI_ALERT_DEFAULT) -> Dict:
    """One drift evaluation: PSI severity + rolling-moment context."""
    cur = window.values()
    value = psi(reference, cur)
    return {
        "metric": f"{reference.name}_psi",
        "value": value,
        "threshold_warn": warn,
        "threshold_alert": alert,
        "severity": severity_from_psi(value, warn, alert),
        "window_n": len(window),
        "window_start_ns": window.window_start_ns,
        "window_end_ns": window.window_end_ns,
        "current_mean": float(cur.mean()),
        "current_p50": float(np.median(cur)),
        "reference_mean": reference.mean,
    }
