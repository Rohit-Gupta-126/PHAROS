"""Latency / throughput / keep-rate instrumentation for the Phase 1 scorers.

Latency is producer timestamp -> scorer output timestamp (same host, same
clock). On close, writes summary JSON + a latency/throughput plot to
``reports/phase1/``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class StreamStats:
    def __init__(self, name: str) -> None:
        self.name = name
        self.latencies_ms: List[float] = []
        self.arrival_ts: List[float] = []   # scorer-side wall clock, seconds
        self.n_total = 0
        self.n_kept = 0

    def record(self, producer_ts_ns: int, scored_ts_ns: int,
               kept: bool) -> None:
        self.latencies_ms.append((scored_ts_ns - producer_ts_ns) / 1e6)
        self.arrival_ts.append(scored_ts_ns / 1e9)
        self.n_total += 1
        self.n_kept += int(kept)

    def summary(self) -> Dict[str, Any]:
        lat = np.asarray(self.latencies_ms)
        ts = np.asarray(self.arrival_ts)
        span = float(ts.max() - ts.min()) if len(ts) > 1 else 0.0
        return {
            "stream": self.name,
            "n_messages": self.n_total,
            "n_kept": self.n_kept,
            "keep_rate": self.n_kept / self.n_total if self.n_total else None,
            "throughput_msgs_per_sec": self.n_total / span if span else None,
            "latency_ms": {
                "mean": float(lat.mean()),
                "p50": float(np.percentile(lat, 50)),
                "p95": float(np.percentile(lat, 95)),
                "p99": float(np.percentile(lat, 99)),
                "max": float(lat.max()),
            } if len(lat) else None,
        }

    def write_report(self, reports_dir: str | Path,
                     extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        if extra:
            summary.update(extra)
        (reports_dir / f"{self.name}_stream_metrics.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        if self.latencies_ms:
            self._plot(reports_dir / f"{self.name}_latency.png")
        return summary

    def _plot(self, path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lat = np.asarray(self.latencies_ms)
        t = np.asarray(self.arrival_ts) - self.arrival_ts[0]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.hist(np.clip(lat, 0, np.percentile(lat, 99.5)), bins=60)
        ax1.set_xlabel("latency producer->scored (ms)")
        ax1.set_ylabel("count")
        ax1.set_title(f"{self.name}: e2e latency "
                      f"(p50={np.percentile(lat, 50):.1f} ms, "
                      f"p99={np.percentile(lat, 99):.1f} ms)")
        # Throughput in 1 s bins.
        if t.max() > 0:
            bins = np.arange(0, t.max() + 1, 1.0)
            counts, _ = np.histogram(t, bins=bins)
            ax2.plot(bins[:-1], counts, drawstyle="steps-post")
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("msgs/sec")
        ax2.set_title(f"{self.name}: throughput")
        for ax in (ax1, ax2):
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
