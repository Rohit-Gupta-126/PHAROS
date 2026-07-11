"""Phase 2 decision / rate-control layer -- an L1-accept-budget emulator.

Consumes fully-scored physics events from ``events.physics.scored`` (produced
by ``physics_scorer_sofie --forward-all``) and decides which reach
``anomalies.scouting``, applying BOTH:

1. the derived Phase 1 p99 background threshold on Sum mu^2, and
2. a hard output-rate budget per fixed time window (keep-top-N), emulating the
   L1 accept budget: even a burst of above-threshold events cannot exceed the
   bandwidth allocation.

Windowing is on scorer timestamps (``scored_ts_ns``). Each window's budget is
``ceil(budget_fraction * window_input_count)`` -- an accept *fraction*, like an
L1 trigger's fixed share of collision rate. At window close, above-threshold
events are ranked by score and the top-N kept. The published decision reason:

- ``threshold_pass``: kept; the window's budget was not binding.
- ``rate_limited``:   kept as one of the top-N in a window where MORE events
                      passed the threshold than the budget allowed (the rest
                      of that window's passers were dropped).

Stats (achieved keep-rate, reduction factor, per-window accept counts) go to
``reports/phase2/decision_stats.json``.

Run: ``python -m services.decision.physics_decision``
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

from src.common.config import resolve_path
from services import common

TOPIC_SCORED = "events.physics.scored"
SCHEMA_DECISION = "pharos.scouting.v2"


class WindowedBudget:
    """Collect scored events into fixed time windows; emit keep decisions."""

    def __init__(self, window_s: float, budget_fraction: float,
                 threshold: float) -> None:
        self.window_ns = int(window_s * 1e9)
        self.budget_fraction = budget_fraction
        self.threshold = threshold
        self.window_start_ns: int | None = None
        self.buffer: List[Dict[str, Any]] = []
        # Aggregates.
        self.n_in = 0
        self.n_kept = 0
        self.n_threshold_pass_in = 0
        self.n_rate_dropped = 0
        self.window_accepts: List[int] = []
        self.window_inputs: List[int] = []

    def add(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Add a record; return decided keepers of any window that closed."""
        ts = record["scored_ts_ns"]
        out: List[Dict[str, Any]] = []
        if self.window_start_ns is None:
            self.window_start_ns = ts
        elif ts - self.window_start_ns >= self.window_ns:
            out = self.flush()
            self.window_start_ns = ts
        self.buffer.append(record)
        return out

    def flush(self) -> List[Dict[str, Any]]:
        """Close the current window: rank passers, apply budget, decide."""
        if not self.buffer:
            return []
        events = self.buffer
        self.buffer = []
        self.n_in += len(events)
        self.window_inputs.append(len(events))

        budget = int(np.ceil(self.budget_fraction * len(events)))
        passers = [e for e in events if e["score"] > self.threshold]
        self.n_threshold_pass_in += len(passers)
        passers.sort(key=lambda e: e["score"], reverse=True)
        binding = len(passers) > budget
        kept = passers[:budget] if binding else passers
        self.n_rate_dropped += len(passers) - len(kept)
        self.n_kept += len(kept)
        self.window_accepts.append(len(kept))

        reason = "rate_limited" if binding else "threshold_pass"
        decided_ts = common.now_ns()
        return [{
            "schema": SCHEMA_DECISION,
            "event_id": e["event_id"],
            "score": e["score"],
            "threshold": self.threshold,
            "backend": e.get("backend"),
            "decision_reason": reason,
            "producer_ts_ns": e["producer_ts_ns"],
            "scored_ts_ns": e["scored_ts_ns"],
            "decided_ts_ns": decided_ts,
        } for e in kept]

    def stats(self) -> Dict[str, Any]:
        acc = np.asarray(self.window_accepts) if self.window_accepts else np.zeros(1)
        return {
            "n_input": self.n_in,
            "n_threshold_pass": self.n_threshold_pass_in,
            "n_kept": self.n_kept,
            "n_rate_dropped": self.n_rate_dropped,
            "achieved_keep_rate": self.n_kept / self.n_in if self.n_in else None,
            "reduction_factor": self.n_in / self.n_kept if self.n_kept else None,
            "budget_fraction": self.budget_fraction,
            "window_s": self.window_ns / 1e9,
            "n_windows": len(self.window_accepts),
            "per_window_accepts": {
                "mean": float(acc.mean()),
                "max": int(acc.max()),
                "counts": self.window_accepts,
            },
            "per_window_inputs": self.window_inputs,
        }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--thresholds", type=str, default="configs/thresholds.json")
    p.add_argument("--window", type=float, default=1.0,
                   help="accept-budget window in seconds")
    p.add_argument("--budget-fraction", type=float, default=0.01,
                   help="max kept fraction of each window's input (L1 budget)")
    p.add_argument("--reports-dir", type=str, default="reports/phase2")
    p.add_argument("--bootstrap", type=str, default=common.BOOTSTRAP_DEFAULT)
    p.add_argument("--group", type=str, default=None,
                   help="fixed consumer group (default: fresh per-run group "
                        "starting at end-of-topic -- offset hygiene)")
    p.add_argument("--idle", type=float, default=15.0)
    args = p.parse_args(argv)

    thr_cfg = json.loads(resolve_path(args.thresholds).read_text())
    threshold = thr_cfg["physics"]["threshold"]
    budget = WindowedBudget(args.window, args.budget_fraction, threshold)
    print(f"[decision] threshold={threshold:.6g} window={args.window}s "
          f"budget_fraction={args.budget_fraction}")

    group = args.group or common.fresh_group("pharos-physics-decision")
    consumer = common.make_consumer(TOPIC_SCORED, group, args.bootstrap,
                                    from_beginning=bool(args.group))
    producer = common.make_producer(args.bootstrap)

    def publish(decisions: List[Dict[str, Any]]) -> None:
        for d in decisions:
            common.produce_json(producer, common.TOPIC_SCOUTING, d,
                                key=d["event_id"])
        producer.poll(0)

    try:
        for record in common.consume_json(consumer, idle_timeout_s=args.idle):
            publish(budget.add(record))
        publish(budget.flush())  # close the final partial window on idle exit
    finally:
        producer.flush(30)
        consumer.close()
        if budget.n_in:
            stats = {"generated": datetime.now(timezone.utc).isoformat(),
                     "threshold": threshold, **budget.stats()}
            out = resolve_path(args.reports_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "decision_stats.json").write_text(json.dumps(stats, indent=2))
            brief = {k: v for k, v in stats.items()
                     if k not in ("per_window_accepts", "per_window_inputs")}
            print(f"[decision] {json.dumps(brief, indent=2)}")
        else:
            print("[decision] no messages consumed")


if __name__ == "__main__":
    main()
