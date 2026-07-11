"""Mid-stream physics shift injector: background, then black-box events.

Replays held-out ADC2021 background to ``events.physics`` exactly like the
Phase 1 producer, then switches the source to the unlabeled black-box file
(``BlackBox_13TeV_PU20.h5`` -- may contain new physics) at the same rate and
schema. The scorer/monitor cannot tell the producer changed except through
the data itself.

At the switch it records an injection marker (wall-clock ns + sequence) to
``reports/phase3/injection_marker.json`` AND to the ``ctrl.inject`` topic.
The monitor never reads either; ``measure_lead_time.py`` joins the marker
against ``alerts.drift`` afterwards to compute detection lead time.

Run: ``python -m tools.inject.inject_physics --rate 500``
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np

from src.common.config import resolve_path
from src.preprocessing import adc2021

from services import common


def emit(producer, events: np.ndarray, limiter, start_seq: int,
         prefix: str) -> int:
    seq = start_seq
    for vec in events:
        limiter.wait()
        record = {
            "schema": common.SCHEMA_PHYSICS,
            "event_id": common.new_event_id(prefix, seq),
            "producer_ts_ns": common.now_ns(),
            "features": vec.tolist(),
        }
        common.produce_json(producer, common.TOPIC_PHYSICS, record,
                            key=record["event_id"])
        producer.poll(0)
        seq += 1
    return seq


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--baseline-source", type=str,
                   default="data/raw/adc2021/background_for_training.h5")
    p.add_argument("--inject-source", type=str,
                   default="data/raw/adc2021/BlackBox_13TeV_PU20.h5")
    p.add_argument("--baseline", type=int, default=4000,
                   help="background events before the switch")
    p.add_argument("--inject", type=int, default=6000,
                   help="black-box events after the switch")
    p.add_argument("--rate", type=float, default=500.0)
    p.add_argument("--bootstrap", type=str, default=common.BOOTSTRAP_DEFAULT)
    p.add_argument("--marker-out", type=str,
                   default="reports/phase3/injection_marker.json")
    args = p.parse_args(argv)

    baseline = adc2021.load_events(resolve_path(args.baseline_source),
                                   max_events=args.baseline, region=(0.9, 1.0))
    injected = adc2021.load_events(resolve_path(args.inject_source),
                                   max_events=args.inject)
    print(f"[inject-physics] baseline={len(baseline)} from "
          f"{args.baseline_source} | injected={len(injected)} from "
          f"{args.inject_source} at {args.rate} ev/s")

    producer = common.make_producer(args.bootstrap)
    limiter = common.RateLimiter(args.rate)

    seq = emit(producer, baseline, limiter, 0, "phys")
    switch_ts = common.now_ns()
    marker = {
        "schema": common.SCHEMA_INJECT,
        "stream": "physics",
        "inject_source": args.inject_source,
        "start_ts_ns": switch_ts,
        "start_seq": seq,
        "baseline_events": len(baseline),
        "inject_events": len(injected),
        "rate": args.rate,
        "written": datetime.now(timezone.utc).isoformat(),
    }
    common.produce_json(producer, common.TOPIC_INJECT, marker)
    out = resolve_path(args.marker_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    print(f"[inject-physics] SWITCH at seq={seq} ts={switch_ts} "
          f"(marker -> {out})")

    # Same event-id prefix on purpose: downstream must not be able to tell.
    emit(producer, injected, limiter, seq, "phys")
    producer.flush(30)
    print(f"[inject-physics] done: {len(baseline)} baseline + "
          f"{len(injected)} injected events sent")


if __name__ == "__main__":
    main()
