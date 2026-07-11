"""Stream the RDataFrame-extracted CMS 57-vectors to ``events.physics``.

Host-side counterpart of ``ingest_nanoaod.py``: it replays the compact
``(N, 57)`` array the ROOT container produced through the EXACT same producer
interface and wire format as the Phase 1 sim producer
(``services/producers/physics_producer.py``) -- ``pharos.physics.v1`` records,
raw pre-normalization vectors, scorer owns normalization. No second wire format.

The point of the exercise: real CMS events now flow through the frozen,
sim-trained scorer + drift monitor, exposing the sim-to-real domain gap
(reported, not tuned away -- see ``scripts/phase4_sim_vs_real.py``).

Run: ``python -m services.ingest_root.stream_cms --rate 500 --limit 10000``
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from src.common.config import resolve_path
from services import common


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source-npy", default="data/interim/cms_events_57.npy",
                   help="(N,57) float32 array from ingest_nanoaod.py")
    p.add_argument("--rate", type=float, default=500.0,
                   help="events/sec (0 = unthrottled)")
    p.add_argument("--limit", type=int, default=0,
                   help="max events to emit (0 = all in the array)")
    p.add_argument("--bootstrap", default=common.BOOTSTRAP_DEFAULT)
    args = p.parse_args(argv)

    events = np.load(resolve_path(args.source_npy))
    if events.ndim != 2 or events.shape[1] != 57:
        raise SystemExit(f"expected (N,57), got {events.shape}")
    if args.limit:
        events = events[:args.limit]
    print(f"[cms-stream] {len(events)} CMS events at "
          f"{args.rate or 'max'} ev/s -> {common.TOPIC_PHYSICS}")

    producer = common.make_producer(args.bootstrap)
    limiter = common.RateLimiter(args.rate)
    for i, vec in enumerate(events):
        limiter.wait()
        record = {
            "schema": common.SCHEMA_PHYSICS,
            "event_id": common.new_event_id("cms", i),
            "producer_ts_ns": common.now_ns(),
            "features": vec.tolist(),   # 57 floats, pre-normalization
        }
        common.produce_json(producer, common.TOPIC_PHYSICS, record,
                            key=record["event_id"])
        producer.poll(0)
        if (i + 1) % 1000 == 0:
            print(f"[cms-stream] sent {i + 1}/{len(events)}", file=sys.stderr)
    producer.flush(30)
    print(f"[cms-stream] done: {len(events)} events sent")


if __name__ == "__main__":
    main()
