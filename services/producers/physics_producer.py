"""Replay ADC2021 events to ``events.physics`` as pre-normalization vectors.

Each record is the raw 57-dim ``[pT, eta, phi] x 19`` feature vector (class
column dropped, NO log1p/z-score -- the scorer owns normalization). By default
events come from the held-out tail region (0.9-1.0) of the background file so
the scorer's keep-rate on this stream is directly comparable to the Phase 0
background distribution the threshold was derived from.

Run: ``python -m services.producers.physics_producer --rate 200 --limit 5000``
"""
from __future__ import annotations

import argparse
import sys

from src.common.config import resolve_path
from src.preprocessing import adc2021

from services import common


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", type=str,
                   default="data/raw/adc2021/background_for_training.h5",
                   help="HDF5 file to replay (background or a signal file)")
    p.add_argument("--region", type=float, nargs=2, default=(0.9, 1.0),
                   metavar=("LO", "HI"),
                   help="fractional file region to sample (default: held-out tail)")
    p.add_argument("--rate", type=float, default=100.0,
                   help="events/sec (0 = unthrottled)")
    p.add_argument("--speedup", type=float, default=1.0,
                   help="multiplier on --rate")
    p.add_argument("--limit", type=int, default=10_000,
                   help="number of events to emit (0 = whole region)")
    p.add_argument("--bootstrap", type=str, default=common.BOOTSTRAP_DEFAULT)
    args = p.parse_args(argv)

    limit = args.limit or None
    events = adc2021.load_events(resolve_path(args.source), max_events=limit,
                                 region=tuple(args.region))
    rate = args.rate * args.speedup
    print(f"[physics-producer] {len(events)} events from {args.source} "
          f"region={tuple(args.region)} at {rate or 'max'} ev/s "
          f"-> {common.TOPIC_PHYSICS}")

    producer = common.make_producer(args.bootstrap)
    limiter = common.RateLimiter(rate)
    for i, vec in enumerate(events):
        limiter.wait()
        record = {
            "schema": common.SCHEMA_PHYSICS,
            "event_id": common.new_event_id("phys", i),
            "producer_ts_ns": common.now_ns(),
            "features": vec.tolist(),   # 57 floats, pre-normalization
        }
        common.produce_json(producer, common.TOPIC_PHYSICS, record,
                            key=record["event_id"])
        producer.poll(0)
        if (i + 1) % 1000 == 0:
            print(f"[physics-producer] sent {i + 1}/{len(events)}",
                  file=sys.stderr)
    producer.flush(30)
    print(f"[physics-producer] done: {len(events)} events sent")


if __name__ == "__main__":
    main()
