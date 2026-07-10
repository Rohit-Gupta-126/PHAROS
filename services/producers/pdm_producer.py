"""Replay SNS HVCM pulses to ``events.pdm`` as pre-normalization waveforms.

Each record carries one downsampled (target_len, 14) waveform, time-major,
exactly as produced by the Phase 0 preprocessing (``src.preprocessing.hvcm``
average pooling) but before per-channel z-scoring -- the scorer owns
normalization. ``ground_truth`` (Run/Fault state) is included for offline
keep-rate accounting only; the scorer must not use it.

Run: ``python -m services.producers.pdm_producer --system RFQ --rate 5``
"""
from __future__ import annotations

import argparse
import sys

from src.common.config import resolve_path
from src.preprocessing import hvcm

from services import common


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=str, default="data/raw/hvcm")
    p.add_argument("--systems", type=str, nargs="+", default=hvcm.SYSTEMS,
                   choices=hvcm.SYSTEMS)
    p.add_argument("--target-len", type=int, default=500)
    p.add_argument("--normal-only", action="store_true",
                   help="replay only Run (normal) pulses, for keep-rate checks")
    p.add_argument("--rate", type=float, default=10.0,
                   help="pulses/sec (0 = unthrottled)")
    p.add_argument("--speedup", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=1000,
                   help="max pulses per system (0 = all)")
    p.add_argument("--bootstrap", type=str, default=common.BOOTSTRAP_DEFAULT)
    args = p.parse_args(argv)

    rate = args.rate * args.speedup
    producer = common.make_producer(args.bootstrap)
    limiter = common.RateLimiter(rate)
    data_dir = resolve_path(args.data_dir)

    total = 0
    for system in args.systems:
        sys_data = hvcm.load_system(data_dir, system,
                                    target_len=args.target_len)
        idx = range(len(sys_data.waves))
        if args.normal_only:
            idx = [i for i in idx if not sys_data.is_fault[i]]
        if args.limit:
            idx = list(idx)[: args.limit]
        print(f"[pdm-producer] {system}: {len(list(idx))} pulses at "
              f"{rate or 'max'} /s -> {common.TOPIC_PDM}")
        for i in idx:
            limiter.wait()
            record = {
                "schema": common.SCHEMA_PDM,
                "event_id": common.new_event_id(f"pdm-{system}", i),
                "producer_ts_ns": common.now_ns(),
                "system": system,
                "shape": list(sys_data.waves[i].shape),  # [target_len, 14]
                "wave": sys_data.waves[i].tolist(),      # pre-normalization
                "ground_truth": "Fault" if sys_data.is_fault[i] else "Run",
            }
            common.produce_json(producer, common.TOPIC_PDM, record,
                                key=record["event_id"])
            producer.poll(0)
            total += 1
            if total % 200 == 0:
                print(f"[pdm-producer] sent {total}", file=sys.stderr)
    producer.flush(30)
    print(f"[pdm-producer] done: {total} pulses sent")


if __name__ == "__main__":
    main()
