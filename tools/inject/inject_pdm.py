"""PDM calibration-skew injector: normal-validation slice, then file head.

Deliberately reproduces the Phase 1 calibration mismatch (design log, Phase
1): thresholds/references were derived on the *normal-validation* split of
each system, but the Phase 1 demo replayed the *file head* -- a different
slice of the same machine in the same state. This is BENIGN sampling skew,
not concept drift.

Baseline: normal-val pulses of the tracked system (Phase 0 split, seed 1337).
After the switch: file-head pulses (normal-only, the Phase 1 demo behavior).
Marker -> ``reports/phase3/injection_marker_pdm.json`` + ``ctrl.inject``.

The question ``analyze_pdm_skew.py`` then asks: does the monitor's signature
(which metrics fired -- score PSI vs raw channel-mean PSI) distinguish this
benign skew from a real distribution shift?

Run: ``python -m tools.inject.inject_pdm --system RFQ --rate 20``
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import List

import numpy as np

from src.common.config import load_config, resolve_path
from src.preprocessing import hvcm

from services import common


def emit(producer, sys_data, indices: List[int], system: str, limiter) -> int:
    for i in indices:
        limiter.wait()
        record = {
            "schema": common.SCHEMA_PDM,
            "event_id": common.new_event_id(f"pdm-{system}", int(i)),
            "producer_ts_ns": common.now_ns(),
            "system": system,
            "shape": list(sys_data.waves[i].shape),
            "wave": sys_data.waves[i].tolist(),
            "ground_truth": "Fault" if sys_data.is_fault[i] else "Run",
        }
        common.produce_json(producer, common.TOPIC_PDM, record,
                            key=record["event_id"])
        producer.poll(0)
    return len(indices)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pdm-config", type=str, default="configs/pdm_ae.yaml")
    p.add_argument("--system", type=str, default="RFQ", choices=hvcm.SYSTEMS)
    p.add_argument("--head", type=int, default=120,
                   help="file-head normal pulses to inject after the switch")
    p.add_argument("--rate", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=1337,
                   help="Phase 0 split seed (must match derive_thresholds)")
    p.add_argument("--bootstrap", type=str, default=common.BOOTSTRAP_DEFAULT)
    p.add_argument("--marker-out", type=str,
                   default="reports/phase3/injection_marker_pdm.json")
    args = p.parse_args(argv)

    cfg = load_config(args.pdm_config)
    sys_data = hvcm.load_system(resolve_path(cfg["data_dir"]), args.system,
                                target_len=cfg["target_len"],
                                max_pulses=cfg.get("max_pulses"))
    _, val_idx = hvcm.split_normal_train_val(sys_data,
                                             val_frac=cfg["val_frac"],
                                             seed=args.seed)
    normal_idx = np.flatnonzero(~np.asarray(sys_data.is_fault))
    head_idx = normal_idx[: args.head]
    print(f"[inject-pdm] {args.system}: baseline={len(val_idx)} normal-val "
          f"pulses, then {len(head_idx)} file-head pulses at {args.rate}/s")

    producer = common.make_producer(args.bootstrap)
    limiter = common.RateLimiter(args.rate)

    emit(producer, sys_data, list(val_idx), args.system, limiter)
    switch_ts = common.now_ns()
    marker = {
        "schema": common.SCHEMA_INJECT,
        "stream": f"pdm/{args.system}",
        "inject_source": f"hvcm/{args.system} file-head slice "
                         f"(Phase 1 calibration skew, benign)",
        "start_ts_ns": switch_ts,
        "baseline_events": int(len(val_idx)),
        "inject_events": int(len(head_idx)),
        "rate": args.rate,
        "written": datetime.now(timezone.utc).isoformat(),
    }
    common.produce_json(producer, common.TOPIC_INJECT, marker)
    out = resolve_path(args.marker_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    print(f"[inject-pdm] SWITCH ts={switch_ts} (marker -> {out})")

    emit(producer, sys_data, list(head_idx), args.system, limiter)
    producer.flush(30)
    print("[inject-pdm] done")


if __name__ == "__main__":
    main()
