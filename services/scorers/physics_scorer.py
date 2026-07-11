"""Score ``events.physics`` with the frozen Phase 0 VAE; republish keepers.

Loads the Phase 0 normalizer + checkpoint (no refit), applies the SAME
log1p/z-score transform, computes the Sum mu^2 trigger score, and republishes
``{event_id, score, ts}`` to ``anomalies.scouting`` for events above the
derived threshold in ``configs/thresholds.json``.

Messages are drained into micro-batches (up to ``--batch``) before the model
call so GPU inference is amortized; latency is still tracked per message.
Exits after ``--idle`` seconds with no data and writes latency/throughput/
keep-rate stats to ``reports/phase1/``.

Run: ``python -m services.scorers.physics_scorer``
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import numpy as np
import torch

from src.common.config import resolve_path
from src.common.device import describe_device, get_device
from src.inference.scores import vae_anomaly_score
from src.preprocessing.adc2021 import N_FEATURES, Normalizer

from scripts.eval_physics import _load_model
from services import common
from services.scorers.instrumentation import StreamStats


def drain_batch(consumer, first: Dict[str, Any], max_batch: int) -> List[Dict[str, Any]]:
    batch = [first]
    while len(batch) < max_batch:
        msg = consumer.poll(0)
        if msg is None or msg.error():
            break
        batch.append(json.loads(msg.value().decode("utf-8")))
    return batch


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=str, default="models/physics_vae")
    p.add_argument("--thresholds", type=str, default="configs/thresholds.json")
    p.add_argument("--reports-dir", type=str, default="reports/phase1")
    p.add_argument("--bootstrap", type=str, default=common.BOOTSTRAP_DEFAULT)
    p.add_argument("--group", type=str, default=None,
                   help="fixed consumer group (default: fresh per-run group "
                        "starting at end-of-topic -- offset hygiene)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--idle", type=float, default=15.0,
                   help="exit after this many seconds with no messages")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    args = p.parse_args(argv)

    device = get_device(args.device)
    model_dir = resolve_path(args.model_dir)
    normalizer = Normalizer.load(model_dir / "norm.npz")
    model = _load_model(model_dir, device)
    thr_cfg = json.loads(resolve_path(args.thresholds).read_text())
    threshold = thr_cfg["physics"]["threshold"]
    print(f"[physics-scorer] device={describe_device(device)} "
          f"threshold(Sum mu^2)={threshold:.6g} "
          f"(p{thr_cfg['physics']['percentile']:g} background)")

    group = args.group or common.fresh_group("pharos-physics-scorer")
    consumer = common.make_consumer(common.TOPIC_PHYSICS, group,
                                    args.bootstrap,
                                    from_beginning=bool(args.group))
    producer = common.make_producer(args.bootstrap)
    stats = StreamStats("physics")
    all_scores: List[float] = []

    try:
        for record in common.consume_json(consumer, idle_timeout_s=args.idle):
            batch = drain_batch(consumer, record, args.batch)
            feats = np.asarray([r["features"] for r in batch], dtype=np.float32)
            assert feats.shape[1] == N_FEATURES, feats.shape
            x = torch.from_numpy(normalizer.transform(feats))
            scores = vae_anomaly_score(model, x, device)
            scored_ts = common.now_ns()
            for r, s in zip(batch, scores):
                kept = bool(s > threshold)
                stats.record(r["producer_ts_ns"], scored_ts, kept)
                all_scores.append(float(s))
                if kept:
                    common.produce_json(producer, common.TOPIC_SCOUTING, {
                        "schema": common.SCHEMA_SCOUTING,
                        "event_id": r["event_id"],
                        "score": float(s),
                        "threshold": threshold,
                        "producer_ts_ns": r["producer_ts_ns"],
                        "scored_ts_ns": scored_ts,
                    }, key=r["event_id"])
            producer.poll(0)
    finally:
        producer.flush(30)
        consumer.close()
        if stats.n_total:
            sc = np.asarray(all_scores)
            summary = stats.write_report(resolve_path(args.reports_dir), extra={
                "threshold": threshold,
                "configured_percentile": thr_cfg["physics"]["percentile"],
                "expected_keep_rate": thr_cfg["expected_background_keep_rate"],
                "score_mean": float(sc.mean()),
                "score_median": float(np.median(sc)),
                "score_p99": float(np.percentile(sc, 99)),
            })
            print(f"[physics-scorer] {json.dumps(summary, indent=2)}")
        else:
            print("[physics-scorer] no messages consumed")


if __name__ == "__main__":
    main()
