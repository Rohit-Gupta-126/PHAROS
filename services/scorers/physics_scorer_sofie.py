"""Phase 2 trigger-realistic scorer for ``events.physics``.

Same stream contract as ``physics_scorer`` (Phase 1), but scores through the
deploy-path backend instead of batched PyTorch: ONNX Runtime on the exported
``encoder_mu.onnx`` by default, or the SOFIE C++ binary once it is built
elsewhere (``--backend sofie``; see services/inference_sofie/README.md).
Events are scored one at a time (batch 1) -- the L1-trigger processing model.

Two output modes:
- default: republish above-threshold keepers to ``anomalies.scouting``
  (drop-in replacement for the Phase 1 scorer);
- ``--forward-all``: publish EVERY scored event to ``events.physics.scored``
  so the Phase 2 decision layer can apply the threshold + rate budget itself.

Stats go to ``reports/phase2/physics_<backend>_stream_metrics.json``.

Run: ``python -m services.scorers.physics_scorer_sofie --backend ort``
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.common.config import resolve_path
from src.preprocessing.adc2021 import N_FEATURES, Normalizer

from services import common
from services.scorers.instrumentation import StreamStats
from services.scorers.trigger_backends import make_backend

TOPIC_SCORED = "events.physics.scored"
SCHEMA_SCORED = "pharos.physics_scored.v1"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", choices=["ort", "sofie"], default="ort")
    p.add_argument("--model-dir", type=str, default="models/physics_vae")
    p.add_argument("--thresholds", type=str, default="configs/thresholds.json")
    p.add_argument("--reports-dir", type=str, default="reports/phase2")
    p.add_argument("--bootstrap", type=str, default=common.BOOTSTRAP_DEFAULT)
    p.add_argument("--group", type=str, default=None,
                   help="fixed consumer group (default: fresh per-run group "
                        "starting at end-of-topic -- offset hygiene)")
    p.add_argument("--idle", type=float, default=15.0)
    p.add_argument("--forward-all", action="store_true",
                   help="publish every scored event (for the decision layer), "
                        "not only above-threshold keepers")
    p.add_argument("--output-topic", type=str, default=None,
                   help="default: events.physics.scored with --forward-all, "
                        "anomalies.scouting otherwise")
    p.add_argument("--model-pointer", type=str, default=None,
                   help="hot-swap pointer file (e.g. models/physics_vae/"
                        "current.json); polled every --swap-poll messages, "
                        "the scorer reloads when its model_dir changes")
    p.add_argument("--swap-poll", type=int, default=500)
    args = p.parse_args(argv)

    model_dir = resolve_path(args.model_dir)
    normalizer = Normalizer.load(model_dir / "norm.npz")
    backend = make_backend(args.backend, model_dir)
    thr_cfg = json.loads(resolve_path(args.thresholds).read_text())
    threshold = thr_cfg["physics"]["threshold"]

    pointer_path = resolve_path(args.model_pointer) if args.model_pointer else None
    active_dir = str(model_dir)
    n_swaps = 0
    if pointer_path:
        from services.monitor.hot_swap import ModelPointer
        ptr = ModelPointer.read(pointer_path)
        if ptr:  # pointer pre-exists: serve what it names
            active_dir = ptr.model_dir
            model_dir = resolve_path(ptr.model_dir)
            normalizer = Normalizer.load(model_dir / "norm.npz")
            backend = make_backend(args.backend, model_dir)
            threshold = ptr.threshold
        print(f"[physics-scorer-p2] hot-swap pointer: {pointer_path} "
              f"(active={active_dir})")
    out_topic = args.output_topic or (
        TOPIC_SCORED if args.forward_all else common.TOPIC_SCOUTING)
    print(f"[physics-scorer-p2] backend={args.backend} "
          f"threshold(Sum mu^2)={threshold:.6g} -> {out_topic} "
          f"(forward_all={args.forward_all})")

    group = args.group or common.fresh_group("pharos-physics-scorer-p2")
    consumer = common.make_consumer(common.TOPIC_PHYSICS, group,
                                    args.bootstrap,
                                    from_beginning=bool(args.group))
    producer = common.make_producer(args.bootstrap)
    stats = StreamStats(f"physics_{args.backend}")

    n_since_poll = 0
    try:
        for r in common.consume_json(consumer, idle_timeout_s=args.idle):
            if pointer_path:
                n_since_poll += 1
                if n_since_poll >= args.swap_poll:
                    n_since_poll = 0
                    ptr = ModelPointer.read(pointer_path)
                    if ptr and ptr.model_dir != active_dir:
                        # Load the NEW model fully before dropping the old
                        # one -- a failed load keeps the old model serving.
                        try:
                            new_dir = resolve_path(ptr.model_dir)
                            new_norm = Normalizer.load(new_dir / "norm.npz")
                            new_backend = make_backend(args.backend, new_dir)
                        except Exception as exc:  # noqa: BLE001
                            print(f"[physics-scorer-p2] swap to "
                                  f"{ptr.model_dir} FAILED to load ({exc}); "
                                  f"keeping {active_dir}")
                        else:
                            if hasattr(backend, "close"):
                                backend.close()
                            normalizer, backend = new_norm, new_backend
                            threshold = ptr.threshold
                            active_dir = ptr.model_dir
                            n_swaps += 1
                            print(f"[physics-scorer-p2] HOT-SWAP -> "
                                  f"{active_dir} threshold={threshold:.6g}")
            feats = np.asarray([r["features"]], dtype=np.float32)
            assert feats.shape[1] == N_FEATURES, feats.shape
            score = float(backend.score(normalizer.transform(feats))[0])
            scored_ts = common.now_ns()
            kept = score > threshold
            stats.record(r["producer_ts_ns"], scored_ts, kept)
            if kept or args.forward_all:
                common.produce_json(producer, out_topic, {
                    "schema": SCHEMA_SCORED,
                    "event_id": r["event_id"],
                    "score": score,
                    "threshold": threshold,
                    "backend": args.backend,
                    "producer_ts_ns": r["producer_ts_ns"],
                    "scored_ts_ns": scored_ts,
                }, key=r["event_id"])
            producer.poll(0)
    finally:
        producer.flush(30)
        consumer.close()
        if hasattr(backend, "close"):
            backend.close()
        if stats.n_total:
            summary = stats.write_report(resolve_path(args.reports_dir), extra={
                "backend": args.backend,
                "threshold": threshold,
                "active_model_dir": active_dir,
                "n_hot_swaps": n_swaps,
                "output_topic": out_topic,
                "forward_all": args.forward_all,
            })
            print(f"[physics-scorer-p2] {json.dumps(summary, indent=2)}")
        else:
            print("[physics-scorer-p2] no messages consumed")


if __name__ == "__main__":
    main()
