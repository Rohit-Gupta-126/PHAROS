"""Score ``events.pdm`` with the frozen Phase 0 conv AEs; republish keepers.

Loads each system's Phase 0 per-channel normalizer + conv-AE checkpoint (no
refit), applies the SAME z-score transform, computes reconstruction MSE, and
republishes ``{event_id, system, score, ts}`` to ``alerts.pdm`` for pulses
above the per-system derived threshold in ``configs/thresholds.json``.

Exits after ``--idle`` seconds with no data and writes latency/throughput/
keep-rate stats to ``reports/phase1/``.

Run: ``python -m services.scorers.pdm_scorer``
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
import torch

from src.common.config import resolve_path
from src.common.device import describe_device, get_device
from src.inference.scores import ae_recon_error
from src.preprocessing.hvcm import ChannelNormalizer

from scripts.eval_pdm import _load_ae
from services import common
from services.scorers.instrumentation import StreamStats


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=str, default="models/pdm")
    p.add_argument("--thresholds", type=str, default="configs/thresholds.json")
    p.add_argument("--reports-dir", type=str, default="reports/phase1")
    p.add_argument("--bootstrap", type=str, default=common.BOOTSTRAP_DEFAULT)
    p.add_argument("--group", type=str, default=None,
                   help="fixed consumer group (default: fresh per-run group "
                        "starting at end-of-topic -- offset hygiene)")
    p.add_argument("--idle", type=float, default=15.0)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    args = p.parse_args(argv)

    device = get_device(args.device)
    model_root = resolve_path(args.model_dir)
    thr_cfg = json.loads(resolve_path(args.thresholds).read_text())

    norms: Dict[str, ChannelNormalizer] = {}
    models: Dict[str, torch.nn.Module] = {}
    thresholds: Dict[str, float] = {}
    for system, entry in thr_cfg["pdm"].items():
        mdir = model_root / system
        norms[system] = ChannelNormalizer.load(mdir / "channel_norm.npz")
        models[system] = _load_ae(mdir, device)
        thresholds[system] = entry["threshold"]
    print(f"[pdm-scorer] device={describe_device(device)} systems="
          f"{list(thresholds)} thresholds={ {k: round(v, 5) for k, v in thresholds.items()} }")

    group = args.group or common.fresh_group("pharos-pdm-scorer")
    consumer = common.make_consumer(common.TOPIC_PDM, group,
                                    args.bootstrap,
                                    from_beginning=bool(args.group))
    producer = common.make_producer(args.bootstrap)
    stats = StreamStats("pdm")
    normal_total = 0
    normal_kept = 0
    scores_seen: List[float] = []

    try:
        for r in common.consume_json(consumer, idle_timeout_s=args.idle):
            system = r["system"]
            wave = np.asarray(r["wave"], dtype=np.float32)[None]  # (1, L, C)
            x = torch.from_numpy(
                norms[system].transform_channels_first(wave))
            score = float(ae_recon_error(models[system], x, device)[0])
            scored_ts = common.now_ns()
            kept = score > thresholds[system]
            stats.record(r["producer_ts_ns"], scored_ts, kept)
            scores_seen.append(score)
            if r.get("ground_truth") == "Run":
                normal_total += 1
                normal_kept += int(kept)
            if kept:
                common.produce_json(producer, common.TOPIC_ALERTS, {
                    "schema": common.SCHEMA_ALERT,
                    "event_id": r["event_id"],
                    "system": system,
                    "score": score,
                    "threshold": thresholds[system],
                    "producer_ts_ns": r["producer_ts_ns"],
                    "scored_ts_ns": scored_ts,
                }, key=r["event_id"])
                producer.poll(0)
    finally:
        producer.flush(30)
        consumer.close()
        if stats.n_total:
            summary = stats.write_report(resolve_path(args.reports_dir), extra={
                "thresholds": thresholds,
                "configured_percentile": next(
                    iter(thr_cfg["pdm"].values()))["percentile"],
                "expected_background_keep_rate":
                    thr_cfg["expected_background_keep_rate"],
                "normal_pulses_seen": normal_total,
                "normal_keep_rate":
                    normal_kept / normal_total if normal_total else None,
                "score_median": float(np.median(scores_seen)),
            })
            print(f"[pdm-scorer] {json.dumps(summary, indent=2)}")
        else:
            print("[pdm-scorer] no messages consumed")


if __name__ == "__main__":
    main()
