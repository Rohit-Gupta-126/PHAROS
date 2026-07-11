"""Phase 3 drift monitor: sliding-window PSI/KS vs frozen Phase 0 references.

Consumes three streams with one consumer:

* ``events.physics.scored`` -- anomaly-score distribution (Stream A);
* ``events.physics``        -- raw input features at the tracked indices;
* ``events.pdm.scored``     -- per-system PDM scores + per-channel means
                               (``pdm_scorer --forward-all``).

Every evaluation point (window full, then every ``step`` samples) a drift
event is published to ``alerts.drift`` -- severity ``ok``/``warn``/``alert``
by the PSI heuristics in ``configs/monitor.yaml``; score metrics also carry a
two-sample KS test against the stored reference sample. The monitor is
deliberately *marker-blind*: it never reads ``ctrl.inject``, so detection
lead time (tools/inject/measure_lead_time.py) is an honest measurement.

Run: ``python -m services.monitor.drift_monitor``
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Tuple

import yaml
from confluent_kafka import Consumer, KafkaError

from src.common.config import resolve_path
from services import common
from services.monitor.drift_stats import (SlidingWindow, evaluate_window,
                                          ks_stat)
from services.monitor.reference import (PdmReferences, PhysicsReferences,
                                        Reference)


def consume_json_multi(consumer: Consumer, idle_timeout_s: float,
                       poll_s: float = 1.0
                       ) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Like common.consume_json but yields ``(topic, record)``."""
    idle = 0.0
    while True:
        msg = consumer.poll(poll_s)
        if msg is None:
            idle += poll_s
            if idle >= idle_timeout_s:
                return
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            raise RuntimeError(f"Kafka error: {msg.error()}")
        idle = 0.0
        yield msg.topic(), json.loads(msg.value().decode("utf-8"))


class Tracker:
    """One tracked quantity: reference + sliding window -> drift events."""

    def __init__(self, stream: str, ref: Reference, window: int, step: int,
                 warn: float, alert: float, with_ks: bool = False) -> None:
        self.stream = stream
        self.ref = ref
        self.window = SlidingWindow(window, step)
        self.warn = warn
        self.alert = alert
        self.with_ks = with_ks

    def add(self, value: float, ts_ns: int) -> Dict[str, Any] | None:
        if not self.window.add(value, ts_ns):
            return None
        ev = evaluate_window(self.ref.dist, self.window, self.warn, self.alert)
        if self.with_ks:
            ev["ks"] = ks_stat(self.ref.samples, self.window.values())
        return {"schema": common.SCHEMA_DRIFT, "stream": self.stream,
                "detected_ts_ns": common.now_ns(), **ev}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=str, default="configs/monitor.yaml")
    p.add_argument("--physics-ref", type=str,
                   default="models/physics_vae/reference_stats.json")
    p.add_argument("--pdm-ref", type=str,
                   default="models/pdm/reference_stats.json")
    p.add_argument("--reports-dir", type=str, default="reports/phase3")
    p.add_argument("--bootstrap", type=str, default=common.BOOTSTRAP_DEFAULT)
    p.add_argument("--group", type=str, default=None,
                   help="fixed consumer group (default: fresh per-run group "
                        "starting at end-of-topic -- offset hygiene)")
    p.add_argument("--idle", type=float, default=30.0)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(resolve_path(args.config).read_text())
    warn, alert = float(cfg["psi_warn"]), float(cfg["psi_alert"])
    phys_ref = PhysicsReferences.load(resolve_path(args.physics_ref))
    pdm_ref = PdmReferences.load(resolve_path(args.pdm_ref))

    pw, ps = int(cfg["physics"]["window"]), int(cfg["physics"]["step"])
    dw, ds = int(cfg["pdm"]["window"]), int(cfg["pdm"]["step"])
    trackers: Dict[str, Tracker] = {
        "phys_score": Tracker("physics", phys_ref.score, pw, ps, warn, alert,
                              with_ks=True)}
    feat_idx = list(cfg["physics_feature_indices"])
    for idx in feat_idx:
        key = f"f{idx:02d}"
        trackers[f"phys_{key}"] = Tracker("physics", phys_ref.features[key],
                                          pw, ps, warn, alert)
    for system, ref in pdm_ref.scores.items():
        trackers[f"pdm_score_{system}"] = Tracker(f"pdm/{system}", ref,
                                                  dw, ds, warn, alert,
                                                  with_ks=True)
    ch_idx = list(cfg["pdm_channel_indices"])
    for j in ch_idx:
        key = f"ch{j:02d}"
        trackers[f"pdm_{key}"] = Tracker(f"pdm/{pdm_ref.tracked_system}",
                                         pdm_ref.channel_means[key],
                                         dw, ds, warn, alert)

    topics = [common.TOPIC_PHYSICS, "events.physics.scored",
              common.TOPIC_PDM_SCORED]
    group = args.group or common.fresh_group("pharos-monitor")
    consumer = Consumer({
        "bootstrap.servers": args.bootstrap,
        "group.id": group,
        "auto.offset.reset": "earliest" if args.group else "latest",
        "enable.auto.commit": True,
        "fetch.message.max.bytes": 2_000_000,
    })
    consumer.subscribe(topics)
    producer = common.make_producer(args.bootstrap)
    print(f"[monitor] tracking {list(trackers)} on {topics} "
          f"(psi warn={warn} alert={alert})")

    n_seen = Counter()
    sev_counts = Counter()
    n_events = 0

    def emit(ev: Dict[str, Any] | None) -> None:
        nonlocal n_events
        if ev is None:
            return
        n_events += 1
        sev_counts[(ev["stream"], ev["metric"], ev["severity"])] += 1
        common.produce_json(producer, common.TOPIC_DRIFT, ev)
        producer.poll(0)
        if ev["severity"] != "ok":
            print(f"[monitor] {ev['severity'].upper()} {ev['stream']} "
                  f"{ev['metric']}={ev['value']:.4f} (n={ev['window_n']})")

    try:
        for topic, r in consume_json_multi(consumer, idle_timeout_s=args.idle):
            n_seen[topic] += 1
            if topic == "events.physics.scored":
                emit(trackers["phys_score"].add(r["score"], r["scored_ts_ns"]))
            elif topic == common.TOPIC_PHYSICS:
                ts = r["producer_ts_ns"]
                for idx in feat_idx:
                    emit(trackers[f"phys_f{idx:02d}"].add(
                        r["features"][idx], ts))
            elif topic == common.TOPIC_PDM_SCORED:
                ts = r["scored_ts_ns"]
                emit(trackers[f"pdm_score_{r['system']}"].add(r["score"], ts))
                if r["system"] == pdm_ref.tracked_system:
                    for j in ch_idx:
                        emit(trackers[f"pdm_ch{j:02d}"].add(
                            r["channel_means"][j], ts))
    finally:
        producer.flush(30)
        consumer.close()
        summary = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "consumed": dict(n_seen),
            "drift_events_emitted": n_events,
            "by_metric": [
                {"stream": s, "metric": m, "severity": sev, "count": c}
                for (s, m, sev), c in sorted(sev_counts.items())],
        }
        out = resolve_path(args.reports_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "monitor_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[monitor] {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
