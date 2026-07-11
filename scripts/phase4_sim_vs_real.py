"""Phase 4: measure the sim-to-real domain gap and report it (not tune it away).

Feeds the RDataFrame-ingested CMS Open Data stream through the FROZEN,
sim-trained ORT scorer + drift monitor and records the resulting PSI/KS against
the Phase 0 simulation reference. Real CMS data has a different object
composition than the Delphes ADC2021 sim, so the tracked features and the
anomaly score WILL drift -- that is the finding (a legitimate domain gap), not a
bug. Whatever the monitor emits is written verbatim to
``reports/phase4/sim_vs_real_drift.json``.

Orchestration (all reusing existing components):
  1. subscribe to ``alerts.drift`` (fresh group, latest) BEFORE anything emits;
  2. launch ORT scorer (--forward-all) and drift monitor as subprocesses;
  3. stream the CMS 57-vectors to ``events.physics`` (services.ingest_root.stream_cms);
  4. drain the drift feed until idle, aggregate per metric, tear down, report.

Prereqs: broker up (make up); a CMS npy from ingest_nanoaod.py; thresholds +
reference stats derived (Phases 1/3). Run: ``python -m scripts.phase4_sim_vs_real``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import resolve_path
from services import common


def _drain_drift(events: list, stop: threading.Event, bootstrap: str) -> None:
    """Background: collect every alerts.drift record until told to stop."""
    consumer = common.make_consumer(common.TOPIC_DRIFT,
                                    common.fresh_group("pharos-phase4"),
                                    bootstrap, from_beginning=False)
    try:
        while not stop.is_set():
            msg = consumer.poll(0.5)
            if msg is None or msg.error():
                continue
            events.append(json.loads(msg.value().decode("utf-8")))
    finally:
        consumer.close()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source-npy", default="data/interim/cms_events_57.npy")
    p.add_argument("--rate", type=float, default=500.0)
    p.add_argument("--limit", type=int, default=10_000)
    p.add_argument("--reports-dir", default="reports/phase4")
    p.add_argument("--bootstrap", default=common.BOOTSTRAP_DEFAULT)
    args = p.parse_args(argv)

    py = sys.executable
    events: list = []
    stop = threading.Event()
    drift_thread = threading.Thread(
        target=_drain_drift, args=(events, stop, args.bootstrap), daemon=True)
    drift_thread.start()

    print("[phase4] launching ORT scorer + drift monitor...")
    scorer = subprocess.Popen(
        [py, "-m", "services.scorers.physics_scorer_sofie", "--backend", "ort",
         "--forward-all", "--reports-dir", args.reports_dir, "--idle", "30"])
    monitor = subprocess.Popen(
        [py, "-m", "services.monitor.drift_monitor",
         "--reports-dir", args.reports_dir, "--idle", "35"])
    # Let both reach end-of-topic before we start producing.
    import time
    time.sleep(5)

    print(f"[phase4] streaming {args.limit} CMS events at {args.rate} ev/s...")
    subprocess.run(
        [py, "-m", "services.ingest_root.stream_cms",
         "--source-npy", args.source_npy, "--rate", str(args.rate),
         "--limit", str(args.limit)], check=True)

    print("[phase4] stream done; waiting for scorer + monitor to idle out...")
    scorer.wait()
    monitor.wait()
    stop.set()
    drift_thread.join(timeout=5)

    # Aggregate: per (stream, metric), the peak + last PSI and its severity.
    per_metric: dict = defaultdict(lambda: {"n": 0, "max_psi": 0.0,
                                             "last_psi": 0.0, "severity": "ok",
                                             "last_ks": None})
    sev_rank = {"ok": 0, "warn": 1, "alert": 2}
    for ev in events:
        key = f"{ev['stream']}::{ev['metric']}"
        m = per_metric[key]
        m["n"] += 1
        m["last_psi"] = ev["value"]
        m["max_psi"] = max(m["max_psi"], ev["value"])
        if sev_rank[ev["severity"]] >= sev_rank[m["severity"]]:
            m["severity"] = ev["severity"]
        if "ks" in ev:
            m["last_ks"] = ev["ks"]

    worst = sorted(per_metric.items(), key=lambda kv: -kv[1]["max_psi"])
    report = {
        "schema": "pharos.sim_vs_real.v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "note": ("CMS Open Data (real) streamed through the frozen Phase 0 "
                 "sim-trained scorer + monitor. Drift here is the sim-to-real "
                 "domain gap (different object composition than ADC2021 "
                 "Delphes sim), reported as a finding -- NOT tuned away."),
        "source_npy": args.source_npy,
        "events_streamed": args.limit,
        "drift_evaluations": len(events),
        "n_metrics_alert": sum(1 for _, m in per_metric.items()
                               if m["severity"] == "alert"),
        "n_metrics_warn": sum(1 for _, m in per_metric.items()
                              if m["severity"] == "warn"),
        "by_metric": [
            {"metric": k, **v} for k, v in worst],
    }
    out = resolve_path(args.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "sim_vs_real_drift.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[phase4] wrote {dest}")
    print(f"[phase4] {report['drift_evaluations']} evaluations; "
          f"alert metrics={report['n_metrics_alert']} "
          f"warn={report['n_metrics_warn']}")
    for row in report["by_metric"][:8]:
        print(f"  {row['severity'].upper():5} {row['metric']:28} "
              f"max_psi={row['max_psi']:.3f}")


if __name__ == "__main__":
    main()
