"""Post-hoc detection lead time: injection marker vs first drift alert.

Reads the injection marker written by an injector, drains ``alerts.drift``
and the relevant scored topic from the beginning, and computes how long the
monitor took to raise its first ALERT after the injection started -- in wall
seconds and in scored messages. Also dumps every drift event to
``reports/phase3/drift_events.json`` and renders the PSI timeline with the
injection marker to ``reports/phase3/drift_timeline.png``.

Run AFTER the monitor has exited (so alerts.drift is complete):
``python -m tools.inject.measure_lead_time``
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from src.common.config import resolve_path
from services import common

# dataviz palette (light surface): categorical slots + status colors.
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7"]
C_WARN = "#eda100"
C_ALERT = "#d03b3b"
C_MARKER = "#52514e"


def drain_topic(topic: str, bootstrap: str, idle: float = 5.0
                ) -> List[Dict[str, Any]]:
    consumer = common.make_consumer(topic, common.fresh_group("pharos-leadtime"),
                                    bootstrap, from_beginning=True)
    try:
        return list(common.consume_json(consumer, idle_timeout_s=idle))
    finally:
        consumer.close()


def plot_timeline(events: List[Dict[str, Any]], start_ts_ns: int,
                  stream: str, out_png, warn: float, alert: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = sorted({e["metric"] for e in events})
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for metric, color in zip(metrics, SERIES):
        pts = [e for e in events if e["metric"] == metric]
        t = [(e["detected_ts_ns"] - start_ts_ns) / 1e9 for e in pts]
        v = [e["value"] for e in pts]
        ax.plot(t, v, color=color, linewidth=2, marker="o", markersize=4,
                label=metric)
        ax.annotate(metric, xy=(t[-1], v[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, color="#52514e",
                    va="center")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.axhline(warn, color=C_WARN, linewidth=1.2, linestyle="--")
    ax.annotate(f"warn ({warn})", xy=(1, warn), xycoords=("axes fraction", "data"),
                xytext=(-4, 4), textcoords="offset points", ha="right",
                fontsize=8, color="#52514e")
    ax.axhline(alert, color=C_ALERT, linewidth=1.2, linestyle="--")
    ax.annotate(f"alert ({alert})", xy=(1, alert),
                xycoords=("axes fraction", "data"), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=8,
                color="#52514e")
    ax.axvline(0, color=C_MARKER, linewidth=1.2, linestyle=":")
    ax.annotate("injection start", xy=(0, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -12), textcoords="offset points", fontsize=8,
                color="#52514e")
    ax.set_xlabel("seconds since injection start")
    ax.set_ylabel("PSI")
    ax.set_title(f"Drift timeline ({stream}) vs frozen Phase 0 reference")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--marker", type=str,
                   default="reports/phase3/injection_marker.json")
    p.add_argument("--stream", type=str, default="physics",
                   help="drift-event stream to match (physics or pdm/<SYS>)")
    p.add_argument("--metric", type=str, default=None,
                   help="restrict to one metric (default: first alert on ANY "
                        "metric of the stream counts as detection)")
    p.add_argument("--scored-topic", type=str, default="events.physics.scored")
    p.add_argument("--reports-dir", type=str, default="reports/phase3")
    p.add_argument("--bootstrap", type=str, default=common.BOOTSTRAP_DEFAULT)
    args = p.parse_args(argv)

    marker = json.loads(resolve_path(args.marker).read_text())
    start = marker["start_ts_ns"]
    out = resolve_path(args.reports_dir)
    out.mkdir(parents=True, exist_ok=True)

    drift = drain_topic(common.TOPIC_DRIFT, args.bootstrap)
    (out / "drift_events.json").write_text(json.dumps(drift, indent=2),
                                           encoding="utf-8")
    print(f"[lead-time] {len(drift)} drift events -> drift_events.json")

    sel = [e for e in drift if e["stream"] == args.stream
           and (args.metric is None or e["metric"] == args.metric)]
    first_alert = next((e for e in sel if e["severity"] == "alert"
                        and e["detected_ts_ns"] >= start), None)
    first_warn = next((e for e in sel if e["severity"] in ("warn", "alert")
                       and e["detected_ts_ns"] >= start), None)

    result: Dict[str, Any] = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "stream": args.stream,
        "metric_filter": args.metric or "any",
        "injection_start_ts_ns": start,
        "inject_source": marker.get("inject_source"),
        "n_drift_events": len(sel),
    }
    if first_alert is None:
        result["detected"] = False
        print("[lead-time] NO alert-severity drift event after injection "
              "-- detection failed (recorded honestly)")
    else:
        scored = drain_topic(args.scored_topic, args.bootstrap)
        n_msgs = sum(1 for r in scored
                     if start <= r["scored_ts_ns"]
                     <= first_alert["detected_ts_ns"])
        result.update({
            "detected": True,
            "lead_time_s": (first_alert["detected_ts_ns"] - start) / 1e9,
            "lead_time_messages": n_msgs,
            "alert_metric": first_alert["metric"],
            "first_alert": first_alert,
            "first_warn_metric": first_warn["metric"],
            "first_warn_lead_s":
                (first_warn["detected_ts_ns"] - start) / 1e9,
        })
        print(f"[lead-time] ALERT ({first_alert['metric']}) after "
              f"{result['lead_time_s']:.2f}s / {n_msgs} scored messages "
              f"(first warn: {first_warn['metric']} at "
              f"{result['first_warn_lead_s']:.2f}s)")

    (out / "lead_time.json").write_text(json.dumps(result, indent=2),
                                        encoding="utf-8")
    if sel:
        warn_thr = sel[0]["threshold_warn"]
        alert_thr = sel[0]["threshold_alert"]
        plot_timeline(sel, start, args.stream, out / "drift_timeline.png",
                      warn_thr, alert_thr)
        print(f"[lead-time] wrote {out / 'drift_timeline.png'}")


if __name__ == "__main__":
    main()
