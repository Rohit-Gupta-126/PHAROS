"""PHAROS dashboard bridge: a read-only SSE shim over the Redpanda topics.

Browsers cannot speak the Kafka protocol, so this is the ONE small data shim
between Redpanda and the static ``dashboard_web/`` frontend. It is deliberately
*not* a UI framework -- all rendering/logic lives in the static JS; this process
only tails topics and serves JSON.

Design (mirrors the retired Streamlit dashboard's threading model):
- one confluent-kafka consumer per topic on its own daemon thread (the client is
  NOT thread-safe -- never share one), each a fresh per-run group starting at
  end-of-topic (offset hygiene), feeding bounded deques;
- stdlib ``ThreadingHTTPServer`` only -- no Flask/FastAPI/Streamlit.

Endpoints:
    GET /            -> dashboard_web/index.html
    GET /<asset>     -> static file from dashboard_web/
    GET /reference   -> frozen Phase 0 reference score samples (for the overlay)
    GET /events      -> text/event-stream: a compact snapshot pushed every ~1 s

Run: ``python -m services.dashboard_api.app --port 8070``  (make dashboard-api)
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.common.config import resolve_path
from services import common

WEB_DIR = resolve_path("dashboard_web")
MAXLEN = 2000
RATE_HORIZON_S = 10.0

_CTYPES = {".html": "text/html; charset=utf-8",
           ".css": "text/css; charset=utf-8",
           ".js": "text/javascript; charset=utf-8"}


class TopicTail:
    """Background thread tailing one topic into bounded deques."""

    def __init__(self, topic: str, bootstrap: str) -> None:
        self.topic = topic
        self.records: deque = deque(maxlen=MAXLEN)
        self.times: deque = deque(maxlen=MAXLEN)     # arrival monotonic (s)
        self.n_total = 0
        self.kept = Counter()                        # True/False
        self._thread = threading.Thread(target=self._run, args=(bootstrap,),
                                        daemon=True)
        self._thread.start()

    def _run(self, bootstrap: str) -> None:
        consumer = common.make_consumer(
            self.topic, common.fresh_group("pharos-bridge"), bootstrap,
            from_beginning=False)
        while True:
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            r = json.loads(msg.value().decode("utf-8"))
            self.records.append(r)
            self.times.append(time.monotonic())
            self.n_total += 1
            if "score" in r and "threshold" in r:
                self.kept[bool(r.get("kept", r["score"] > r["threshold"]))] += 1

    def rate(self) -> float:
        now = time.monotonic()
        n = sum(1 for t in self.times if now - t <= RATE_HORIZON_S)
        return n / RATE_HORIZON_S

    def recent_scores(self, limit: int = 600) -> list:
        return [r["score"] for r in list(self.records)[-limit:] if "score" in r]


class Bridge:
    """Owns the topic tails + the frozen reference and builds snapshots."""

    def __init__(self, bootstrap: str) -> None:
        self.physics = TopicTail("events.physics.scored", bootstrap)
        self.pdm = TopicTail(common.TOPIC_PDM_SCORED, bootstrap)
        self.drift = TopicTail(common.TOPIC_DRIFT, bootstrap)
        self.reference = self._load_reference()

    @staticmethod
    def _load_reference() -> dict:
        out = {}
        phys = resolve_path("models/physics_vae/reference_stats.json")
        if phys.exists():
            d = json.loads(phys.read_text())
            out["physics"] = d["score"]["samples"]
        pdm = resolve_path("models/pdm/reference_stats.json")
        if pdm.exists():
            d = json.loads(pdm.read_text())
            out["pdm"] = {s: v["score"]["samples"]
                          for s, v in d["systems"].items()}
        return out

    def _stream_snapshot(self, tail: TopicTail) -> dict:
        kept, dropped = tail.kept[True], tail.kept[False]
        tot = kept + dropped
        return {"rate": round(tail.rate(), 1), "kept": kept, "dropped": dropped,
                "keep_rate": (kept / tot) if tot else None,
                "n_total": tail.n_total, "scores": tail.recent_scores()}

    def snapshot(self) -> dict:
        # PDM: report the dominant system's recent scores for the overlay.
        pdm_recs = list(self.pdm.records)
        pdm_system = None
        pdm_scores: list = []
        if pdm_recs:
            sys_counts = Counter(r["system"] for r in pdm_recs if "system" in r)
            if sys_counts:
                pdm_system = sys_counts.most_common(1)[0][0]
                pdm_scores = [r["score"] for r in pdm_recs[-600:]
                              if r.get("system") == pdm_system]
        pdm_snap = self._stream_snapshot(self.pdm)
        pdm_snap["scores"] = pdm_scores
        pdm_snap["system"] = pdm_system

        drift_events = [r for r in list(self.drift.records)
                        if r.get("severity") != "ok"]
        return {
            "ts": time.time(),
            "physics": self._stream_snapshot(self.physics),
            "pdm": pdm_snap,
            "drift": {
                "n_total": self.drift.n_total,
                "events": list(reversed(drift_events))[:25],
            },
        }


def make_handler(bridge: Bridge):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quieter console
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/events":
                return self._sse()
            if path == "/reference":
                body = json.dumps(bridge.reference).encode("utf-8")
                return self._send(200, body, "application/json")
            # static files
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            fp = (WEB_DIR / rel).resolve()
            if not str(fp).startswith(str(WEB_DIR.resolve())) or not fp.is_file():
                return self._send(404, b"not found", "text/plain")
            ctype = _CTYPES.get(fp.suffix, "application/octet-stream")
            return self._send(200, fp.read_bytes(), ctype)

        def _sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    payload = json.dumps(bridge.snapshot())
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(1.0)
            except (BrokenPipeError, ConnectionResetError):
                return  # browser navigated away / reconnecting

    return Handler


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=8070)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--bootstrap", default=common.BOOTSTRAP_DEFAULT)
    args = p.parse_args(argv)

    bridge = Bridge(args.bootstrap)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(bridge))
    url = f"http://{args.host}:{args.port}/"
    print(f"[dashboard-api] SSE bridge on {url}  (topics: physics/pdm/drift)")
    print(f"[dashboard-api] open {url} in a browser  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard-api] stopped")


if __name__ == "__main__":
    main()
