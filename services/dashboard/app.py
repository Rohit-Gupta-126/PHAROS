"""PHAROS Phase 3 live dashboard (Streamlit, host process, no DB).

Reads the Kafka topics directly: one confluent-kafka Consumer per background
thread (the client is NOT thread-safe -- never share one), each with a fresh
per-run group starting at end-of-topic, feeding bounded deques. The UI
refreshes every 2 s via ``st.fragment``.

Panels: live throughput + keep-rate per stream, reference-vs-current score
histograms (reference = frozen Phase 0 ``reference_stats.json`` samples),
kept-vs-dropped counts, and the ``alerts.drift`` feed.

Run: ``streamlit run services/dashboard/app.py``  (make dashboard)
"""
from __future__ import annotations

import json
import threading
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import streamlit as st

from services import common

REFRESH_S = 2.0
MAXLEN = 5000

# dataviz palette (light surface).
C_REF = "#2a78d6"      # reference series (blue, slot 1)
C_CUR = "#eda100"      # current window (yellow, slot 3 -- CVD-safe vs blue)
C_TEXT2 = "#52514e"
SEV_COLOR = {"warn": "#eda100", "alert": "#d03b3b"}


class TopicTail:
    """Background thread tailing one topic into bounded deques."""

    def __init__(self, topic: str, bootstrap: str) -> None:
        self.topic = topic
        self.records: deque = deque(maxlen=MAXLEN)
        self.times: deque = deque(maxlen=MAXLEN)   # arrival wall-clock (s)
        self.n_total = 0
        self.kept = Counter()                      # kept True/False
        self._thread = threading.Thread(
            target=self._run, args=(bootstrap,), daemon=True)
        self._thread.start()

    def _run(self, bootstrap: str) -> None:
        consumer = common.make_consumer(
            self.topic, common.fresh_group("pharos-dash"), bootstrap,
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
                self.kept[bool(r.get("kept",
                                     r["score"] > r["threshold"]))] += 1

    def rate(self, horizon_s: float = 10.0) -> float:
        now = time.monotonic()
        n = sum(1 for t in self.times if now - t <= horizon_s)
        return n / horizon_s

    def scores(self, key: str = "score") -> np.ndarray:
        return np.asarray([r[key] for r in self.records if key in r])


@st.cache_resource
def get_tails(bootstrap: str) -> dict:
    return {
        "physics": TopicTail("events.physics.scored", bootstrap),
        "pdm": TopicTail(common.TOPIC_PDM_SCORED, bootstrap),
        "drift": TopicTail(common.TOPIC_DRIFT, bootstrap),
    }


@st.cache_data
def load_reference_samples() -> dict:
    out = {}
    phys = Path("models/physics_vae/reference_stats.json")
    if phys.exists():
        out["physics"] = np.asarray(
            json.loads(phys.read_text())["score"]["samples"])
    pdm = Path("models/pdm/reference_stats.json")
    if pdm.exists():
        d = json.loads(pdm.read_text())
        out["pdm"] = {s: np.asarray(v["score"]["samples"])
                      for s, v in d["systems"].items()}
    return out


def hist_overlay(ref: np.ndarray, cur: np.ndarray, title: str,
                 log_x: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    both = np.concatenate([ref, cur]) if cur.size else ref
    if log_x:
        both = both[both > 0]
        bins = np.geomspace(both.min(), both.max(), 40)
        ax.set_xscale("log")
    else:
        bins = np.linspace(both.min(), both.max(), 40)
    ax.hist(ref, bins=bins, density=True, histtype="stepfilled", alpha=0.35,
            color=C_REF, edgecolor=C_REF, linewidth=1.5, label="reference")
    if cur.size:
        ax.hist(cur, bins=bins, density=True, histtype="step", linewidth=2,
                color=C_CUR, label=f"current (n={cur.size})")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(title, fontsize=10)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="PHAROS Phase 3", layout="wide")
    st.title("PHAROS — live drift monitor")
    bootstrap = st.sidebar.text_input("Kafka bootstrap",
                                      common.BOOTSTRAP_DEFAULT)
    tails = get_tails(bootstrap)
    refs = load_reference_samples()

    @st.fragment(run_every=REFRESH_S)
    def body() -> None:
        phys, pdm, drift = tails["physics"], tails["pdm"], tails["drift"]

        cols = st.columns(4)
        cols[0].metric("physics rate (ev/s, 10 s)", f"{phys.rate():.0f}")
        cols[1].metric("pdm rate (ev/s, 10 s)", f"{pdm.rate():.1f}")
        for c, tail, name in ((cols[2], phys, "physics"),
                              (cols[3], pdm, "pdm")):
            kept, dropped = tail.kept[True], tail.kept[False]
            tot = kept + dropped
            c.metric(f"{name} keep-rate",
                     f"{kept / tot:.2%}" if tot else "—",
                     f"{kept} kept / {dropped} dropped",
                     delta_color="off")

        left, right = st.columns(2)
        with left:
            st.subheader("Score: reference vs current")
            if "physics" in refs:
                st.pyplot(hist_overlay(refs["physics"], phys.scores(),
                                       "physics Σμ² (log x)", log_x=True),
                          clear_figure=True)
            pdm_scores = pdm.records
            if "pdm" in refs and pdm_scores:
                sys_counts = Counter(r["system"] for r in pdm_scores)
                system = sys_counts.most_common(1)[0][0]
                cur = np.asarray([r["score"] for r in pdm_scores
                                  if r["system"] == system])
                st.pyplot(hist_overlay(refs["pdm"][system], cur,
                                       f"pdm/{system} recon MSE (log x)",
                                       log_x=True), clear_figure=True)

        with right:
            st.subheader("Drift alerts")
            events = [r for r in drift.records if r["severity"] != "ok"]
            if not events:
                st.caption("no warn/alert drift events yet "
                           f"({drift.n_total} evaluations seen)")
            for r in list(reversed(events))[:25]:
                color = SEV_COLOR[r["severity"]]
                st.markdown(
                    f"<span style='color:{color};font-weight:600'>"
                    f"{r['severity'].upper()}</span> "
                    f"`{r['stream']}` {r['metric']} = {r['value']:.3f} "
                    f"<span style='color:{C_TEXT2}'>(n={r['window_n']})</span>",
                    unsafe_allow_html=True)

        st.caption(f"consumed: physics={phys.n_total} pdm={pdm.n_total} "
                   f"drift={drift.n_total} — refresh {REFRESH_S:.0f}s, "
                   "consumers start at end-of-topic")

    body()


main()
