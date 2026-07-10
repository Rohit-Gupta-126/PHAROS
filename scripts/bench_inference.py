"""Per-event inference latency: PyTorch encoder vs ONNX Runtime (vs SOFIE).

Offline benchmark on a held-out background sample, batch 1 per event (the
trigger processing model), all on CPU for a like-for-like comparison. Also
re-verifies that every deploy backend reproduces the PyTorch Sum mu^2 within
``--tol``. Writes ``reports/phase2/inference_latency.{json,png}``.

The SOFIE backend is included automatically iff the C++ binary has been built
(services/inference_sofie/sofie_score); otherwise it is reported as skipped.

Run: ``python -m scripts.bench_inference``
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np

from src.common.config import resolve_path
from src.preprocessing.adc2021 import Normalizer, load_events
from services.scorers.trigger_backends import (OrtBackend, SofieBackend,
                                               TorchBackend)


def bench(backend, x: np.ndarray, warmup: int = 100) -> dict:
    backend.score(x[:warmup])
    lat_us = np.empty(len(x))
    for i in range(len(x)):
        t0 = time.perf_counter_ns()
        backend.score(x[i:i + 1])
        lat_us[i] = (time.perf_counter_ns() - t0) / 1e3
    return {
        "n_events": len(x),
        "latency_us_per_event": {
            "mean": float(lat_us.mean()),
            "p50": float(np.percentile(lat_us, 50)),
            "p95": float(np.percentile(lat_us, 95)),
            "p99": float(np.percentile(lat_us, 99)),
            "max": float(lat_us.max()),
        },
        "throughput_events_per_sec": float(1e6 / lat_us.mean()),
        "_lat_us": lat_us,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=str, default="models/physics_vae")
    p.add_argument("--background", type=str,
                   default="data/raw/adc2021/background_for_training.h5")
    p.add_argument("--n-events", type=int, default=5000)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--reports-dir", type=str, default="reports/phase2")
    args = p.parse_args(argv)

    model_dir = resolve_path(args.model_dir)
    reports_dir = resolve_path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    normalizer = Normalizer.load(model_dir / "norm.npz")
    raw = load_events(args.background, max_events=args.n_events, seed=1339,
                      region=(0.9, 1.0))
    x = normalizer.transform(raw)

    backends = {"pytorch": TorchBackend(model_dir), "ort": OrtBackend(model_dir)}
    sofie_status = "skipped: binary not built (see services/inference_sofie/README.md)"
    try:
        backends["sofie"] = SofieBackend(model_dir)
        sofie_status = "benchmarked"
    except FileNotFoundError:
        pass

    results: dict = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "sample": f"{len(x)} background events, batch 1, CPU",
        "tolerance": args.tol,
        "sofie": sofie_status,
        "backends": {},
    }
    ref_scores = backends["pytorch"].score(x)
    lat_curves = {}
    for name, be in backends.items():
        r = bench(be, x)
        lat_curves[name] = r.pop("_lat_us")
        scores = be.score(x) if name != "pytorch" else ref_scores
        max_diff = float(np.abs(scores - ref_scores).max())
        r["max_abs_score_diff_vs_pytorch"] = max_diff
        r["parity_pass"] = bool(max_diff <= args.tol)
        results["backends"][name] = r
        print(f"[bench] {name}: mean={r['latency_us_per_event']['mean']:.1f} us "
              f"p99={r['latency_us_per_event']['p99']:.1f} us "
              f"max|dscore|={max_diff:.2e} "
              f"parity={'PASS' if r['parity_pass'] else 'FAIL'}")
        if hasattr(be, "close"):
            be.close()

    (reports_dir / "inference_latency.json").write_text(
        json.dumps(results, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, lat in lat_curves.items():
        clip = np.percentile(lat, 99.5)
        ax.hist(np.clip(lat, 0, clip), bins=80, alpha=0.55,
                label=f"{name} (p50={np.percentile(lat, 50):.0f} us)")
    ax.set_xlabel("per-event inference latency (us, batch 1, CPU)")
    ax.set_ylabel("events")
    ax.set_title("PHAROS trigger inference: deploy backends vs PyTorch")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(reports_dir / "inference_latency.png", dpi=120)
    print(f"[bench] wrote {reports_dir / 'inference_latency.json'} and .png")

    if not all(r["parity_pass"] for r in results["backends"].values()):
        raise SystemExit("backend parity FAILED")


if __name__ == "__main__":
    main()
