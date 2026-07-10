"""Verify the exported encoder ONNX against the PyTorch encoder.

Runs the same held-out background sample (Phase 0 eval region, tail 10% of the
training file) through (a) the PyTorch encoder-to-mu path and (b) onnxruntime
on ``encoder_mu.onnx`` (batch-1 loop, matching the trigger), and asserts both
``mu`` and the derived Sum mu^2 score agree within ``--tol`` (default 1e-5).
Writes a short report to ``reports/phase2/onnx_parity.txt``.

Run: ``python -m scripts.verify_onnx``
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import onnxruntime as ort
import torch

from src.common.config import resolve_path
from src.preprocessing.adc2021 import Normalizer, load_events
from scripts.eval_physics import _load_model


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=str, default="models/physics_vae")
    p.add_argument("--background", type=str,
                   default="data/raw/adc2021/background_for_training.h5")
    p.add_argument("--n-events", type=int, default=2000)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--report", type=str, default="reports/phase2/onnx_parity.txt")
    args = p.parse_args(argv)

    model_dir = resolve_path(args.model_dir)
    onnx_path = model_dir / "encoder_mu.onnx"
    device = torch.device("cpu")
    model = _load_model(model_dir, device)
    normalizer = Normalizer.load(model_dir / "norm.npz")

    raw = load_events(args.background, max_events=args.n_events, seed=1338,
                      region=(0.9, 1.0))
    x = normalizer.transform(raw)

    # (a) PyTorch reference.
    with torch.no_grad():
        mu_pt, _ = model.encode(torch.from_numpy(x))
    mu_pt = mu_pt.numpy()

    # (b) onnxruntime, batch-1 loop as in the trigger path.
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    mu_ort = np.concatenate([
        sess.run(["mu"], {"features": x[i:i + 1]})[0] for i in range(len(x))
    ])

    score_pt = np.sum(mu_pt ** 2, axis=1)
    score_ort = np.sum(mu_ort ** 2, axis=1)
    mu_max = float(np.abs(mu_pt - mu_ort).max())
    score_max = float(np.abs(score_pt - score_ort).max())
    ok = mu_max <= args.tol and score_max <= args.tol

    report = "\n".join([
        "PHAROS Phase 2 -- ONNX export parity (PyTorch encoder vs onnxruntime)",
        f"generated: {datetime.now(timezone.utc).isoformat()}",
        f"model:     {onnx_path}",
        f"sample:    {len(x)} background events (held-out region 0.9-1.0)",
        f"tolerance: {args.tol:g}",
        "",
        f"max |mu_pytorch - mu_onnx|            = {mu_max:.3e}",
        f"max |summu2_pytorch - summu2_onnx|    = {score_max:.3e}",
        f"score range on sample: [{score_pt.min():.3e}, {score_pt.max():.3e}]",
        "",
        f"RESULT: {'PASS' if ok else 'FAIL'}",
        "",
    ])
    out = resolve_path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)
    if not ok:
        raise SystemExit("ONNX parity check FAILED")


if __name__ == "__main__":
    main()
