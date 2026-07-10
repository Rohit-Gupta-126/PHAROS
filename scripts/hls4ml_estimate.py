"""hls4ml feasibility estimate for the trigger encoder (NO Vivado/Vitis).

Converts the frozen Phase 0 encoder (PyTorch frontend, fixed batch 1) into an
hls4ml Vitis-HLS project with fixed-point quantization (post-training
``ap_fixed<24,8>`` -- see the ``--precision`` note for why the tutorial
default ``<16,6>`` is insufficient here), then:

- compiles the C-emulation library (g++ only) and checks numerical agreement
  of Sum mu^2 against PyTorch on a background sample;
- writes the generated project + a resource/latency ESTIMATE based on model
  arithmetic (MACs, parameter count, precision) to
  ``reports/phase2/hls4ml_estimate.json``.

Full C-synthesis latency/resource numbers require Vitis HLS (>60 GB install)
-- run elsewhere per ``docs/hls4ml_synthesis.md``. QKeras QAT is intentionally
NOT used: it is deprecated/broken on Keras 3-era stacks; quantization here is
post-training fixed-point via hls4ml precision config (see design log).

Run: ``python -m scripts.hls4ml_estimate``
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.common.config import resolve_path
from src.preprocessing.adc2021 import Normalizer, load_events
from scripts.eval_physics import _load_model
from scripts.export_onnx import EncoderMu


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=str, default="models/physics_vae")
    p.add_argument("--background", type=str,
                   default="data/raw/adc2021/background_for_training.h5")
    p.add_argument("--n-events", type=int, default=2000)
    p.add_argument("--out-dir", type=str, default="models/physics_vae/hls4ml_prj")
    # <16,6> (the tutorial default) breaks trigger parity here: mu values sit
    # near a ~1e-3 threshold, so 10 fractional bits is too coarse (91% p99
    # decision agreement). <24,8> restores 100% agreement (max mu diff ~6e-4).
    p.add_argument("--precision", type=str, default="ap_fixed<24,8>")
    p.add_argument("--reuse-factor", type=int, default=1,
                   help="1 = fully parallel (lowest latency, most DSPs)")
    p.add_argument("--reports-dir", type=str, default="reports/phase2")
    args = p.parse_args(argv)

    import hls4ml

    model_dir = resolve_path(args.model_dir)
    vae = _load_model(model_dir, torch.device("cpu"))
    enc = EncoderMu(vae).eval()

    config = hls4ml.utils.config_from_pytorch_model(
        enc, (57,), granularity="name",
        default_precision=args.precision,
        default_reuse_factor=args.reuse_factor)
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        enc, hls_config=config,
        output_dir=str(resolve_path(args.out_dir)),
        backend="Vitis", io_type="io_parallel",
        project_name="pharos_encoder")
    hls_model.compile()  # g++ C-emulation only; no HLS synthesis

    # Numerical check: fixed-point emulation vs float PyTorch.
    normalizer = Normalizer.load(model_dir / "norm.npz")
    raw = load_events(args.background, max_events=args.n_events, seed=1340,
                      region=(0.9, 1.0))
    x = normalizer.transform(raw)
    with torch.no_grad():
        mu_ref = enc(torch.from_numpy(x)).numpy()
    mu_hls = hls_model.predict(np.ascontiguousarray(x))
    score_ref = np.sum(mu_ref.astype(np.float64) ** 2, axis=1)
    score_hls = np.sum(mu_hls.astype(np.float64) ** 2, axis=1)
    # Rank agreement matters for a trigger: does quantization reorder events
    # around the p99 threshold?
    thr = np.percentile(score_ref, 99)
    agree = float(np.mean((score_ref > thr) == (score_hls > thr)))

    # Static arithmetic estimate (full synthesis numbers come from Vitis HLS).
    dims = [57, *[m.out_features for m in enc.modules()
                  if isinstance(m, torch.nn.Linear)]]
    macs = sum(a * b for a, b in zip(dims[:-1], dims[1:]))
    params = sum(p_.numel() for p_ in enc.parameters())
    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "tool": f"hls4ml {hls4ml.__version__}, backend Vitis, io_parallel",
        "precision": args.precision,
        "reuse_factor": args.reuse_factor,
        "quantization": "post-training fixed-point (QKeras QAT deferred; "
                        "deprecated on Keras 3 stacks)",
        "network": {"layers": dims, "parameters": int(params),
                    "macs_per_event": int(macs)},
        "static_estimate": {
            "dsp_multipliers_at_reuse_1": int(macs),
            "note": "at ReuseFactor=1 each MAC maps to ~1 DSP48; RF=R divides "
                    "DSPs by R and multiplies II by R. ~2.4k DSPs exceeds a "
                    "mid-range part, so RF 4-8 or 8-bit weights are the "
                    "realistic deployment point.",
            "pipeline_latency_cycles_order": "O(layers) ~ 10-30 cycles at "
                                             "II=1, i.e. ~50-150 ns at 200 MHz",
        },
        "c_emulation_check": {
            "n_events": len(x),
            "max_abs_mu_diff_vs_float": float(np.abs(mu_ref - mu_hls).max()),
            "trigger_decision_agreement_at_p99": agree,
        },
        "full_synthesis": "requires Vitis HLS -- see docs/hls4ml_synthesis.md",
    }
    out = resolve_path(args.reports_dir) / "hls4ml_estimate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
