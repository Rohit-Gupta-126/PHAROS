"""Export the Phase 0 VAE *encoder-to-mu* path to ONNX for the trigger score.

The L1-style trigger score is Sum mu^2, which only needs the encoder MLP up to
the latent means -- the decoder and logvar head are dead weight in the trigger
path, so we export a wrapper module ``x(57) -> mu(8)`` only. Batch size is
fixed at 1 (the trigger processes one event at a time) and we use opset 13
(Gemm/Relu only -- well inside TMVA SOFIE's supported operator set).

No retraining: this loads ``models/physics_vae/ckpt.pt`` frozen.

Run: ``python -m scripts.export_onnx``
"""
from __future__ import annotations

import argparse
import json

import torch

from src.common.config import resolve_path
from scripts.eval_physics import _load_model


class EncoderMu(torch.nn.Module):
    """Encoder trunk + mu head of the trained VAE (the trigger-score path)."""

    def __init__(self, vae) -> None:
        super().__init__()
        self.encoder = vae.encoder
        self.fc_mu = vae.fc_mu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_mu(self.encoder(x))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=str, default="models/physics_vae")
    p.add_argument("--out", type=str, default=None,
                   help="output path (default <model-dir>/encoder_mu.onnx)")
    p.add_argument("--opset", type=int, default=13)
    args = p.parse_args(argv)

    model_dir = resolve_path(args.model_dir)
    out = resolve_path(args.out) if args.out else model_dir / "encoder_mu.onnx"

    device = torch.device("cpu")  # export on CPU; weights are tiny
    vae = _load_model(model_dir, device)
    enc = EncoderMu(vae).eval()

    dummy = torch.zeros(1, vae.input_dim, dtype=torch.float32)
    torch.onnx.export(
        enc, dummy, str(out),
        input_names=["features"], output_names=["mu"],
        opset_version=args.opset,
        dynamo=False,  # legacy exporter: plain Gemm/Relu graph, SOFIE-friendly
    )

    with torch.no_grad():
        mu = enc(dummy)
    meta = {
        "source_ckpt": str(model_dir / "ckpt.pt"),
        "input": {"name": "features", "shape": [1, vae.input_dim]},
        "output": {"name": "mu", "shape": [1, vae.latent_dim]},
        "opset": args.opset,
        "score": "sum_mu2 (sum of squared latent means)",
    }
    (out.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    print(f"[export-onnx] wrote {out} "
          f"(57 -> mu[{vae.latent_dim}], opset {args.opset}); "
          f"zero-input mu[0]={mu[0, 0]:.6g}")


if __name__ == "__main__":
    main()
