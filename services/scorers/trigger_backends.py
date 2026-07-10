"""Interchangeable trigger-score backends: Sum mu^2 from normalized features.

All backends score event-at-a-time (batch 1) -- the trigger-realistic mode --
even when handed an (N, 57) array. Inputs are ALREADY normalized; the caller
owns the log1p/z-score transform.

- ``TorchBackend``  : Phase 0 checkpoint encoder (reference).
- ``OrtBackend``    : onnxruntime on ``encoder_mu.onnx`` (runnable deploy path).
- ``SofieBackend``  : the standalone SOFIE C++ binary over a line protocol
                      (future work until ``services/inference_sofie/sofie_score``
                      is built; see that directory's README).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


class TorchBackend:
    name = "pytorch"

    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        import torch
        from scripts.eval_physics import _load_model
        self._torch = torch
        self.device = torch.device(device)
        self.model = _load_model(model_dir, self.device)

    def score(self, x: np.ndarray) -> np.ndarray:
        torch = self._torch
        out = np.empty(len(x), dtype=np.float64)
        with torch.no_grad():
            for i in range(len(x)):
                xi = torch.from_numpy(x[i:i + 1]).to(self.device)
                mu, _ = self.model.encode(xi)
                out[i] = float(torch.sum(mu * mu))
        return out


class OrtBackend:
    name = "ort"

    def __init__(self, model_dir: Path) -> None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # tiny model; threading only adds jitter
        self.sess = ort.InferenceSession(str(model_dir / "encoder_mu.onnx"),
                                         sess_options=opts,
                                         providers=["CPUExecutionProvider"])

    def score(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(len(x), dtype=np.float64)
        for i in range(len(x)):
            mu = self.sess.run(["mu"], {"features": x[i:i + 1]})[0]
            out[i] = float(np.sum(mu.astype(np.float64) ** 2))
        return out


class SofieBackend:
    name = "sofie"
    BINARY = Path(__file__).resolve().parents[1] / "inference_sofie" / "sofie_score"

    def __init__(self, model_dir: Path | None = None) -> None:
        if not self.BINARY.exists():
            raise FileNotFoundError(
                f"{self.BINARY} not built -- SOFIE is future work on this "
                "machine (see services/inference_sofie/README.md); "
                "use --backend ort")
        self.proc = subprocess.Popen(
            [str(self.BINARY)], cwd=str(self.BINARY.parent),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            bufsize=1)

    def score(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(len(x), dtype=np.float64)
        for i, row in enumerate(x):
            self.proc.stdin.write(" ".join(f"{v:.9g}" for v in row) + "\n")
            self.proc.stdin.flush()
            out[i] = float(self.proc.stdout.readline())
        return out

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=5)


def make_backend(name: str, model_dir: Path):
    if name == "pytorch":
        return TorchBackend(model_dir)
    if name == "ort":
        return OrtBackend(model_dir)
    if name == "sofie":
        return SofieBackend(model_dir)
    raise ValueError(f"unknown backend {name!r}")
