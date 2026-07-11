"""Blue/green model pointer for hot-swapping the ORT scorer's model.

The pointer is a tiny JSON file (``models/physics_vae/current.json``) naming
the model directory the scorer should serve, plus the threshold that was
derived for it. The retrain pipeline writes it ONLY after the candidate
passes the ONNX parity gate; the scorer polls it every N messages and
reloads atomically when ``model_dir`` changes.

A pointer *file* (not a symlink -- Windows; not a control topic -- must
survive restarts and stay inspectable) written via ``os.replace`` so readers
never see a partial file.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ModelPointer:
    model_dir: str
    threshold: float
    swapped_at: str

    @classmethod
    def read(cls, path: Path) -> "ModelPointer | None":
        """Return the pointer, or None if absent/corrupt (keep serving)."""
        try:
            d = json.loads(Path(path).read_text())
            return cls(model_dir=d["model_dir"],
                       threshold=float(d["threshold"]),
                       swapped_at=d.get("swapped_at", ""))
        except (OSError, ValueError, KeyError):
            return None

    @classmethod
    def write(cls, path: Path, model_dir: str,
              threshold: float) -> "ModelPointer":
        """Atomically publish a new pointer (write temp + os.replace)."""
        ptr = cls(model_dir=str(model_dir), threshold=float(threshold),
                  swapped_at=datetime.now(timezone.utc).isoformat())
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "model_dir": ptr.model_dir, "threshold": ptr.threshold,
            "swapped_at": ptr.swapped_at}, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return ptr
