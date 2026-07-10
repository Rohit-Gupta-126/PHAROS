"""Probe whether this ROOT build has TMVA SOFIE with ONNX parsing enabled.

Meant to run as a one-shot inside the ``rootproject/root`` container:

    docker run --rm -v <repo>:/work -w /work rootproject/root:latest \
        python3 scripts/sofie_probe.py

Tries to parse ``models/physics_vae/encoder_mu.onnx`` with
``TMVA::Experimental::SOFIE::RModelParser_ONNX``. On success it also generates
the C++ inference code (header + .dat weights) into
``services/inference_sofie/generated/``. Result goes to
``reports/phase2/sofie_probe.txt``.
"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ONNX_PATH = Path("models/physics_vae/encoder_mu.onnx")
GEN_DIR = Path("services/inference_sofie/generated")
REPORT = Path("reports/phase2/sofie_probe.txt")

lines = [
    "PHAROS Phase 2 -- TMVA SOFIE availability probe",
    f"generated: {datetime.now(timezone.utc).isoformat()}",
]

status = "FAIL"
try:
    import ROOT
    lines.append(f"ROOT version: {ROOT.gROOT.GetVersion()}")
    parser = ROOT.TMVA.Experimental.SOFIE.RModelParser_ONNX()
    model = parser.Parse(str(ONNX_PATH))
    lines.append(f"parsed OK: {ONNX_PATH}")
    model.Generate()
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    # OutputGenerated writes <name>.hxx (+ .dat weights) next to the given path.
    model.OutputGenerated(str(GEN_DIR / "encoder_mu.hxx"))
    generated = sorted(p.name for p in GEN_DIR.iterdir())
    lines.append(f"generated C++ inference code: {generated}")
    status = "SOFIE AVAILABLE"
except Exception:
    lines.append("SOFIE parse/generate failed:")
    lines.append(traceback.format_exc())
    status = "SOFIE NOT AVAILABLE"

lines.insert(2, f"RESULT: {status}")
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if status == "SOFIE AVAILABLE" else 1)
