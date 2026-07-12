"""RDataFrame ingestion of a CMS Open Data NanoAOD file -> ADC2021 57-vectors.

This is the *real-data* ROOT entry point. It runs inside the
``rootproject/root:latest`` container (PyROOT + RDataFrame), reads the physics
objects PHAROS uses from the ``Events`` tree, maps each event onto the ADC2021
57-feature layout (``src/preprocessing/nanoaod.build_event_vector``), and writes
a compact ``(N, 57)`` float32 array to ``data/interim/``. A host-side streamer
(``services/ingest_root/stream_cms.py``) then replays that array through the SAME
producer interface as Phase 1 -- so the Kafka client stays on the host (matching
"producers are host processes") and the ROOT image is not fattened with
confluent-kafka.

Memory: only the needed branches are read and ``--limit`` caps the number of
events materialized (via ``RDataFrame.Range``) so the 12 GB WSL guest never OOMs.
A capped run stays single-threaded because ``Range()`` and implicit MT are
mutually exclusive in RDataFrame; a full-file ingest (``--limit 0``) enables
implicit MT. The source is a **local file** written by ``fetch_nanoaod.sh``:
``rootproject/root:latest`` no longer bundles an XRootD client, so ``root://``
streaming is unavailable and we ingest the downloaded local copy.

Run (in-container):
    python3 services/ingest_root/ingest_nanoaod.py \
        --source data/raw/cms_opendata/<file>.root --limit 50000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the repo importable when invoked as a bare script inside the container.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocessing.nanoaod import build_event_vector, N_FEATURES  # noqa: E402


def _import_root():
    """Import PyROOT lazily -- it only exists inside rootproject/root, but the
    rest of this module (e.g. _met_branches) must import on the host for tests."""
    try:
        import ROOT  # type: ignore
        return ROOT
    except ImportError as exc:  # pragma: no cover - host has no ROOT
        print("[ingest-cms] PyROOT not found -- run inside rootproject/root:\n"
              "  docker run --rm -v \"$PWD:/work\" -w /work rootproject/root:latest \\\n"
              "    python3 services/ingest_root/ingest_nanoaod.py --source <file>.root",
              file=sys.stderr)
        raise SystemExit(2) from exc

# NanoAOD branches PHAROS needs. MET uses PuppiMET with a plain-MET fallback.
_OBJ_BRANCHES = [
    "Electron_pt", "Electron_eta", "Electron_phi",
    "Muon_pt", "Muon_eta", "Muon_phi",
    "Jet_pt", "Jet_eta", "Jet_phi",
]


def _met_branches(columns: set[str]) -> tuple[str, str]:
    for prefix in ("PuppiMET", "MET"):
        if f"{prefix}_pt" in columns and f"{prefix}_phi" in columns:
            return f"{prefix}_pt", f"{prefix}_phi"
    raise RuntimeError("No PuppiMET_/MET_ branches found in the Events tree")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True,
                   help="local .root file path (Events tree)")
    p.add_argument("--tree", default="Events")
    p.add_argument("--out", default="data/interim/cms_events_57.npy")
    p.add_argument("--limit", type=int, default=50_000,
                   help="max events to materialize (0 = all; caps memory)")
    p.add_argument("--threads", type=int, default=0,
                   help="implicit-MT threads (0 = ROOT default = all cores)")
    args = p.parse_args(argv)

    ROOT = _import_root()
    # RDataFrame.Range() and implicit MT are mutually exclusive: Range() throws
    # under ImplicitMT. A capped run (--limit > 0) uses Range(), so it stays
    # single-threaded on purpose; only a full-file ingest (--limit == 0) enables
    # implicit MT.
    if args.limit == 0 and args.threads != 1:
        ROOT.EnableImplicitMT(args.threads if args.threads > 0 else 0)

    df = ROOT.RDataFrame(args.tree, args.source)
    columns = {str(c) for c in df.GetColumnNames()}
    met_pt_br, met_phi_br = _met_branches(columns)
    branches = [met_pt_br, met_phi_br] + _OBJ_BRANCHES

    # Range() caps the event loop lazily -- only --limit events are read.
    node = df.Range(args.limit) if args.limit else df
    # AsNumpy pulls the (variable-length) collections as arrays of std vectors.
    data = node.AsNumpy(columns=branches)
    n = len(data[met_pt_br])
    print(f"[ingest-cms] {n} events from {args.source} "
          f"(MET={met_pt_br}) -> building 57-vectors", file=sys.stderr)

    out = np.zeros((n, N_FEATURES), dtype=np.float32)
    for i in range(n):
        out[i] = build_event_vector(
            met_pt=float(data[met_pt_br][i]), met_phi=float(data[met_phi_br][i]),
            ele_pt=data["Electron_pt"][i], ele_eta=data["Electron_eta"][i],
            ele_phi=data["Electron_phi"][i],
            mu_pt=data["Muon_pt"][i], mu_eta=data["Muon_eta"][i],
            mu_phi=data["Muon_phi"][i],
            jet_pt=data["Jet_pt"][i], jet_eta=data["Jet_eta"][i],
            jet_phi=data["Jet_phi"][i])
        if (i + 1) % 20_000 == 0:
            print(f"[ingest-cms] built {i + 1}/{n}", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)
    print(f"[ingest-cms] wrote {out.shape} float32 -> {out_path}")


if __name__ == "__main__":
    main()
