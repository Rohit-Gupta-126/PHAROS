"""RDataFrame offline analysis of the Stream A physics observables (Phase 4).

The ROOT/RDataFrame *analysis* entry point. Runs inside
``rootproject/root:latest`` on the columnar ``.npz`` produced by
``analysis/prep_adc_npy.py``: builds an RDataFrame via ``RDF.FromNumpy``, enables
implicit multithreading, and books overlaid background-vs-signal histograms for
each physics observable and the two anomaly scores in a single multithreaded
event loop. Saves PNGs to ``reports/phase4/``.

AUC is computed on the host (needs torch/sklearn) in ``prep_adc_npy.py``; this
job is the ROOT-side histogramming that exploits RDataFrame's MT.

Run (in-container):
    python3 analysis/physics_rdf.py --npz data/interim/adc_obs.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import ROOT  # type: ignore
except ImportError as exc:  # pragma: no cover - host has no ROOT
    print("[physics-rdf] PyROOT not found -- run inside rootproject/root:\n"
          "  docker run --rm -v \"$PWD:/work\" -w /work rootproject/root:latest \\\n"
          "    python3 analysis/physics_rdf.py --npz data/interim/adc_obs.npz",
          file=sys.stderr)
    raise SystemExit(2) from exc

# observable column -> (axis label, log-y?, robust upper-quantile clip)
_OBS = {
    "met_pt":      ("MET pT [GeV]", False, 0.99),
    "lead_ele_pt": ("leading electron pT [GeV]", False, 0.99),
    "lead_mu_pt":  ("leading muon pT [GeV]", False, 0.99),
    "n_jet":       ("jet multiplicity", False, 1.0),
    "ht":          ("H_T (scalar jet pT sum) [GeV]", False, 0.99),
    "summu2":      ("anomaly score Sum mu^2 (log10)", True, 0.999),
    "recon_mse":   ("reconstruction MSE (log10)", True, 0.999),
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--npz", default="data/interim/adc_obs.npz")
    p.add_argument("--reports-dir", default="reports/phase4")
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args(argv)

    ROOT.EnableImplicitMT(args.threads if args.threads > 0 else 0)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gROOT.SetBatch(True)

    d = np.load(args.npz)
    label = d["label"].astype(np.float64)
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    # One RDataFrame from all columns; filter to bg/sig lazily so the booked
    # histograms all fill in a single multithreaded event loop.
    cols = {k: np.ascontiguousarray(d[k].astype(np.float64)) for k in d.files}
    # log10 transform for the heavy-tailed scores (drop non-positive first).
    for sk in ("summu2", "recon_mse"):
        v = cols[sk]
        cols[sk] = np.where(v > 0, np.log10(v + 1e-12), -12.0)
    rdf = ROOT.RDF.FromNumpy(cols)
    bg = rdf.Filter("label < 0.5", "background")
    sig = rdf.Filter("label > 0.5", "signal")

    booked = []
    for col, (_, _, clip) in _OBS.items():
        v = cols[col]
        lo = float(np.quantile(v, 0.001))
        hi = float(np.quantile(v, clip))
        if hi <= lo:
            hi = lo + 1.0
        nb = 40
        booked.append((
            col,
            bg.Histo1D((f"bg_{col}", col, nb, lo, hi), col),
            sig.Histo1D((f"sig_{col}", col, nb, lo, hi), col)))

    for col, hbg, hsig in booked:
        label_txt, logy, _ = _OBS[col]
        _draw(col, hbg.GetValue(), hsig.GetValue(), label_txt, logy,
              reports / f"rdf_{col}.png")
    print(f"[physics-rdf] wrote {len(booked)} plots -> {reports}")


def _draw(col, hbg, hsig, xlabel, logy, path: Path) -> None:
    c = ROOT.TCanvas(f"c_{col}", col, 720, 520)
    if logy:
        c.SetLogy()
    for h, color in ((hbg, ROOT.kAzure + 1), (hsig, ROOT.kOrange + 7)):
        integral = h.Integral()
        if integral > 0:
            h.Scale(1.0 / integral)   # density: fair bg-vs-signal shape overlay
        h.SetLineColor(color)
        h.SetLineWidth(2)
        h.SetFillColorAlpha(color, 0.25)
    hbg.GetXaxis().SetTitle(xlabel)
    hbg.GetYaxis().SetTitle("normalized")
    ymax = 1.35 * max(hbg.GetMaximum(), hsig.GetMaximum())
    hbg.SetMaximum(ymax if ymax > 0 else 1.0)
    hbg.SetTitle(f"PHAROS Stream A - {xlabel}")
    hbg.Draw("HIST")
    hsig.Draw("HIST SAME")
    leg = ROOT.TLegend(0.62, 0.75, 0.88, 0.88)
    leg.AddEntry(hbg, "background", "l")
    leg.AddEntry(hsig, "A#rightarrow4l", "l")
    leg.SetBorderSize(0)
    leg.Draw()
    c.SaveAs(str(path))


if __name__ == "__main__":
    main()
