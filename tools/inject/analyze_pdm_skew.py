"""Classify the PDM drift signature: benign calibration skew vs real shift.

The key intellectual claim of Phase 3: a *benign calibration/sampling skew*
(same machine state, different slice of the file -- the Phase 1 mismatch)
should move the anomaly-SCORE distribution while the raw per-channel input
means stay put, whereas a *real* distribution shift moves both. This script
tests that claim against what actually landed in ``alerts.drift`` after the
``inject_pdm`` marker, and reports the verdict honestly either way:

* score PSI fired, no channel-mean PSI fired  -> ``calibration_suspect``
* score PSI and channel-mean PSI both fired   -> ``real_shift_signature``
* nothing fired                               -> ``no_drift_detected``

Output: ``reports/phase3/pdm_skew_analysis.json``. If the signature does NOT
separate the cases, that is a documented limitation, not something to tune
away (see docs/design_log.md, Phase 3).

Run: ``python -m tools.inject.analyze_pdm_skew``
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from src.common.config import resolve_path
from services import common
from tools.inject.measure_lead_time import drain_topic


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--marker", type=str,
                   default="reports/phase3/injection_marker_pdm.json")
    p.add_argument("--reports-dir", type=str, default="reports/phase3")
    p.add_argument("--bootstrap", type=str, default=common.BOOTSTRAP_DEFAULT)
    args = p.parse_args(argv)

    marker = json.loads(resolve_path(args.marker).read_text())
    start = marker["start_ts_ns"]
    stream = marker["stream"]  # e.g. "pdm/RFQ"

    drift = drain_topic(common.TOPIC_DRIFT, args.bootstrap)
    post = [e for e in drift if e["stream"] == stream
            and e["detected_ts_ns"] >= start]
    fired = lambda pred: sorted({e["metric"] for e in post  # noqa: E731
                                 if e["severity"] != "ok" and pred(e["metric"])})
    score_fired = fired(lambda m: m == "score_psi")
    feature_fired = fired(lambda m: m.startswith("ch"))

    if score_fired and not feature_fired:
        verdict = "calibration_suspect"
        note = ("Score distribution moved but raw channel means did not: "
                "consistent with benign sampling/calibration skew, not a "
                "physical change in the machine.")
    elif score_fired and feature_fired:
        verdict = "real_shift_signature"
        note = ("Both score AND raw-input drift fired: the monitor cannot "
                "distinguish this replay skew from a real shift on the "
                "metrics tracked -- documented limitation.")
    elif feature_fired:
        verdict = "input_only_drift"
        note = ("Raw inputs moved without a score response -- shift in a "
                "direction the AE reconstructs well.")
    else:
        verdict = "no_drift_detected"
        note = ("Neither score nor input metrics fired on this window "
                "sizing; the skew is below detection resolution.")

    result = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "stream": stream,
        "injection_start_ts_ns": start,
        "inject_source": marker.get("inject_source"),
        "n_post_injection_evaluations": len(post),
        "score_metrics_fired": score_fired,
        "feature_metrics_fired": feature_fired,
        "post_injection_max": {
            m: max(e["value"] for e in post if e["metric"] == m)
            for m in sorted({e["metric"] for e in post})},
        "verdict": verdict,
        "note": note,
    }
    out = resolve_path(args.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pdm_skew_analysis.json").write_text(json.dumps(result, indent=2),
                                                encoding="utf-8")
    print(f"[pdm-skew] {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
