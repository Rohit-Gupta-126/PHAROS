#!/usr/bin/env bash
# Phase 3 end-to-end demo: drift monitor + injectors + closed-loop retrain.
#
# Prereqs: broker up (make up), Phase 0 models trained, thresholds derived,
# ONNX exported (make export-onnx), reference stats derived
# (make reference-stats). Timing consumers start at end-of-topic (offset
# hygiene), so stale backlog never skews the run.
#
# Stages:
#   1. scorers (physics ORT --forward-all --model-pointer, pdm --forward-all),
#      drift monitor, retrain trigger -- all listening;
#   2. physics injector: background, then ADC2021 black-box mid-stream;
#   3. pdm injector: normal-val slice, then file-head calibration skew;
#   4. post-hoc: detection lead time + PDM skew-signature verdict;
#   5. wait for the (possible) retrain + parity-gated hot-swap, then a short
#      verification replay to show which model dir is serving.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

PHYS_RATE="${PHYS_RATE:-500}"
BASELINE="${BASELINE:-4000}"
INJECT="${INJECT:-6000}"
PDM_RATE="${PDM_RATE:-20}"
POINTER="models/physics_vae/current.json"

rm -f "$POINTER"   # start the demo blue/green-clean: Phase 0 model serves

# Idle timeouts are generous (90 s): inject_physics spends up to a minute
# loading the 4.2M-row black-box HDF5 before emitting, and nothing must exit
# during that silence.
python -m services.scorers.physics_scorer_sofie --backend ort --forward-all \
    --model-pointer "$POINTER" --reports-dir reports/phase3 --idle 90 &
PHYS_PID=$!
python -m services.scorers.pdm_scorer --forward-all \
    --reports-dir reports/phase3 --idle 90 &
PDM_PID=$!
python -m services.monitor.drift_monitor --idle 95 &
MON_PID=$!
# --min-severity warn: the black-box mixture is mostly background-like, so
# score PSI plateaus in the warn band (~0.15-0.20) rather than crossing 0.25;
# sustained-window confirmation (3 consecutive) still applies.
python -m services.monitor.retrain_trigger --max-retrains 1 --idle 120 \
    --min-severity warn &
RETRAIN_PID=$!
sleep 5

echo "[phase3-demo] injecting pdm calibration skew..."
python -m tools.inject.inject_pdm --system RFQ --rate "$PDM_RATE"
echo "[phase3-demo] injecting physics black-box shift..."
python -m tools.inject.inject_physics --rate "$PHYS_RATE" \
    --baseline "$BASELINE" --inject "$INJECT"

wait "$PHYS_PID" "$PDM_PID" "$MON_PID"
echo "[phase3-demo] streams idle; measuring outcomes..."
python -m tools.inject.measure_lead_time
python -m tools.inject.analyze_pdm_skew

echo "[phase3-demo] waiting for retrain trigger (trains on confirmed drift)..."
wait "$RETRAIN_PID"

if [ -f "$POINTER" ]; then
    echo "[phase3-demo] pointer written; verification replay on the swapped model:"
    python -m services.scorers.physics_scorer_sofie --backend ort --forward-all \
        --model-pointer "$POINTER" --reports-dir reports/phase3 --idle 15 &
    VERIFY_PID=$!
    sleep 3
    python -m services.producers.physics_producer --rate "$PHYS_RATE" --limit 2000
    wait "$VERIFY_PID"
else
    echo "[phase3-demo] no hot-swap happened (no confirmed drift or parity fail)"
fi
echo "[phase3-demo] done; reports in reports/phase3/"
