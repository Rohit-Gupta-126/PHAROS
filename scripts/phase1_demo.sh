#!/usr/bin/env bash
# Phase 1 end-to-end demo: scorers listening concurrently while producers
# stream, so reports/phase1/ captures true in-flight latency.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

PHYS_RATE="${PHYS_RATE:-500}"
PHYS_LIMIT="${PHYS_LIMIT:-10000}"
PDM_RATE="${PDM_RATE:-20}"
PDM_LIMIT="${PDM_LIMIT:-150}"

python -m services.scorers.physics_scorer --idle 20 &
PHYS_PID=$!
python -m services.scorers.pdm_scorer --idle 20 &
PDM_PID=$!
sleep 5

python -m services.producers.physics_producer --rate "$PHYS_RATE" --limit "$PHYS_LIMIT" &
PHYS_PROD_PID=$!
python -m services.producers.pdm_producer --rate "$PDM_RATE" --limit "$PDM_LIMIT"
wait "$PHYS_PROD_PID"

wait "$PHYS_PID" "$PDM_PID"
echo "[phase1-demo] done; reports in reports/phase1/"
