#!/usr/bin/env bash
# Phase 2 end-to-end demo: ORT deploy-path scorer (--forward-all) feeds the
# decision/rate-control layer while the producer replays background events.
# Consumer groups are fixed, so reruns resume from committed offsets.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

PHYS_RATE="${PHYS_RATE:-500}"
PHYS_LIMIT="${PHYS_LIMIT:-10000}"
BUDGET_FRACTION="${BUDGET_FRACTION:-0.01}"
WINDOW_S="${WINDOW_S:-1.0}"

python -m services.decision.physics_decision \
    --window "$WINDOW_S" --budget-fraction "$BUDGET_FRACTION" --idle 25 &
DEC_PID=$!
python -m services.scorers.physics_scorer_sofie --backend ort --forward-all --idle 20 &
SCORER_PID=$!
sleep 5

python -m services.producers.physics_producer --rate "$PHYS_RATE" --limit "$PHYS_LIMIT"

wait "$SCORER_PID" "$DEC_PID"
echo "[phase2-demo] done; reports in reports/phase2/"
