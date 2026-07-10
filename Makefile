# PHAROS Phase 0 + Phase 1 targets.
# Run inside the WSL `pharos` conda env:
#   conda activate pharos && make train-physics
# Override the interpreter with:  make PYTHON=/path/to/python train-physics

PYTHON ?= python
export PYTHONPATH := $(CURDIR)

PHYSICS_CFG ?= configs/physics_vae.yaml
PDM_CFG ?= configs/pdm_ae.yaml

# Phase 1 demo knobs (override on the command line).
PHYS_RATE ?= 500
PHYS_LIMIT ?= 10000
PDM_RATE ?= 20
PDM_LIMIT ?= 150

.PHONY: help setup train-physics eval-physics train-pdm eval-pdm phase0 smoke clean \
        up down thresholds produce-physics produce-pdm score-physics score-pdm phase1 \
        export-onnx verify-onnx sofie-probe bench-inference score-physics-p2 decision phase2 \
        hls4ml-estimate

help:
	@echo "PHAROS Phase 0 targets:"
	@echo "  make setup          install pytest/pyyaml/joblib into the active env"
	@echo "  make train-physics  train Stream A VAE on ADC2021 background (GPU if avail)"
	@echo "  make eval-physics   ROC/AUC background vs A->4l signal -> reports/phase0/"
	@echo "  make train-pdm      train Stream B conv AE + IsolationForest on HVCM"
	@echo "  make eval-pdm       per-fault-class AUC table + plots -> reports/phase0/"
	@echo "  make phase0         train + eval both streams"
	@echo "  make smoke          run the tiny-subset smoke tests (CPU)"
	@echo "PHAROS Phase 1 targets (broker in Docker; producers/scorers on host):"
	@echo "  make up             start Redpanda + Console (broker profile) + topics"
	@echo "  make down           stop the broker containers"
	@echo "  make thresholds     derive scorer thresholds from Phase 0 backgrounds"
	@echo "  make produce-physics  replay ADC2021 background -> events.physics"
	@echo "  make produce-pdm      replay HVCM pulses -> events.pdm"
	@echo "  make score-physics    score events.physics -> anomalies.scouting"
	@echo "  make score-pdm        score events.pdm -> alerts.pdm"
	@echo "  make phase1           broker up + short end-to-end demo of both streams"
	@echo "PHAROS Phase 2 targets (trigger-realistic inference + decision layer):"
	@echo "  make export-onnx      export VAE encoder->mu to ONNX (batch 1)"
	@echo "  make verify-onnx      PyTorch vs onnxruntime parity -> reports/phase2/"
	@echo "  make sofie-probe      one-shot rootproject/root container SOFIE check"
	@echo "  make bench-inference  per-event latency: pytorch vs ort (vs sofie)"
	@echo "  make score-physics-p2 ORT scorer, forward-all -> events.physics.scored"
	@echo "  make decision         L1-budget decision layer -> anomalies.scouting"
	@echo "  make phase2           broker up + scorer + decision end-to-end demo"
	@echo "  make hls4ml-estimate  hls4ml conversion + C-emulation check (no Vivado)"

setup:
	$(PYTHON) -m pip install pytest pyyaml joblib

train-physics:
	$(PYTHON) -m src.training.train_physics --config $(PHYSICS_CFG)

eval-physics:
	$(PYTHON) -m scripts.eval_physics --config $(PHYSICS_CFG)

train-pdm:
	$(PYTHON) -m src.training.train_pdm --config $(PDM_CFG)

eval-pdm:
	$(PYTHON) -m scripts.eval_pdm --config $(PDM_CFG)

phase0: train-physics eval-physics train-pdm eval-pdm

smoke:
	$(PYTHON) -m pytest tests/ -q

# ---------------------------------------------------------------- Phase 1 ----

up:
	docker compose --profile broker up -d
	docker compose --profile broker ps

down:
	docker compose --profile broker down

thresholds:
	$(PYTHON) -m scripts.derive_thresholds

produce-physics:
	$(PYTHON) -m services.producers.physics_producer --rate $(PHYS_RATE) --limit $(PHYS_LIMIT)

produce-pdm:
	$(PYTHON) -m services.producers.pdm_producer --rate $(PDM_RATE) --limit $(PDM_LIMIT)

score-physics:
	$(PYTHON) -m services.scorers.physics_scorer

score-pdm:
	$(PYTHON) -m services.scorers.pdm_scorer

# Short end-to-end demo: broker up, scorers listening in the background,
# producers replay a bounded sample, scorers exit on idle and write
# reports/phase1/ metrics.
phase1: up
	PHYS_RATE=$(PHYS_RATE) PHYS_LIMIT=$(PHYS_LIMIT) PDM_RATE=$(PDM_RATE) PDM_LIMIT=$(PDM_LIMIT) \
		bash scripts/phase1_demo.sh

# ---------------------------------------------------------------- Phase 2 ----

export-onnx:
	$(PYTHON) -m scripts.export_onnx

verify-onnx:
	$(PYTHON) -m scripts.verify_onnx

# One-shot ROOT container; writes reports/phase2/sofie_probe.txt. A non-zero
# exit just means "SOFIE not available", which is a valid documented outcome.
sofie-probe:
	docker run --rm -v "$(CURDIR):/work" -w /work rootproject/root:latest \
		python3 scripts/sofie_probe.py || true

bench-inference:
	$(PYTHON) -m scripts.bench_inference

score-physics-p2:
	$(PYTHON) -m services.scorers.physics_scorer_sofie --backend ort --forward-all

decision:
	$(PYTHON) -m services.decision.physics_decision

# hls4ml conversion + C-emulation estimate (NO Vivado/Vitis; see
# docs/hls4ml_synthesis.md for the one-time full synthesis elsewhere).
hls4ml-estimate:
	$(PYTHON) -m scripts.hls4ml_estimate

phase2: up
	PHYS_RATE=$(PHYS_RATE) PHYS_LIMIT=$(PHYS_LIMIT) bash scripts/phase2_demo.sh

clean:
	rm -rf models/physics_vae models/pdm
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
