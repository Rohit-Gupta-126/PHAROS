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
        hls4ml-estimate \
        reference-stats monitor score-pdm-p3 inject-physics inject-pdm lead-time pdm-skew \
        retrain-trigger dashboard phase3 \
        fetch-cms ingest-cms stream-cms sim-vs-real analysis-prep analysis-rdf \
        dashboard-api phase4

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
	@echo "PHAROS Phase 3 targets (MLOps: drift monitor + closed retrain loop):"
	@echo "  make reference-stats  frozen drift references from Phase 0 artifacts"
	@echo "  make monitor          drift monitor: PSI/KS vs references -> alerts.drift"
	@echo "  make score-pdm-p3     PDM scorer with --forward-all -> events.pdm.scored"
	@echo "  make inject-physics   background then black-box mid-stream shift"
	@echo "  make inject-pdm       normal-val then file-head calibration skew"
	@echo "  make lead-time        detection lead time + drift timeline plot"
	@echo "  make pdm-skew         benign-skew vs real-shift signature verdict"
	@echo "  make retrain-trigger  confirmed drift -> retrain -> parity -> hot-swap"
	@echo "  make dashboard        Streamlit live dashboard (host, reads topics)"
	@echo "  make phase3           broker up + full drift/inject/retrain demo"
	@echo "PHAROS Phase 4 targets (real CMS ingestion + RDataFrame + web dashboard):"
	@echo "  make fetch-cms        xrdcp a CMS Open Data NanoAOD file (one-shot ROOT)"
	@echo "  make ingest-cms       RDataFrame NanoAOD -> data/interim/cms_events_57.npy"
	@echo "  make stream-cms       replay CMS 57-vectors -> events.physics (host)"
	@echo "  make sim-vs-real      stream CMS through scorer+monitor -> domain-gap report"
	@echo "  make analysis-prep    host: AUC table + observables npz (active pointer)"
	@echo "  make analysis-rdf     RDataFrame physics histograms -> reports/phase4/"
	@echo "  make dashboard-api    read-only SSE bridge + static dashboard (no Streamlit)"
	@echo "  make phase4           real-data ingest + domain-gap + analysis (needs Docker)"

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

# ---------------------------------------------------------------- Phase 3 ----

reference-stats:
	$(PYTHON) -m scripts.derive_reference_stats

monitor:
	$(PYTHON) -m services.monitor.drift_monitor

score-pdm-p3:
	$(PYTHON) -m services.scorers.pdm_scorer --forward-all --reports-dir reports/phase3

inject-physics:
	$(PYTHON) -m tools.inject.inject_physics --rate $(PHYS_RATE)

inject-pdm:
	$(PYTHON) -m tools.inject.inject_pdm --system RFQ --rate $(PDM_RATE)

lead-time:
	$(PYTHON) -m tools.inject.measure_lead_time

pdm-skew:
	$(PYTHON) -m tools.inject.analyze_pdm_skew

retrain-trigger:
	$(PYTHON) -m services.monitor.retrain_trigger --max-retrains 1

dashboard:
	$(PYTHON) -m streamlit run services/dashboard/app.py

phase3: up
	PHYS_RATE=$(PHYS_RATE) PDM_RATE=$(PDM_RATE) bash scripts/phase3_demo.sh

# ---------------------------------------------------------------- Phase 4 ----
# ROOT jobs run as one-shot rootproject/root containers (mirror sofie-probe);
# the Kafka client stays on the host. Override the source with CMS_NANOAOD_URL.

ROOT_IMAGE ?= rootproject/root:latest
ROOT_RUN = docker run --rm -v "$(CURDIR):/work" -w /work $(ROOT_IMAGE)
CMS_LIMIT ?= 50000
DASH_PORT ?= 8070

fetch-cms:
	$(ROOT_RUN) bash services/ingest_root/fetch_nanoaod.sh

# Ingest the first local NanoAOD file found in data/raw/cms_opendata/.
ingest-cms:
	$(ROOT_RUN) sh -c 'python3 services/ingest_root/ingest_nanoaod.py \
		--source "$$(ls data/raw/cms_opendata/*.root | head -n1)" --limit $(CMS_LIMIT)'

stream-cms:
	$(PYTHON) -m services.ingest_root.stream_cms --rate $(PHYS_RATE) --limit $(PHYS_LIMIT)

sim-vs-real:
	$(PYTHON) -m scripts.phase4_sim_vs_real --rate $(PHYS_RATE) --limit $(PHYS_LIMIT)

analysis-prep:
	$(PYTHON) -m analysis.prep_adc_npy

analysis-rdf:
	$(ROOT_RUN) python3 analysis/physics_rdf.py --npz data/interim/adc_obs.npz

dashboard-api:
	$(PYTHON) -m services.dashboard_api.app --port $(DASH_PORT)

# Real-data end-to-end: extract with RDataFrame, then measure the domain gap.
# (fetch-cms first if data/raw/cms_opendata/ is empty.)
phase4: up ingest-cms sim-vs-real
	@echo "[phase4] domain-gap report -> reports/phase4/sim_vs_real_drift.json"

clean:
	rm -rf models/physics_vae models/pdm
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
