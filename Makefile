# PHAROS Phase 0 targets.
# Run inside the WSL `pharos` conda env:
#   conda activate pharos && make train-physics
# Override the interpreter with:  make PYTHON=/path/to/python train-physics

PYTHON ?= python
export PYTHONPATH := $(CURDIR)

PHYSICS_CFG ?= configs/physics_vae.yaml
PDM_CFG ?= configs/pdm_ae.yaml

.PHONY: help setup train-physics eval-physics train-pdm eval-pdm phase0 smoke clean

help:
	@echo "PHAROS Phase 0 targets:"
	@echo "  make setup          install pytest/pyyaml/joblib into the active env"
	@echo "  make train-physics  train Stream A VAE on ADC2021 background (GPU if avail)"
	@echo "  make eval-physics   ROC/AUC background vs A->4l signal -> reports/phase0/"
	@echo "  make train-pdm      train Stream B conv AE + IsolationForest on HVCM"
	@echo "  make eval-pdm       per-fault-class AUC table + plots -> reports/phase0/"
	@echo "  make phase0         train + eval both streams"
	@echo "  make smoke          run the tiny-subset smoke tests (CPU)"

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

clean:
	rm -rf models/physics_vae models/pdm
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
