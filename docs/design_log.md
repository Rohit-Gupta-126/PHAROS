# PHAROS design log

Decisions are recorded newest-last, tagged by phase.

## Phase 0 — offline ML cores

**Environment.** Execution env is the WSL Ubuntu-22.04 `pharos` conda env
(Python 3.11, `torch 2.11.0+cu128`, CUDA on RTX 2050 4 GB). Code is
device-agnostic: CUDA when available (with fp16 autocast), CPU otherwise, so the
same code runs in CI. Models are small enough that gradient checkpointing is
unnecessary on the 4 GB card.

**Stream A (physics VAE).**
- Input representation: ADC2021 `(N,19,4)` → drop class column → flatten to 57
  features `[pT,eta,phi] × 19 slots`. See `docs/data_schema_adc2021.md`.
- Normalization: `log1p(pT)` + z-score; eta/phi z-score; stats fit on train
  split only, constant slots guarded to 0.
- Model: MLP VAE `57→32→16→(μ8,logσ²8)→16→32→57`. Loss = MSE + β·KL with a
  short KL warm-up to avoid posterior collapse (the score depends on μ staying
  informative).
- Anomaly score = `Σ μ²` (AXOL1TL / CICADA convention).
- Training data: 2M background events subsampled, 90/10 train/val, batch 1024.
- Eval: ROC/AUC of background vs `A→4ℓ` signal → `reports/phase0/`.

**Stream B (HVCM PDM).**
- Data: 4 systems (RFQ/DTL/CCL/SCL), waveforms `(pulses,4500,14)`. Downsample
  4500→500 by average pooling, done in row chunks so the largest array (SCL,
  ~3.6 GB) is never fully materialized.
- Primary detector: 1D conv autoencoder trained on **normal** pulses only;
  score = reconstruction MSE. Baseline: IsolationForest on per-channel summary
  features (mean/std/min/max/ptp/energy).
- Eval: per-fault-class detection AUC (normal vs each fault class) for both
  detectors, plus an "ALL faults" AUC → `reports/phase0/pdm_auc.csv` + plots.

**Tooling.** Configs are YAML (`configs/`), each pipeline exposes `run(cfg)` so
smoke tests drive tiny CPU configs directly. Makefile targets: `train-physics`,
`eval-physics`, `train-pdm`, `eval-pdm`, plus `setup`/`smoke`/`phase0`.
Optional ONNX export is wired into `train_physics` for the later ONNX→SOFIE
deploy path.

**Memory constraint (WSL).** The WSL Ubuntu guest has only ~8 GB RAM. An initial
scattered fancy-index read of 2M background rows peaked at ~7 GB and got
OOM-killed. Fixed by reading HDF5 in contiguous row chunks, subsampling within
each chunk with global-stride alignment, and downcasting to float32 immediately
(`adc2021.load_events`); peak dropped to ~3 GB for the 2M-event load (32 s).
Installed `make` and `pytest` into the `pharos` env (`make setup` handles pytest).

**Phase 0 verification results (RTX 2050).**
- Stream A: trained on 2M background events (1.8M/0.2M split), 20 epochs on GPU,
  ~13 s/epoch; train loss 1.20 → 0.90. **ROC AUC = 0.766** for A→4ℓ
  (background from the held-out tail 10%). `mean(Σμ²)`: bg 0.021 vs signal 0.083.
  β=1.0 keeps the latent modestly informative (`mean_mu2`≈0.024); lowering β is
  the lever if higher separation is wanted later.
- Stream B: conv AE + IsolationForest trained per system (RFQ/DTL/CCL/SCL) in
  ~52 s total. Per-fault AUCs across **68 fault classes**. Aggregate: IsolationForest
  (baseline) median AUC 0.835 > conv AE 0.709 — simple per-channel summary stats
  capture many HVCM faults well; the AE wins on 21/68 classes (e.g. IGBT/driver
  faults). Overall (all-faults) AUC per system: RFQ 0.80/0.82, DTL 0.76/0.77,
  CCL 0.58/0.75, SCL 0.68/0.76 (AE/Iso). Both are reported so later phases can
  pick per fault type.
