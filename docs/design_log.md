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

## Phase 0.5 — reporting fixes (no retraining)

Eval/plotting only; both existing checkpoints (`models/physics_vae`,
`models/pdm/*`) reused unchanged.

**Stream A — dual score reporting.** `eval_physics` now scores background vs
`A→4ℓ` with two scores from the *same* VAE checkpoint:
- `auc_latent_summu2` = **0.766** — the FPGA-cheap AXOL1TL-style trigger score
  (Σμ², encoder only).
- `auc_recon_mse` = **0.889** — full-VAE reconstruction MSE (decode from μ), the
  offline accuracy reference.
The **0.12 gap** quantifies what the cheap trigger score gives up versus running
the decoder: A→4ℓ events reconstruct far worse (recon mean bg 0.84 vs signal
264) than their latent means suggest, so recon-MSE is the better *offline*
discriminator while Σμ² remains the deployable trigger. Both recorded in
`physics_metrics.json` (with a `score_note`); `auc` kept as an alias of the
trigger score for backward compat.
`physics_score_hist.png` rebuilt as a two-panel **log10 x-axis** figure (bins on
`log10(score+eps)`, clipped to the 0.5–99.5 percentile range) so the
right-skewed distributions actually show the background peak vs the signal tail.

**Stream B — label hygiene + honest aggregates.** Fault-class labels are now
canonicalized (collapse internal whitespace + title-case) before aggregation, so
case/spacing variants such as `C FLUX Low Fault` / `C Flux Low Fault` merge into
one class. This drops the per-system fault-class **row** count in `pdm_auc.csv`
from 68 to **67** (one within-system `FLUX`/`Flux` duplicate merged); those 67
rows span **42 distinct** fault classes (the same class in two systems is two
rows). Note `B Flux Low` and `C Flux Low` are genuinely different classes and
stay separate. `pdm_auc.csv` gains an `n_below_floor` column flagging classes
with n<5 (17 such rows); every class is still listed, but the headline
**median AUC** stat is computed only over the **50** rows with n≥5: median **AE 0.711**, **IsolationForest 0.805** — the
IsoForest baseline still leads. Per-system ALL-faults AUCs are unchanged
(RFQ 0.80/0.82, DTL 0.76/0.77, CCL 0.58/0.75, SCL 0.68/0.76 AE/Iso).
`pdm_metrics.json` restructured to `{headline, per_system}`.

## Phase 1 — streaming backbone

**Broker.** Single-node Redpanda v24.2 (Kafka API on `localhost:9092`) +
Redpanda Console (`:8080`) under a `broker` compose profile — the only
containers. Capped for the 12 GB WSL guest: `--smp 1 --memory 1500M
--reserve-memory 0M --overprovisioned`. A one-shot `rpk` container creates the
four single-partition topics (`events.physics`, `events.pdm`,
`anomalies.scouting`, `alerts.pdm`).

**Process placement.** Producers and scorers run as host Python processes in
the WSL `pharos` env, NOT in containers — saves RAM and gives the scorers
direct CUDA access. Plain `confluent-kafka` clients; no Flink/Spark/Faust.

**Wire format** (`docs/wire_format.md`). Versioned JSON, keyed by `event_id`,
with a producer nanosecond timestamp. Producers emit *pre-normalization*,
model-ready-shape data (physics: raw 57-dim vectors; PDM: avg-pooled
`(500,14)` waves via the Phase 0 `src.preprocessing` code) so the scorer owns
normalization — one source of truth, streaming scores match Phase 0 exactly
(asserted by `tests/test_phase1.py` JSON round-trip tests). PDM records carry
`ground_truth` for offline keep-rate accounting only.

**Thresholds are derived** (`scripts/derive_thresholds.py` → `make
thresholds`): the p99 (configurable) of the Phase 0 background/normal score
distribution, scored with the frozen artifacts — physics on the held-out file
tail (0.9–1.0), PDM on the normal-validation pulses per system (Phase 0 seed,
so the AE never saw them). Stored in `configs/thresholds.json`; expected
background keep-rate is `(100−p)/100` by construction. Derived: physics
p99(Σμ²)=1.12e-3; PDM p99(recon MSE) RFQ 0.164 / DTL 0.152 / CCL 0.227 /
SCL 0.161.

**Scorers** load Phase 0 checkpoints + normalizers frozen (no refit). The
physics scorer micro-batches (default 256) to amortize GPU calls; PDM scores
per pulse. Keepers republished as compact `{event_id, score, threshold, ts}`
records. On idle-exit each scorer writes latency (producer→scored),
throughput, and keep-rate metrics + plots to `reports/phase1/`.

**Phase 0 consistency check.** Replaying 10k held-out background events
through the wire gave a scorer-side median Σμ² of 3.42e-05 vs the Phase 0
eval median 3.43e-05, and a background keep-rate of **0.99%** against the 1%
configured — the streaming path reproduces the offline distribution.
`tests/test_phase1.py` additionally asserts producer-JSON→scorer scores match
direct scoring to rtol 1e-5.

**End-to-end results** (concurrent scorers + producers, `make phase1` /
`scripts/phase1_demo.sh`, reports in `reports/phase1/`):
- Physics: 10k events at 500 ev/s; keep-rate 0.99% (99/10000);
  producer→scored latency p50 6.1 ms / p95 9.6 ms / max 195 ms (first-batch
  warm-up); throughput matches the configured rate. Unthrottled scoring
  capacity measured earlier at ~40k events/s (GPU micro-batch 256).
- PDM: 600 pulses (150 × 4 systems) at 20 /s; keep-rate 1.67% (10/600) —
  slightly above the 1% target because the demo replays the *head* of each
  file rather than the normal-validation split the threshold was derived on;
  latency p50 20 ms / p95 26 ms.
- Container budget: Redpanda ~300 MiB + Console ~23 MiB — the only
  containers, well inside the 12 GB WSL guest.
