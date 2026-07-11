# PHAROS design log

Decisions are recorded newest-last, tagged by phase. Read as a narrative, the
arc is: **build the two detectors offline (Phase 0)** → **put them on a streaming
backbone with one honest wire format (Phase 1)** → **make inference
trigger-realistic and add an L1 budget (Phase 2)** → **watch for drift and close
a parity-gated retrain loop (Phase 3)** → **feed the pipeline real detector data
and give it a face (Phase 4)**. A recurring theme runs through all of it: when a
result is inconvenient — the cheap trigger score is weaker than reconstruction,
fixed-point precision flips trigger decisions, the drift "benign-skew" signature
fails to separate, real data doesn't match the sim — the log records the finding
rather than tuning a clean-looking demo.

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

## Phase 2 — trigger-realistic inference + decision layer

**Encoder-only ONNX export.** The trigger score Σμ² needs only the encoder
trunk + μ head, so `scripts/export_onnx.py` exports a 57→μ[8] wrapper (fixed
batch 1, opset 13, legacy exporter → plain Gemm/Relu graph) from the frozen
Phase 0 checkpoint. Parity vs PyTorch on 2k held-out background events:
max |Δμ| = 8.9e-8, max |ΔΣμ²| = 7.5e-9 (`reports/phase2/onnx_parity.txt`).

**SOFIE probe → deferred, ORT is the runnable path.** `rootproject/root:latest`
(ROOT 6.38.00) ships the SOFIE runtime (`libROOTTMVASofie.so`) but NOT the
ONNX parser — `RModelParser_ONNX` is absent and `root-config --features`
lacks `tmva-sofie` (`reports/phase2/sofie_probe.txt`). Building ROOT with
`-Dtmva-sofie=ON` is a multi-hour >8 GB build — too heavy for this laptop by
the project's own stop rule. Decision: `docker/root-sofie.Dockerfile` records
the build recipe for a bigger machine; `services/inference_sofie/` holds a
ready-to-compile BLAS-only C++ `main.cpp` (line protocol: 57 floats in, Σμ²
out) against SOFIE's generated-header interface; **ONNX Runtime is the
documented runnable deploy path**.

**Deploy-path scorer + latency.** `services/scorers/trigger_backends.py`
gives interchangeable batch-1 backends (pytorch / ort / sofie-subprocess);
`services/scorers/physics_scorer_sofie.py` streams `events.physics` through
them. Offline benchmark (`make bench-inference`, 5k events, batch 1, CPU,
1 intra-op thread): **ORT 7.4 µs/event mean (p99 15.6 µs) vs PyTorch 32 µs
mean (p99 91 µs)** — ~4.3× faster, parity 1.9e-6 ≤ 1e-5
(`reports/phase2/inference_latency.{json,png}`).

**Decision / rate-control layer.** `services/decision/physics_decision.py`
consumes fully-scored events from a new topic `events.physics.scored`
(scorer run with `--forward-all`), applies the Phase 1 p99 threshold AND a
hard L1-style accept budget: per 1 s window, keep the top-N passers where
N = ceil(1% × window input). Kept events go to `anomalies.scouting` with
`decision_reason` = `threshold_pass` (budget not binding) or `rate_limited`
(window had more passers than budget; the rest dropped). End-to-end
(`make phase2`, 10k events at 500 ev/s): scorer keep-rate 0.99%, e2e latency
p50 5.5 ms; decision kept 87/10000 (12 rate-dropped in bursty windows) →
**reduction factor 115×** at the 1% budget
(`reports/phase2/decision_stats.json`).

**hls4ml (report-only).** hls4ml 1.3.0 installed cleanly (PyTorch frontend,
Vitis backend, io_parallel); C-emulation compiled with conda `cxx-compiler`
(no Vivado anywhere). Key finding: the tutorial default `ap_fixed<16,6>` is
unusable for this trigger — μ sits near a ~1e-3 threshold and 10 fractional
bits gave only 91% p99 decision agreement; **`ap_fixed<24,8>` restores 100%
agreement** (max |Δμ| ≈ 6e-4). Static estimate: 2 520 params / 2 464 MACs →
~2.5k DSPs at ReuseFactor 1 (RF 4–8 or 8-bit weights is the realistic
deployment point), pipeline latency O(100 ns) at 200 MHz. QKeras QAT skipped
(unmaintained on Keras-3-era stacks); post-training fixed-point suffices at
`<24,8>`. Full Vitis HLS synthesis is a documented one-time recipe for a lab
machine: `docs/hls4ml_synthesis.md`.

## Phase 3 — MLOps / drift monitoring

**Offset hygiene.** All timing/monitoring consumers (both scorers, the
decision layer, and the new Phase 3 monitor) now default to a *fresh per-run
consumer group* with `auto.offset.reset=latest`, i.e. they start at
end-of-topic. Phases 1–2 saw the first run after a break drain stale backlog,
which skewed latency and throughput stats. A fixed group can still be forced
with `--group` (resumes from committed offsets, `earliest` on first use) for
replay-style runs.

**Drift statistics.** Hand-rolled PSI + scipy `ks_2samp` instead of
alibi-detect/river: two well-understood statistics cover the need, and both
libraries drag heavy/fragile dependency trees onto a Windows/WSL laptop. The
reference is FROZEN at Phase 0: `scripts/derive_reference_stats.py` scores
the same held-out samples the thresholds came from (seed 1337) and stores
quantile-binned histograms (PSI, open-ended outer bins) plus a 2 000-sample
subsample (KS + dashboard overlays). PSI 0.1/0.25 warn/alert bounds are the
standard model-monitoring heuristics, not calibrated tests — stated as such.

**Monitor honesty.** The monitor (`services/monitor/drift_monitor.py`) is
marker-blind: injectors record the switch point to `ctrl.inject` + a marker
file, and only the post-hoc `measure_lead_time.py` joins the two, so detection
lead time is a real measurement, not a self-fulfilling one.

**Benign skew vs real drift (the key claim).** `inject_pdm` reproduces the
Phase 1 calibration mismatch on purpose (normal-val slice → file-head slice);
`analyze_pdm_skew.py` classifies the signature: score-PSI-only ⇒
`calibration_suspect` (model-view moved, raw channel means did not), score+
feature PSI ⇒ indistinguishable from a real shift on the tracked metrics —
whichever verdict the run produces is recorded as-is in
`reports/phase3/pdm_skew_analysis.json`; no tuning until the demo looks clean.

**Closed retrain loop.** `retrain_trigger` requires 3 *consecutive*
alert-severity physics `score_psi` windows (never a single window), then runs
train → ONNX export → parity gate (tol 1e-5) → p99 threshold + AUC eval into
a NEW model dir, and only after parity passes writes the atomic pointer file
`models/physics_vae/current.json` (`os.replace`; a file, not a symlink —
Windows — and not a control topic — inspectable, restart-safe). The running
scorer polls the pointer every 500 messages and keeps the old model on any
load failure. Retrain is demo-scale (6 epochs / 500 k events vs Phase 0's
20 / 2 M) so the loop closes in minutes on the 4 GB-VRAM laptop; a production
system would retrain on freshly captured background, not a Phase 0 replay —
recorded as a limitation.

**Dashboard.** Streamlit on the host reading topics directly (no DB): one
confluent-kafka consumer per background thread (client is not thread-safe),
fresh latest-offset groups, bounded deques, 2 s fragment refresh. Grafana out
of scope. Running containers remain Redpanda + Console only.

**Phase 3 measured results (`make phase3`, 10 k physics events @500 ev/s,
RFQ PDM skew @20 p/s).**
- *Black-box injection*: first drift ALERT 4.02 s / 2 002 scored messages
  after the switch — exactly one 2 000-event window, the monitor's resolution
  floor. The alerting metric is the leading-jet-pT feature (`f27_psi` ≈ 0.42
  sustained); score PSI plateaus in the warn band (~0.15–0.20) because the
  black box is mostly background-like. The retrain trigger therefore runs
  with `--min-severity warn` (justified in the demo script); confirmation is
  still 3 consecutive windows.
- *Closed loop*: confirmed drift → demo-scale retrain → ONNX parity PASS →
  pointer swap. AUC (A→4l) 0.766 → 0.775, threshold re-derived at p99
  (3.97e-05 — the retrained VAE has a different score scale, which is exactly
  why the threshold travels with the model in the pointer). A verification
  replay served the new model dir; no unverified model was ever swapped in.
- *PDM skew verdict — the honest one*: `real_shift_signature`. The file-head
  slice moved not only score PSI but ALL tracked raw channel-mean PSIs
  (1.5–3.0): the Phase 1 "calibration skew" is not a model artifact — the
  file head is genuinely a different input distribution than the normal-val
  split. The monitor cannot (and arguably should not) call it benign; the
  claimed score-only-vs-both signature did not separate these cases on this
  data, and that limitation is recorded in
  `reports/phase3/pdm_skew_analysis.json` rather than tuned away. What it
  does establish: the Phase 1 keep-rate mismatch traces to real sampling
  bias in the replay slice, not to a drifting model.

## Phase 4 — real-data ingestion + custom dashboard + final polish

**The real ROOT entry point.** Phases 0–3 ran entirely on simulation (ADC2021
Delphes physics, HVCM waveforms). Phase 4 makes ROOT *run on the real path*:
`services/ingest_root/ingest_nanoaod.py` reads a **CMS Open Data NanoAOD**
`Events` tree with PyROOT + RDataFrame (`EnableImplicitMT`, `Range` cap so the
12 GB WSL guest never OOMs), maps the physics objects onto the **same 57-feature
ADC2021 vector** the VAE was trained on, and writes a compact `(N,57)` `.npy`.
The object→slot map (MET slot 0 with eta≡0; leading 4 e / 4 μ / 10 jets;
zero-pad; PuppiMET with MET fallback) is documented in `docs/ingest_root.md`.

**Container/host split.** RDataFrame extraction runs *inside* the
`rootproject/root` one-shot container, but the Kafka client stays on the host:
`stream_cms.py` replays the `.npy` through the **exact** Phase 1 producer
interface (`pharos.physics.v1`, raw pre-normalization vectors). This honors the
"producers are host processes" rule and avoids fattening the ROOT image with
confluent-kafka — and, crucially, means real and simulated events travel the
identical wire format, so a real event flows through the frozen sim-trained
scorer unchanged.

**Sim-to-real domain gap — reported, not tuned.** Real CMS data has a different
object composition than the Delphes sim (trigger mix, pileup, multiplicities,
spectra), so the *encoding* is identical but the feature *distributions* differ.
`scripts/phase4_sim_vs_real.py` streams the real events through the unchanged ORT
scorer + drift monitor and records the resulting PSI/KS per tracked feature and
the anomaly score to `reports/phase4/sim_vs_real_drift.json`. The drift the
monitor reports here is a **legitimate domain-gap finding**, not a bug — no
artifact is retuned to close it. This is the same discipline as the Phase 3 PDM
negative result, now applied across the sim/real boundary.

**RDataFrame offline analysis.** `analysis/prep_adc_npy.py` (host: torch + h5py +
sklearn) scores background vs A→4ℓ with the active model pointer, extracts a few
physics observables straight from the 57-vector, and writes both the ROC/AUC
table (`reports/phase4/physics_auc_table.{json,md}`) and a columnar `.npz`.
`analysis/physics_rdf.py` (in-container) turns that `.npz` into overlaid
background-vs-signal histograms via `RDF.FromNumpy` in a single multithreaded
event loop. AUC stays host-side (needs torch/sklearn, absent from the ROOT
image); RDataFrame does the histogramming that exploits implicit MT. **Medians**
are quoted alongside means because the scores are heavy-tailed (recon-MSE means
are outlier-dominated). Latent AUC reproduces ~0.775, recon ~0.889.

**Custom dashboard — no Streamlit.** The Phase 3 Streamlit dashboard is replaced
by a hand-built **static** frontend (`dashboard_web/`: `index.html` + `styles.css`
+ `app.js`, vanilla, no build step, no framework) fed by ONE small **read-only
SSE bridge** (`services/dashboard_api/app.py`, stdlib `ThreadingHTTPServer`). The
bridge reuses the Streamlit threading model (one confluent-kafka consumer per
topic, fresh latest-offset groups, bounded deques) and only serves JSON — all
rendering/logic lives in the static JS, which draws the score histograms and
throughput sparklines by hand on `<canvas>` (no charting dependency). The panel
shows live throughput + keep-rate per stream, reference-vs-current score
histograms (overlaid, log-x), kept-vs-dropped counts, and the `alerts.drift`
feed. Design is a dark technical instrument panel; the series palette (reference
blue, current yellow, physics blue, PDM aqua, warn/alert status) was run through
the dataviz palette validator against the dark surface (all checks pass — the
dark-tuned yellow `#c98500` replaced a too-light first pick). Running containers
remain Redpanda + Console only.

**Final polish.** `docs/architecture.md` rewritten with the final Mermaid diagram
(three ROOT slots: RDataFrame ingestion [now real], RDataFrame analysis [now
real], SOFIE inference [deferred/documented]), a consolidated benchmark table,
a one-command demo, and a reproducibility note. A `README.md` now leads with the
project thesis and the three headline findings (Σμ²-vs-recon gap, hls4ml
precision→trigger-decision finding, the drift-separability negative result). No
retraining in Phase 4 — the active model pointer is reused.
