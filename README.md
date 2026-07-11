# PHAROS

**Pipeline for High-throughput Anomaly Recognition in Online Streams**

PHAROS is a real-time streaming anomaly-detection service that emulates the last
stage of an LHC trigger/DAQ system. Two unsupervised detectors share one
Kafka-protocol backbone — a model-independent **new-physics event filter**
(AXOL1TL/CICADA-style variational autoencoder, anomaly score = Σμ²) and
**accelerator predictive maintenance** on real power-system waveforms — behind an
L1-style rate-control decision layer and an MLOps drift/retrain loop. It runs
end-to-end on a 16 GB laptop with a 4 GB-VRAM GPU: PyTorch training, ONNX-Runtime
trigger inference, Redpanda streaming, ROOT/RDataFrame ingestion of **real CMS
Open Data**, and a hand-built instrument-panel dashboard.

> **Thesis.** A cheap, model-independent latent trigger score can filter LHC-like
> event streams at a realistic L1 budget (**115× reduction**) with microsecond
> inference — and the interesting engineering lives in the *gaps*: what the cheap
> score gives up, what fixed-point precision costs the trigger decision, and what
> a drift monitor genuinely can and cannot distinguish.

## Three headline findings

1. **The trigger-vs-accuracy gap (Σμ² vs reconstruction).** The FPGA-cheap
   AXOL1TL-style latent score Σμ² reaches **AUC 0.775** on A→4ℓ, while the full-VAE
   reconstruction MSE from the *same* checkpoint reaches **0.889**. The 0.11 gap
   quantifies exactly what the deployable trigger sacrifices versus running the
   decoder offline — signal events reconstruct far worse than their latent means
   suggest. Σμ² stays the trigger; recon-MSE is the offline discriminator.

2. **Fixed-point precision changes the trigger *decision*, not just the number.**
   In the hls4ml study, the tutorial-default `ap_fixed<16,6>` gives only **91%**
   p99 decision agreement — because μ sits near a ~1e-3 threshold and 10 fractional
   bits are too coarse there. Widening to **`ap_fixed<24,8>` restores 100%**
   agreement. The precision knob is a *trigger-efficiency* knob, not an academic
   rounding detail.

3. **A drift monitor's negative result, reported honestly.** The intended
   "benign calibration skew vs real distribution shift" signature **did not
   separate** on the PDM data: the Phase 1 keep-rate mismatch turned out to move
   *all* tracked raw channel-mean PSIs, so the monitor's verdict is
   `real_shift_signature` — the replay slice is genuinely a different input
   distribution, not a model artifact. We record the limitation rather than tune a
   clean-looking demo. Likewise, Phase 4 streams **real CMS Open Data** through the
   frozen sim-trained monitor and reports the resulting **sim-to-real domain gap**
   as a finding, not a bug.

## Pipeline at a glance

```
CMS Open Data NanoAOD ─(RDataFrame)─┐
ADC2021 sim ───────────────────────┴─► events.physics ─► VAE scorer (ORT, 7.4µs)
                                                          │  Σμ² > p99 threshold
                                                          ▼
                                          L1 budget decision ─► anomalies.scouting  (115× reduction)
                                                          │
HVCM waveforms ─► conv-AE PDM scorer ─┐                   ▼
                                      └─► drift monitor (PSI/KS) ─► alerts.drift ─► retrain (parity-gated hot-swap)
                                                          │
                                          dashboard_api (SSE) ─► static web dashboard
```

See [docs/architecture.md](docs/architecture.md) for the full diagram, the three
ROOT slots (RDataFrame ingestion — real; RDataFrame analysis — real; SOFIE
inference — deferred/documented), the benchmark table, and the topic map.

## Quickstart

Requires Docker (Redpanda + one-shot ROOT containers) and the `pharos` conda env
(PyTorch, confluent-kafka, onnxruntime — see `requirements-phase*.txt`).

```bash
make up                 # Redpanda + Console + topics (only long-running containers)
make phase1             # streaming backbone demo        → reports/phase1/
make phase2             # ORT inference + L1 decision     → reports/phase2/
make phase3             # drift monitor + retrain loop    → reports/phase3/

# Phase 4 — real data + dashboard:
make fetch-cms          # xrdcp a CMS Open Data NanoAOD file (one-shot ROOT container)
make ingest-cms         # RDataFrame → data/interim/cms_events_57.npy
make sim-vs-real        # stream real events, report the domain gap → reports/phase4/
make analysis-prep analysis-rdf   # RDataFrame physics plots + AUC table → reports/phase4/
make dashboard-api      # live dashboard at http://127.0.0.1:8070/
make down
```

## Layout

| Path | What |
|------|------|
| `src/` | training, preprocessing (`adc2021`, `nanoaod`, `hvcm`), inference scores |
| `services/` | producers, scorers, decision, monitor, `ingest_root`, `dashboard_api` |
| `analysis/` | RDataFrame offline analysis (Phase 4) |
| `dashboard_web/` | static HTML/CSS/JS instrument-panel dashboard (no framework) |
| `docs/` | architecture, schema, wire format, ingest, hls4ml, **design log** |
| `reports/phaseN/` | per-phase metrics and plots |
| `models/` | frozen artifacts + `physics_vae/current.json` active pointer |

## Constraints & honesty notes

- Runs within a 16 GB / 4 GB-VRAM budget; only Redpanda + Console stay resident,
  ROOT is one-shot, models are kept small.
- SOFIE inference is **deferred** (documented recipe); ONNX Runtime is the runnable
  deploy path. hls4ml is a **feasibility report** (no local Vivado).
- Phase 4 does **not** retrain — it reuses the active model pointer.
- Full decision history in [docs/design_log.md](docs/design_log.md).
