# PHAROS — project context for Claude Code

PHAROS (Pipeline for High-throughput Anomaly Recognition in Online Streams) is a
real-time streaming anomaly-detection service emulating the last stage of an LHC
trigger/DAQ system. It runs two unsupervised detectors on one Kafka-protocol
backbone: (A) model-independent new-physics event filtering (AXOL1TL/CICADA-style
variational autoencoder), and (B) accelerator predictive maintenance on real
power-system waveforms. A decision/rate-control layer emulates the L1 trigger
budget; an MLOps layer monitors concept drift and triggers retraining.

## Reference points (mirror these, at student scale)
- CMS AXOL1TL / CICADA Run-3 L1 anomaly triggers; anomaly score = sum of squared
  VAE latent means.
- CERN "Next Generation Triggers" programme; hls4ml FPGA inference; autoencoder DQM.

## Tech stack
- ML: PyTorch (train on RTX 2050, 4 GB VRAM — keep models small, batches modest).
- Inference deploy: ONNX -> TMVA SOFIE (C++, ROOT) as the runnable path; hls4ml as
  an FPGA feasibility *report* (no local Vivado); ONNX Runtime as fallback.
- Streaming: Redpanda (Kafka API) single node; Python consumers with confluent-kafka.
  NO Flink/Spark on this 16 GB laptop.
- Ingestion/analysis: ROOT via `rootproject/root` Docker image — PyROOT + RDataFrame.
- MLOps: Evidently / Alibi Detect / river. Dashboard: Streamlit (Grafana optional).
- Orchestration: docker compose, with profiles; Makefile targets per component.

## Hardware constraints (respect these)
- RTX 2050, 4 GB VRAM; 16 GB system RAM. Keep the container footprint small; cap
  Redpanda memory; don't run every service at once during dev.

## Conventions
- Plan before implementing. Small, phase-tagged commits (e.g. feat(phase1): ...).
- Every component gets a Makefile target and a tests/ smoke test.
- Record metrics to reports/phaseN/. Record decisions to docs/design_log.md.
- ROOT/SOFIE and hls4ml are fragile: if a toolchain build is too heavy, STOP and
  ask rather than grinding; fall back to ONNX Runtime and document it.