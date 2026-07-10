# PHAROS architecture (as built through Phase 1)

```
                       Docker (broker profile only)
              ┌────────────────────────────────────────────┐
              │  Redpanda (Kafka API, localhost:9092)       │
              │  --smp 1 --memory 1500M --overprovisioned   │
              │  Redpanda Console UI  → localhost:8080      │
              └────────────────────────────────────────────┘
                    ▲                              │
   host (pharos env)│ produce                      │ consume        host (pharos env)
┌───────────────────┴────────┐        ┌────────────▼──────────────────────┐
│ services/producers/        │        │ services/scorers/                 │
│  physics_producer          │        │  physics_scorer                   │
│   ADC2021 tail region      │──────► │   Phase 0 norm.npz + VAE ckpt     │
│   raw 57-dim vectors       │events. │   log1p/z-score → Σμ²             │
│                            │physics │   > p99 threshold → keep          │──► anomalies.scouting
│  pdm_producer              │        │  pdm_scorer                       │
│   HVCM pulses, avg-pooled  │──────► │   per-system channel_norm + conv AE│
│   (500,14) pre-norm waves  │events. │   recon MSE > p99 threshold → keep│──► alerts.pdm
└────────────────────────────┘ pdm    └───────────────────────────────────┘
                                              │
                                              └─► reports/phase1/ (latency,
                                                  throughput, keep-rate)
```

## Topics (all single-partition)

| topic                | payload schema        | producer          | consumer        |
|----------------------|-----------------------|-------------------|-----------------|
| `events.physics`     | `pharos.physics.v1`   | physics_producer  | physics_scorer  |
| `events.pdm`         | `pharos.pdm.v1`       | pdm_producer      | pdm_scorer      |
| `anomalies.scouting` | `pharos.scouting.v1`  | physics_scorer    | (Phase 2+)      |
| `alerts.pdm`         | `pharos.pdm_alert.v1` | pdm_scorer        | (Phase 2+)      |

See [wire_format.md](wire_format.md) for field-level detail.

## Design decisions

- **Only the broker is containerized.** Producers/scorers are host Python
  processes in the `pharos` conda env (WSL) — keeps the 12 GB WSL guest within
  budget and gives the scorers direct GPU access.
- **Producers emit pre-normalization data**; scorers load the Phase 0
  normalization stats and model checkpoints (frozen — nothing is refit), so the
  transform has a single source of truth and streaming scores match Phase 0.
- **Thresholds are derived, not arbitrary**: `make thresholds` scores the
  Phase 0 background/normal samples with the frozen artifacts and stores the
  configured percentile (default p99 → ~1% background keep-rate) in
  `configs/thresholds.json`.
- Micro-batching in the physics scorer amortizes GPU calls; the PDM stream is
  low-rate and scored per pulse.

## Running it

```
make up               # Redpanda + Console + topics
make thresholds       # (re)derive keep thresholds from Phase 0 backgrounds
make phase1           # short end-to-end demo, metrics → reports/phase1/
make down
```
