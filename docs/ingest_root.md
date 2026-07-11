# Real-data ingestion: CMS Open Data NanoAOD → PHAROS 57-vector (Phase 4)

This is PHAROS's **real** ROOT entry point. Phases 0–3 ran on the simulated
ADC2021 (Delphes) sample; Phase 4 ingests a real **CMS Open Data NanoAOD** file
with **PyROOT + RDataFrame** and maps each event onto the *same* ADC2021
57-feature representation the Stream A VAE was trained on — so real collisions
flow through the frozen sim-trained scorer and drift monitor unchanged.

## Pipeline (container ↔ host split)

```
rootproject/root:latest (one-shot)            host (pharos env)
┌───────────────────────────────────┐         ┌──────────────────────────────┐
│ fetch_nanoaod.sh   (xrdcp)         │         │ services/ingest_root/        │
│   → data/raw/cms_opendata/*.root   │         │   stream_cms.py              │
│ ingest_nanoaod.py  (RDataFrame)    │  npy    │   replays (N,57) through the │
│   Events tree → build 57-vectors   │────────►│   Phase 1 producer interface │
│   → data/interim/cms_events_57.npy │         │   → events.physics           │
└───────────────────────────────────┘         └──────────────────────────────┘
```

The Kafka client stays on the host (producers are host processes — RAM budget,
and we do not fatten the ROOT image with confluent-kafka). The ROOT container
only reads physics and writes a compact `(N, 57)` float32 `.npy`. The wire
format is **identical** to the sim producer (`pharos.physics.v1`, raw
pre-normalization vectors) — there is no second wire format.

## Object → slot mapping

NanoAOD stores per-event, pT-ordered collections. We map them onto the fixed
ADC2021 `(19, 4)` layout (`src/preprocessing/nanoaod.build_event_vector`),
truncating each collection to its slot budget and zero-padding the rest:

| ADC2021 slots | Object   | NanoAOD branches                          | Budget |
|---------------|----------|-------------------------------------------|--------|
| 0             | MET      | `PuppiMET_pt`, `PuppiMET_phi` (fallback `MET_*`) | 1 |
| 1–4           | Electrons| `Electron_pt/eta/phi`                     | 4      |
| 5–8           | Muons    | `Muon_pt/eta/phi`                         | 4      |
| 9–18          | Jets     | `Jet_pt/eta/phi`                          | 10     |

Each slot is `(pT, eta, phi)`; flattened in slot order → 57 features.

**Conventions (matching ADC2021):**
- **MET-eta ≡ 0** — MET has no pseudorapidity, so slot 0's eta is fixed at 0
  (a constant feature the normalizer already guards to 0).
- **Zero padding** — events with fewer objects than the slot budget leave the
  remaining slots all-zero (same as a padded ADC2021 event).
- **Leading objects** — NanoAOD collections are already pT-ordered, so the first
  N entries are the leading objects; extras beyond the budget are dropped.
- No selection/ID cuts are applied — we take the collections as stored, so the
  sim-to-real comparison is on raw object kinematics.

## The sim-to-real domain gap (expected, reported — not a bug)

Real CMS data has a **different object composition** than the ADC2021 Delphes
sim (trigger mix, pileup, object multiplicities, momentum spectra all differ).
The 57-feature *encoding* is identical, but the feature *distributions* differ,
so the frozen sim-trained monitor reports drift. This is the **domain gap**, a
legitimate finding — `scripts/phase4_sim_vs_real.py` streams the CMS vectors
through the unchanged scorer + monitor and records the PSI/KS per tracked feature
and the anomaly score to `reports/phase4/sim_vs_real_drift.json`. We do **not**
retune anything to close it.

## Memory safety on the 12 GB WSL guest

- `ROOT.EnableImplicitMT()` — multithreaded event loop.
- `RDataFrame.Range(--limit)` caps the number of events materialized (default
  50 000); only the needed branches are pulled via `AsNumpy`. Peak stays well
  under the guest budget because the full file is never held in memory — for a
  larger sample, lower `--limit` or ingest in ranges.
- The source may be a **local file** (from `fetch_nanoaod.sh`) or a `root://`
  URL that RDataFrame **streams** directly (no local multi-GB copy).

## Running it

```bash
# 1. fetch a NanoAOD file (one-shot ROOT container; override CMS_NANOAOD_URL)
make fetch-cms
# 2. extract 57-vectors with RDataFrame (one-shot ROOT container)
make ingest-cms
# 3. broker up, then stream the real events + measure the domain gap
make up
make sim-vs-real        # → reports/phase4/sim_vs_real_drift.json
```

The default source is a Run2016 UL NanoAODv9 `DoubleMuon` file; any NanoAOD with
the standard `Electron/Muon/Jet/PuppiMET` branches works. If the default file is
too large for the guest or the endpoint is unreachable, pick a smaller Open Data
record (`CMS_NANOAOD_URL=…`) rather than grinding — the project stop-rule.
