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
│ fetch_nanoaod.sh   (curl HTTPS)    │         │ services/ingest_root/        │
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

- `RDataFrame.Range(--limit)` caps the number of events materialized (default
  50 000); only the needed branches are pulled via `AsNumpy`. Peak stays well
  under the guest budget because the full file is never held in memory — for a
  larger sample, lower `--limit` or ingest in ranges.
- `Range()` and `ROOT.EnableImplicitMT()` are **mutually exclusive** in
  RDataFrame (`Range` throws under implicit MT), so a capped run (`--limit > 0`)
  runs **single-threaded** on purpose. Only a full-file ingest (`--limit 0`)
  enables the multithreaded event loop. The 50 000-event cap is single-pass fast
  regardless.
- Ingestion reads a **local file** written by `fetch_nanoaod.sh`. We do **not**
  stream `root://` directly: `rootproject/root:latest` no longer bundles an
  XRootD client (no `xrdcp`, no `libXrdCl.so.3`), so both xrdcp and RDataFrame
  `root://` streaming are unavailable. We fetch over HTTPS to a local copy first.

## Fetching over HTTPS (`curl -k` + adler32)

`fetch_nanoaod.sh` downloads the NanoAOD file with `curl` over HTTPS instead of
`xrdcp`. Two rationale points, documented here and in the script:

- **`curl -k`** — `eospublic.cern.ch` redirects to an EOS gateway whose TLS
  chain is the CERN Grid CA, which is not in the ROOT image's default CA bundle.
  Rather than install the CERN CA into the container, we skip TLS verification
  and instead establish trust via the file checksum.
- **adler32 integrity check** — after download the script recomputes the file's
  adler32 and fails loudly (deleting the partial file) if it does not match the
  value published in the Open Data record's file index. This is the real
  integrity guarantee and does not depend on the TLS trust chain.

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

The default source is the CMS Open Data **ZeroBias PFNano** file (record 31316,
`/ZeroBias/Run2016G-UL2016_MiniAODv2_PFNanoAODv1`, ~1.05 GB). ZeroBias is an
inclusive/unbiased minimum-bias sample (random bunch-crossing readout, no physics
trigger) — the right basis for a sim-to-real domain-gap study, unlike a triggered
primary dataset. It carries the standard `Electron/Muon/Jet/(Puppi)MET` branches
plus PF candidates. Override with `CMS_NANOAOD_URL=…` (and `CMS_ADLER32=…`, or
empty to skip the checksum) for a different file; if the endpoint is unreachable
or the file is too large for the guest, pick a smaller Open Data record rather
than grinding — the project stop-rule.
