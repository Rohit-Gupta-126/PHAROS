# ADC2021 dataset schema (inferred)

The ADC2021 files are the Delphes-simulated LHC event samples used by the CERN
"Unsupervised New Physics detection at 40 MHz" (Anomaly Detection Challenge 2021)
and the AXOL1TL / CICADA L1-trigger anomaly work. This document records the
schema PHAROS Stream A relies on, as inferred directly from the HDF5 files in
`data/raw/adc2021/`.

## Files

| File | `Particles` shape | Role |
|------|-------------------|------|
| `background_for_training.h5` | `(13,451,915, 19, 4)` | SM background cocktail; training data |
| `Ato4l_lepFilter_13TeV_filtered.h5` | `(55,969, 19, 4)` | Signal: A → 4 leptons (BSM) |
| `BlackBox_13TeV_PU20.h5` | `(4,210,492, 19, 4)` | Mixed "black box" sample (also has `EvtId`) |

## HDF5 layout

Each file contains:

- **`Particles`** — `float64`, shape `(N_events, 19, 4)`. The last axis is
  `[pT, eta, phi, class]` (confirmed by `Particles_Names = [Pt, Eta, Phi, Class]`).
- **`Particles_Names`** — `[b'Pt', b'Eta', b'Phi', b'Class']`.
- **`Particles_Classes`** — `[b'MET_class_1', b'Four_Ele_class_2',
  b'Four_Mu_class_3', b'Ten_Jet_class_4']`.
- **`EvtId`** — present only in the black-box file (`int64`, one id per event).

### The 19 object slots (fixed layout)

Every event is a fixed-size `(19, 4)` array with a canonical object ordering:

| Rows | Object | Class id | Count |
|------|--------|----------|-------|
| 0 | Missing transverse energy (MET) | 1 | 1 |
| 1–4 | Electrons | 2 | up to 4 |
| 5–8 | Muons | 3 | up to 4 |
| 9–18 | Jets | 4 | up to 10 |

Total: `1 + 4 + 4 + 10 = 19` slots.

### Padding and conventions

- **Zero padding**: unused object slots are all-zero with `class = 0`
  (e.g. an event with 2 electrons fills rows 1–2 and zero-pads rows 3–4).
- **MET**: has meaningful `pT` and `phi` but `eta ≡ 0` (MET has no
  pseudorapidity). This makes the MET-eta feature constant.
- **Units/ranges**: `pT` in GeV (heavy-tailed, observed up to ~475 GeV in a
  sample); `eta` roughly within detector acceptance; `phi ∈ [-π, π]`.
- Objects within a class are `pT`-ordered (leading object first).

## PHAROS feature tensor (Stream A)

Implemented in [`src/preprocessing/adc2021.py`](../src/preprocessing/adc2021.py):

1. Drop the `class` column → keep `[pT, eta, phi]` per slot.
2. Flatten `(N, 19, 3)` → **`(N, 57)`** in slot order
   `MET, e1..e4, mu1..mu4, jet1..jet10`, each as `(pt, eta, phi)`.
   Feature names are exposed as `adc2021.FEATURE_NAMES`.
3. **Normalization** (fit on the training split only, saved as `norm.npz`):
   - `pT` → `log1p(pT)` then per-feature z-score (tames the heavy tail);
   - `eta`, `phi` → per-feature z-score.
   - Features with near-zero std (constant MET-eta, always-padded slots) are
     guarded (`std → 1`) so they map to 0 rather than exploding.

The background is subsampled (default 2,000,000 events, deterministic seed) and
split 90/10 into train/val — both drawn from **background only**, since Stream A
is trained unsupervised. Signal files are normalized with the *background*
normalizer at evaluation time.

## Anomaly score

Stream A uses a small VAE (latent dim 8). The anomaly score is the **sum of
squared latent means**, `Σ μ²`, following the AXOL1TL / CICADA convention
(higher = more anomalous). See
[`src/inference/scores.py`](../src/inference/scores.py).
