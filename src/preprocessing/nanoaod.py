"""CMS Open Data NanoAOD -> ADC2021 57-feature event vector.

The ADC2021 schema (``src/preprocessing/adc2021.py``) is a fixed ``(19, 4)``
object table flattened to 57 features ``[pT, eta, phi] x 19`` in the canonical
slot order::

    slot  0        : MET               (eta == 0 by convention)
    slots 1..4     : up to 4 electrons  (pT-ordered)
    slots 5..8     : up to 4 muons      (pT-ordered)
    slots 9..18    : up to 10 jets      (pT-ordered)

Real CMS NanoAOD stores the same physics objects as per-event variable-length
branches (``Electron_pt/eta/phi``, ``Muon_*``, ``Jet_*``, ``PuppiMET_pt/phi``),
already pT-ordered. This module maps one NanoAOD event onto the ADC2021 slots by
truncating each collection to its slot budget and zero-padding the rest -- the
SAME representation Stream A's VAE was trained on, so a real event can flow
through the frozen sim-trained scorer. Because real data has a different object
composition than the Delphes sim, the feature *distributions* differ (the
sim-to-real domain gap PHAROS reports in Phase 4); the *encoding* is identical.

This module is deliberately ROOT-free (pure numpy) so the slot mapping is unit
testable on the host; the RDataFrame job in ``services/ingest_root`` feeds it
per-event Python lists pulled from the ``Events`` tree.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Kept in lock-step with src/preprocessing/adc2021.py (N_SLOTS=19, N_KIN=3), but
# defined locally so this ROOT/HDF5-free builder imports with only numpy -- the
# unit test and the ROOT container both need it without pulling in h5py.
N_KIN = 3                      # (pT, eta, phi) per slot
N_FEATURES = 19 * N_KIN        # 57

# (slot offset, slot count) for each collection, matching the ADC2021 layout.
MET_SLOT = 0
ELECTRON_SLOTS = (1, 4)
MUON_SLOTS = (5, 8)
JET_SLOTS = (9, 18)
N_ELECTRON = 4
N_MUON = 4
N_JET = 10


def _fill(vec: np.ndarray, start_slot: int, budget: int,
          pt: Sequence[float], eta: Sequence[float], phi: Sequence[float]
          ) -> None:
    """Write up to ``budget`` leading objects into consecutive slots.

    NanoAOD collections are already pT-ordered, so we take the first ``budget``
    entries as the leading objects; the rest of the slots stay zero-padded.
    """
    n = min(budget, len(pt))
    for i in range(n):
        base = (start_slot + i) * N_KIN
        vec[base + 0] = pt[i]
        vec[base + 1] = eta[i]
        vec[base + 2] = phi[i]


def build_event_vector(met_pt: float, met_phi: float,
                       ele_pt: Sequence[float], ele_eta: Sequence[float],
                       ele_phi: Sequence[float],
                       mu_pt: Sequence[float], mu_eta: Sequence[float],
                       mu_phi: Sequence[float],
                       jet_pt: Sequence[float], jet_eta: Sequence[float],
                       jet_phi: Sequence[float]) -> np.ndarray:
    """Map one NanoAOD event to the pre-normalization ADC2021 57-vector.

    MET occupies slot 0 with ``eta == 0`` (MET has no pseudorapidity -- the same
    constant-slot convention as ADC2021). Returns float32 of length 57, ready to
    hand to ``common.produce_json`` exactly like the sim producer's ``features``.
    """
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    # MET: slot 0, meaningful pT/phi, eta fixed at 0.
    vec[MET_SLOT * N_KIN + 0] = met_pt
    vec[MET_SLOT * N_KIN + 1] = 0.0
    vec[MET_SLOT * N_KIN + 2] = met_phi
    _fill(vec, ELECTRON_SLOTS[0], N_ELECTRON, ele_pt, ele_eta, ele_phi)
    _fill(vec, MUON_SLOTS[0], N_MUON, mu_pt, mu_eta, mu_phi)
    _fill(vec, JET_SLOTS[0], N_JET, jet_pt, jet_eta, jet_phi)
    return vec
