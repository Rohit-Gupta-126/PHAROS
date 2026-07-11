"""Phase 4 smoke tests: NanoAOD->57-vector slot mapping + SSE snapshot shape.

The slot-mapping tests need only numpy (the builder is deliberately ROOT/HDF5
free), so they run anywhere. The bridge snapshot test is skipped if
confluent-kafka is unavailable.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.nanoaod import build_event_vector, N_FEATURES


def test_vector_shape_and_met_slot():
    v = build_event_vector(met_pt=42.0, met_phi=0.7,
                           ele_pt=[], ele_eta=[], ele_phi=[],
                           mu_pt=[], mu_eta=[], mu_phi=[],
                           jet_pt=[], jet_eta=[], jet_phi=[])
    assert v.shape == (N_FEATURES,) == (57,)
    assert v.dtype == np.float32
    # MET occupies slot 0 with eta fixed at 0.
    assert list(v[0:3]) == [42.0, 0.0, 0.7]
    # empty collections -> everything past MET is zero-padded.
    assert not v[3:].any()


def test_leading_objects_and_truncation():
    v = build_event_vector(
        met_pt=10, met_phi=0.0,
        ele_pt=[30, 20, 10, 5, 2],          # 5 electrons -> only 4 slots
        ele_eta=[1, 2, 3, 4, 5], ele_phi=[.1, .2, .3, .4, .5],
        mu_pt=[25], mu_eta=[-0.8], mu_phi=[2.0],
        jet_pt=list(range(100, 40, -4)),    # 15 jets -> only 10 slots
        jet_eta=[0.1] * 15, jet_phi=[0.2] * 15)
    # e1 in slot 1, leading first.
    assert list(v[3:6]) == [30, 1, 0.1]
    # e4 in slot 4 is the 4th electron; the 5th is dropped.
    assert v[4 * 3] == 5.0
    # mu1 present in slot 5, mu2 (slot 6) zero-padded.
    assert v[5 * 3] == 25.0 and v[6 * 3] == 0.0
    # jet1 in slot 9; jet10 in slot 18 is the 10th of 15 jets (100 - 9*4 = 64).
    assert v[9 * 3] == 100.0
    assert v[18 * 3] == 100 - 9 * 4


def test_met_fallback_and_puppimet_selection():
    # (unit-level: the builder always uses whatever MET the caller passes;
    # branch selection is exercised in ingest_nanoaod._met_branches.)
    from services.ingest_root.ingest_nanoaod import _met_branches
    assert _met_branches({"PuppiMET_pt", "PuppiMET_phi", "MET_pt", "MET_phi"}) \
        == ("PuppiMET_pt", "PuppiMET_phi")
    assert _met_branches({"MET_pt", "MET_phi"}) == ("MET_pt", "MET_phi")
    with pytest.raises(RuntimeError):
        _met_branches({"Jet_pt"})


def test_bridge_snapshot_serializes():
    pytest.importorskip("confluent_kafka")
    import json
    from services.dashboard_api.app import Bridge
    # Build a Bridge but bypass live consumers: construct without threads.
    b = Bridge.__new__(Bridge)
    from collections import Counter, deque

    class _Fake:
        def __init__(self):
            self.records, self.n_total, self.kept = deque(), 0, Counter()

        def rate(self):
            return 0.0

        def recent_scores(self, limit=600):
            return []
    b.physics = _Fake(); b.pdm = _Fake(); b.drift = _Fake()
    b.reference = {}
    snap = b.snapshot()
    assert set(snap) >= {"physics", "pdm", "drift"}
    json.dumps(snap)  # must be JSON-serializable for SSE
