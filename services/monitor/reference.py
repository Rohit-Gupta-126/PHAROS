"""Load the frozen Phase 0 reference distributions for the drift monitor."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from services.monitor.drift_stats import ReferenceDist


@dataclass(frozen=True)
class Reference:
    """One tracked quantity: PSI histogram + raw samples for KS/overlays."""

    dist: ReferenceDist
    samples: np.ndarray

    @classmethod
    def from_entry(cls, entry: Dict) -> "Reference":
        return cls(dist=ReferenceDist.from_dict(entry),
                   samples=np.asarray(entry["samples"], dtype=np.float64))


@dataclass(frozen=True)
class PhysicsReferences:
    score: Reference
    features: Dict[str, Reference]  # "f00" -> Reference (raw feature units)

    @classmethod
    def load(cls, path: Path) -> "PhysicsReferences":
        d = json.loads(path.read_text())
        return cls(score=Reference.from_entry(d["score"]),
                   features={k: Reference.from_entry(v)
                             for k, v in d["features"].items()})


@dataclass(frozen=True)
class PdmReferences:
    tracked_system: str
    scores: Dict[str, Reference]         # per system
    channel_means: Dict[str, Reference]  # "ch00" -> Reference (tracked system)

    @classmethod
    def load(cls, path: Path) -> "PdmReferences":
        d = json.loads(path.read_text())
        systems = d["systems"]
        tracked = d["tracked_system"]
        return cls(
            tracked_system=tracked,
            scores={s: Reference.from_entry(v["score"])
                    for s, v in systems.items()},
            channel_means={k: Reference.from_entry(v) for k, v in
                           systems[tracked].get("channel_means", {}).items()})
