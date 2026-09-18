"""Stage registry -- mirrors frontend/src/lib/pipeline/types.ts STAGE_REGISTRY
exactly (same ids, wave, isGate, dependsOn, blocksOnFailure). The runner
(app/pipeline/runner.py) resolves execution order and blocking from this
table; it is never hardcoded in the runner itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StageId(str, Enum):
    MRZ_READ = "mrz-read"
    VIZ_READ = "viz-read"
    GATE_1 = "gate-1"
    TAMPER = "tamper"
    OVD_SWEEP = "ovd-sweep"
    FACE_VERIFY = "face-verify"
    IDENTITY_GRAPH = "identity-graph"
    CONVERGENCE = "convergence"
    GATE_2 = "gate-2"
    DECISION = "decision"


class WaveId(str, Enum):
    WAVE_1 = "wave-1"
    WAVE_2 = "wave-2"


@dataclass(frozen=True)
class StageDefinition:
    id: StageId
    label: str
    wave: Optional[WaveId]
    is_gate: bool
    depends_on: tuple[StageId, ...]
    blocks_on_failure: bool = False


STAGE_REGISTRY: tuple[StageDefinition, ...] = (
    StageDefinition(StageId.MRZ_READ, "MRZ read · CRNN-CTC", WaveId.WAVE_1, False, ()),
    StageDefinition(StageId.VIZ_READ, "VIZ read · RapidOCR", WaveId.WAVE_1, False, ()),
    StageDefinition(
        StageId.GATE_1, "Gate 1 · MRZ ↔ VIZ cross-check", None, True,
        (StageId.MRZ_READ, StageId.VIZ_READ), blocks_on_failure=True,
    ),
    StageDefinition(StageId.TAMPER, "Tamper forensics · ResNet", WaveId.WAVE_2, False, (StageId.GATE_1,)),
    StageDefinition(StageId.OVD_SWEEP, "OVD sweep verify", WaveId.WAVE_2, False, (StageId.GATE_1,)),
    StageDefinition(StageId.FACE_VERIFY, "Face verify · ArcFace", WaveId.WAVE_2, False, (StageId.GATE_1,)),
    StageDefinition(StageId.IDENTITY_GRAPH, "Identity graph lookup", WaveId.WAVE_2, False, (StageId.GATE_1,)),
    StageDefinition(
        StageId.CONVERGENCE, "Convergence grouping", None, False,
        (StageId.TAMPER, StageId.OVD_SWEEP, StageId.FACE_VERIFY, StageId.IDENTITY_GRAPH),
    ),
    StageDefinition(
        StageId.GATE_2, "Gate 2 · Authentic and physically real", None, True,
        (StageId.CONVERGENCE,), blocks_on_failure=False,
    ),
    StageDefinition(StageId.DECISION, "Decision", None, False, (StageId.GATE_2,)),
)

REGISTRY_BY_ID: dict[StageId, StageDefinition] = {d.id: d for d in STAGE_REGISTRY}

WAVE_1_STAGE_IDS: tuple[StageId, ...] = tuple(
    d.id for d in STAGE_REGISTRY if d.wave is WaveId.WAVE_1
)
WAVE_2_STAGE_IDS: tuple[StageId, ...] = tuple(
    d.id for d in STAGE_REGISTRY if d.wave is WaveId.WAVE_2
)


def dependents_of(stage_id: StageId) -> list[StageId]:
    """Direct dependents (stages whose dependsOn includes stage_id)."""
    return [d.id for d in STAGE_REGISTRY if stage_id in d.depends_on]
