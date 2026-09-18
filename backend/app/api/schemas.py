"""Milestone B1 contract: Pydantic mirrors of the frontend's TypeScript types
(frontend/src/lib/pipeline/types.ts, findings/types.ts, cases/types.ts).
Field names must match exactly; `alias_generator` gives camelCase on the wire
while internal Python code uses snake_case.

Deliberate addition beyond the current frontend union (see module docstring
in app/pipeline/runner.py for the full rationale): StageState gains
`UNAVAILABLE`. `not-evaluated` means a gate blocked a stage that could have
run; `unavailable` means this build cannot run it at all (Wave 2 in B1).
Collapsing the two would tell an officer that Gate 1 blocked a check that
was never built.

MISMATCH WITH FRONTEND (report, do not silently fix on either side):
- StageCoverage (findings/types.ts:32-36) has {modality, ran, blockedBy} but
  no `unavailable` field. The runner requirement (coverage carries `ran` and,
  when false, `blockedBy` OR `unavailable: true`) needs that field. Added
  here on the backend; the frontend type needs a matching field in a
  follow-up, same as the StageState addition above.
- StageId in pipeline/types.ts is a kebab-case id ('mrz-read'); Modality in
  findings/types.ts is a different, overlapping-but-not-identical vocabulary
  ('mrz' | 'viz' | 'cross-check' | 'tamper' | 'ovd' | 'face' |
  'identity-graph'). StageCoverage.modality uses the Modality vocabulary, not
  StageId -- this file follows that distinction (see StageCoverage below).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from app.pipeline.registry import StageId, WaveId


def _camel_model() -> ConfigDict:
    return ConfigDict(alias_generator=to_camel, populate_by_name=True)


class StageState(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not-evaluated"
    UNAVAILABLE = "unavailable"  # stage not implemented in this build


class MismatchRow(BaseModel):
    model_config = _camel_model()
    field: str
    mrz: str
    viz: str


class Stage(BaseModel):
    model_config = _camel_model()
    id: StageId
    label: str
    wave: Optional[WaveId] = None
    is_gate: bool
    depends_on: list[StageId]
    state: StageState
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    detail: Optional[str] = None
    signal_ids: Optional[list[str]] = None
    blocked_by: Optional[StageId] = None
    mismatch_rows: Optional[list[MismatchRow]] = None


# --- findings/types.ts -------------------------------------------------

Severity = Enum("Severity", {"CRITICAL": "critical", "HIGH": "high", "MODERATE": "moderate", "INFORMATIONAL": "informational"}, type=str)
Verdict = Enum("Verdict", {"CLEARED": "cleared", "REVIEW_REQUIRED": "review-required", "FLAGGED": "flagged"}, type=str)
Modality = Enum(
    "Modality",
    {
        "MRZ": "mrz", "VIZ": "viz", "CROSS_CHECK": "cross-check", "TAMPER": "tamper",
        "OVD": "ovd", "FACE": "face", "IDENTITY_GRAPH": "identity-graph",
    },
    type=str,
)


class Region(BaseModel):
    model_config = _camel_model()
    x: float
    y: float
    w: float
    h: float


class Signal(BaseModel):
    model_config = _camel_model()
    signal_id: str
    modality: Modality
    label: str
    detail: str
    region: Optional[Region] = None
    emitted_at: str
    model_pin: str


class Finding(BaseModel):
    model_config = _camel_model()
    finding_id: str
    title: str
    hypothesis: str
    severity: Severity
    signals: list[Signal]
    converged_modalities: list[Modality]
    region: Optional[Region] = None


class StageCoverage(BaseModel):
    model_config = _camel_model()
    modality: Modality
    ran: bool
    blocked_by: Optional[str] = None
    unavailable: bool = False  # see module docstring: not on the frontend type yet


CheckDigitState = Enum("CheckDigitState", {"VALID": "valid", "INVALID": "invalid", "NONE": "none"}, type=str)
MrzCharRole = Enum(
    "MrzCharRole",
    {"NORMAL": "normal", "CHECK_DIGIT": "check-digit", "RECOVERED": "recovered", "FAILED_SIGNAL": "failed-signal"},
    type=str,
)


class MrzCharacter(BaseModel):
    model_config = _camel_model()
    char: str
    index: int
    role: MrzCharRole
    recovery_detail: Optional[str] = None
    finding_id: Optional[str] = None


class MrzFieldRow(BaseModel):
    model_config = _camel_model()
    field: str
    value: str
    check_digit_state: CheckDigitState


class MrzResult(BaseModel):
    model_config = _camel_model()
    format: str
    line_length: int
    lines: list[list[MrzCharacter]]
    fields: list[MrzFieldRow]
    composite_check_digit: dict[str, str]
    model_pin: str


CrossCheckStatus = Enum("CrossCheckStatus", {"MATCH": "match", "MISMATCH": "mismatch", "UNREADABLE": "unreadable"}, type=str)


class CrossCheckRow(BaseModel):
    model_config = _camel_model()
    field: str
    mrz_value: str
    viz_value: str
    status: CrossCheckStatus


class GraphEncounter(BaseModel):
    model_config = _camel_model()
    case_id: str
    date: str
    document_number_masked: str
    issuing_state: str
    decision: str


class GraphResult(BaseModel):
    model_config = _camel_model()
    ran: bool
    has_prior_encounter: bool
    enrolment_count: int
    severity: Optional[Severity] = None
    hypothesis: Optional[str] = None
    encounters: list[GraphEncounter] = []


class ScreeningResult(BaseModel):
    model_config = _camel_model()
    session_id: str
    verdict: Verdict
    reason: str
    findings: list[Finding]
    coverage: list[StageCoverage]
    mrz: Optional[MrzResult] = None
    cross_check: list[CrossCheckRow]
    identity_graph: GraphResult
    completed_at: str

    # NOT on the frontend type, and deliberately absent: no scalar risk/score
    # field exists anywhere in this contract (BACKEND_BRIEF.md §6.9's
    # fusion risk is an internal-only concept in B1; see runner.py header).


# --- stage-level event stream (mirrors pipeline/reducer.ts PipelineEvent) --


class StageEvent(BaseModel):
    model_config = _camel_model()
    type: str  # 'stage.started' | 'stage.settled'
    stage_id: StageId
    state: StageState
    at: str
    detail: Optional[str] = None
    signal_ids: Optional[list[str]] = None
    mismatch_rows: Optional[list[MismatchRow]] = None
    blocked_by: Optional[StageId] = None
    duration_ms: Optional[float] = None
