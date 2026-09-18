"""Wire models for milestone B2: decision submission, the case register, and
the config audit trail. Mirrors frontend/src/lib/audit/types.ts's
`AuditEntry`, frontend/src/lib/cases/types.ts's `CaseSummary`, and
frontend/src/lib/config/types.ts's `Threshold`/`ConfigChange` -- same
approach as app/api/b1_wire.py: small camelCase Pydantic models for
resources app/api/schemas.py doesn't own.

Deliberate additions beyond the current frontend `AuditEntry` type (report,
per this milestone's own instructions, rather than silently reconciling):
- `runId` is on the wire here (the spec's own hash formula names it) but
  `AuditEntry` in audit/types.ts has no `runId` field yet.
- `findingIds` is on the frontend type but is *not* part of the spec's named
  hash-input field set; it's included here as a display-only convenience
  derived from `findingDispositions`, and is not fed into `canonical_json`
  (see app/audit/chain.py's `build_entry_payload`, which omits it).
"""
from __future__ import annotations

import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


def _camel_model() -> ConfigDict:
    return ConfigDict(alias_generator=to_camel, populate_by_name=True)


class EvidenceHashIn(BaseModel):
    model_config = _camel_model()
    kind: str
    sha256: str


class FindingDispositionIn(BaseModel):
    model_config = _camel_model()
    finding_id: str
    disposition: str  # 'accepted' | 'dismissed'
    reason: Optional[str] = None


class DecisionIn(BaseModel):
    """Deliberately has no `divergence` field at all -- spec §2: "divergence
    is computed server-side ... never accepted from the request body." A
    client that includes one anyway is not rejected (Pydantic's default
    `extra="ignore"`, matching every other B1/B2 request model in this
    codebase); it is simply never read. See app/api/b2_decisions.py's test
    asserting a client-supplied divergence changes nothing."""
    model_config = _camel_model()
    officer_id: str
    terminal_id: str
    action: str  # OfficerAction
    finding_dispositions: list[FindingDispositionIn]
    notes: str = ""
    divergence_reason: Optional[str] = None
    attestation: bool = False


class DecisionOut(BaseModel):
    model_config = _camel_model()
    case_id: str
    session_id: str
    entry_hash: str
    entry_id: str


class AuditEntryOut(BaseModel):
    model_config = _camel_model()
    entry_id: str
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    previous_hash: str
    entry_hash: str
    sealed_at: str
    officer_id: str
    terminal_id: str
    evidence_hashes: list[EvidenceHashIn]
    signal_ids: list[str]
    finding_ids: list[str] = Field(default_factory=list)  # display-only, see module docstring
    finding_dispositions: list[FindingDispositionIn]
    model_pins: dict[str, str]
    thresholds: dict[str, float]
    recommendation: Optional[str] = None
    officer_action: Optional[str] = None
    divergence: Optional[str] = None
    divergence_reason: Optional[str] = None
    notes: Optional[str] = None
    entry_type: str = "decision"


class ChainVerifyOut(BaseModel):
    model_config = _camel_model()
    intact: bool
    broken_entry_id: Optional[str] = None
    reason: Optional[str] = None
    entries_checked: int = 0


class ChainHeadOut(BaseModel):
    """Read-only pre-seal preview support (spec §5): the hash a new entry on
    this terminal would reference as its previousHash right now."""

    model_config = _camel_model()
    terminal_id: str
    previous_hash: str


class CaseSummaryOut(BaseModel):
    model_config = _camel_model()
    case_id: str
    session_id: str
    document_class: Optional[str] = None
    issuing_state: Optional[str] = None
    document_number_masked: Optional[str] = None
    screened_at: datetime.datetime
    officer_id: Optional[str] = None
    officer_name: Optional[str] = None
    recommendation: Optional[str] = None
    officer_action: Optional[str] = None
    divergence: Optional[str] = None
    divergence_reason: Optional[str] = None
    coverage_complete: bool
    notes: Optional[str] = None
    entry_hash: Optional[str] = None


class CaseListOut(BaseModel):
    model_config = _camel_model()
    cases: list[CaseSummaryOut]
    total: int
    page: int
    page_size: int
    linked_unavailable: bool = False  # spec §3: an explicit indicator, never a silent empty


class CaseRecordOut(BaseModel):
    model_config = _camel_model()
    case: CaseSummaryOut
    result: Optional[dict[str, object]] = None  # the full ScreeningResult, camelCase-on-wire JSON
    decision: Optional[AuditEntryOut] = None


class ThresholdOut(BaseModel):
    model_config = _camel_model()
    id: str
    label: str
    gate: Optional[str] = None
    value: float
    unit: str
    min: float
    max: float
    step: float
    default: float
    description: str


class ThresholdChangeIn(BaseModel):
    model_config = _camel_model()
    id: str
    value: float


class ThresholdsUpdateIn(BaseModel):
    model_config = _camel_model()
    officer_id: str
    terminal_id: str
    changes: list[ThresholdChangeIn]
    reason: str


class ConfigChangeOut(BaseModel):
    model_config = _camel_model()
    change_id: str
    changed_at: str
    changed_by: str
    reason: str
    changes: list[dict[str, object]]
    audit_entry_hash: str
