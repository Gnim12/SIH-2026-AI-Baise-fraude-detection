"""BACKEND_BRIEF.md §4 / FRONTEND_BRIEF.md §3: the document, face, graph and
session contracts, ported field-for-field.

Two deliberate departures from a literal transcription of the frontend's
`.ts` file, both flagged in the two briefs themselves rather than silently
added:

- `ExtractedField.mismatch`, `MrzGroup.expected/read/signal_id`: settled with
  the frontend already (BACKEND_BRIEF.md §4's MrzGroup docstring; the
  frontend's own `screening.ts` already carries `expected`/`read`/`signalId`
  on `MrzLine['groups']`), included from day one per the task brief.
- `ScreeningSession.model_versions`: FRONTEND_BRIEF.md's `screening.ts` flags
  this as a CONTRACT GAP against BACKEND_BRIEF.md §8.2's audit record, which
  assumes `model_versions` exists on every session. Added here to close that
  gap; the frontend simply ignores the extra field until it adds its own.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from .signals import Region, Signal

Decision = Literal["CLEAR", "SECONDARY", "HOLD", "REFER"]
SystemBand = Literal["CLEAR", "SECONDARY", "HOLD", "ABSTAIN", "RECAPTURE"]
DocType = Literal["PASSPORT", "VISA", "NATIONAL_ID", "LICENCE", "PERMIT", "UNKNOWN"]
FieldSource = Literal["MRZ", "VIZ", "MERGED"]
MrzFormat = Literal["TD1", "TD2", "TD3", "MRV-A", "MRV-B"]
MrzStatus = Literal["VERIFIED", "UNRECOVERABLE"]
FaceStatus = Literal["MATCH", "MISMATCH", "SPOOF", "UNAVAILABLE"]
PadVerdict = Literal["live", "spoof", "not_run"]
CaptureMethod = Literal["live", "upload"]
ViewKey = Literal["rgb", "ela", "noise", "heatmap", "fft"]


class ExtractedField(BaseModel):
    key: str  # 'birth_date'
    label: str  # 'Date of birth'
    value: str
    confidence: float  # 0..1
    source: FieldSource
    mismatch: Optional[bool] = None  # MRZ and VIZ disagree on this field


class MrzGroup(BaseModel):
    """Mirrors mrz/spec.py's CheckGroup. Emit real indices from the format
    layout, not approximations -- the UI underlines exactly those characters."""

    name: str  # 'doc_number' | 'birth_date' | 'expiry_date' | 'composite'
    start: int  # inclusive
    end: int  # exclusive, EXCLUDES the check digit itself
    check_digit_index: int
    valid: bool
    expected: Optional[str] = None  # populated only when invalid
    read: Optional[str] = None  # populated only when invalid
    signal_id: Optional[str] = None  # direct link to the Signal this group failed


class MrzLine(BaseModel):
    text: str  # exactly 30 / 36 / 44 chars
    groups: list[MrzGroup] = []


class MrzInfo(BaseModel):
    format: MrzFormat
    lines: list[MrzLine]
    status: MrzStatus


class ScreenedDocument(BaseModel):
    id: str
    type: DocType
    country: Optional[str] = None  # ISO-3166 alpha-3
    version: Optional[str] = None
    image_url: str  # rectified document image
    views: dict[ViewKey, str] = {}
    mrz: Optional[MrzInfo] = None
    fields: list[ExtractedField] = []
    risk: Optional[float] = None


class FaceResult(BaseModel):
    status: FaceStatus
    similarity: Optional[float] = None  # null when SPOOF or UNAVAILABLE
    threshold: float
    document_portrait_url: Optional[str] = None
    live_portrait_url: Optional[str] = None
    pad_verdict: PadVerdict
    ghost_portrait_consistent: Optional[bool] = None
    # Extra field beyond FRONTEND_BRIEF.md §3's FaceResult, same precedent as
    # ScreeningSession.model_versions above: the dev/test file-upload
    # fallback for the live face frame (LiveFaceCapture.tsx) must never be
    # silently reported as a live capture. Defaults to "live" so every
    # pre-existing caller (and the face branch stub) is unaffected.
    capture_method: CaptureMethod = "live"


class Encounter(BaseModel):
    session_id: str
    timestamp: str  # ISO
    checkpoint: str
    name_on_document: str
    document_number: str
    face_similarity: float
    conflict: bool  # same face, different identity


class GraphResult(BaseModel):
    prior_encounters: list[Encounter] = []
    conflicts: int = 0
    impossible_travel: bool = False


class OfficerDecision(BaseModel):
    decision: Decision
    note: str
    decided_at: str
    override: bool  # differed from the system band


class ScreeningSession(BaseModel):
    session_id: str
    lane_id: str
    officer_id: str
    started_at: str
    band: Optional[SystemBand] = None  # None while still processing
    risk: Optional[float] = None  # None when ABSTAIN or RECAPTURE
    confidence: Optional[float] = None
    abstained: bool = False
    recapture_reason: Optional[str] = None
    recapture_hint: Optional[str] = None
    documents: list[ScreenedDocument] = []
    signals: list[Signal] = []
    face: Optional[FaceResult] = None
    graph: Optional[GraphResult] = None
    cross_document_signals: list[Signal] = []
    coverage_flags: list[str] = []  # 'no_biometric' | 'stale_watchlist' | 'no_ovd' ...
    timing_ms: dict[str, float] = {}
    officer_decision: Optional[OfficerDecision] = None
    sealed: bool = False
    # BACKEND_BRIEF.md §8.2: audit records assume model_versions exists on
    # every session (sourced from the §1.5 model manifest). See module
    # docstring -- this closes the gap FRONTEND_BRIEF.md's screening.ts flags.
    model_versions: Optional[dict[str, str]] = None


__all__ = [
    "Decision",
    "SystemBand",
    "DocType",
    "FieldSource",
    "MrzFormat",
    "MrzStatus",
    "FaceStatus",
    "PadVerdict",
    "ViewKey",
    "ExtractedField",
    "MrzGroup",
    "MrzLine",
    "MrzInfo",
    "ScreenedDocument",
    "FaceResult",
    "Encounter",
    "GraphResult",
    "OfficerDecision",
    "ScreeningSession",
    "Region",
    "Signal",
]
