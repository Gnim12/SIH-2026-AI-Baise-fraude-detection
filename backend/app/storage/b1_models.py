"""Milestone B1b persistence: normalized tables for the new stage-registry
pipeline (app/pipeline/registry.py, analyse.py), on the same SQLAlchemy 2.0
async `Base` as the older BACKEND_BRIEF.md session table (app/storage/models.py)
-- see B1b's own decision note: SQLModel was the milestone brief's literal
wording, but this codebase already has one async ORM in production use, and
running two side by side buys nothing.

Table names are prefixed `b1_` deliberately: the older system already owns a
table named `sessions` (app/storage/models.py's SessionRow) with a completely
different shape (one JSONB payload column). Reusing `session`/`run` bare would
either collide or invite a reader to conflate the two systems.

`B1StageEvent` rows are append-only by convention -- no route or repository
function updates or deletes a row here. `sequence` is monotonic per run,
assigned by the repository layer (app/storage/b1_repositories.py) as it
persists each event alongside publishing it to the in-process RunBus, so a
process restart still has a full ordered log for replay (B2 assembles the
audit hash chain from these rows).

Document numbers: BACKEND_BRIEF.md's masking rule ("stored masked,
'•••• 7891', full value must never reach a table, a log line, or a response
body") is honoured throughout this file -- no column here holds a raw document
number, only `B1Case.document_number_masked`. Note this table's masking rule
is a genuinely new constraint that does not touch `app/api/schemas.py`'s
already-shipped, tested `MrzResult.fields` contract (the character-by-character
MRZ ribbon), which by design displays what the reader actually read so the
officer can verify it against the physical document -- that pre-existing
wire contract is out of this milestone's scope and is not masked. See the
B1b implementation report for this called out explicitly as a spec tension
rather than a silent choice.
"""
from __future__ import annotations

import datetime
from typing import Any, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def mask_document_number(raw: Optional[str]) -> Optional[str]:
    """`X1234567Y` -> `•••• 4567`. Never called with, and never returns, the
    full value in a place that persists or logs it."""
    if not raw:
        return None
    tail = raw[-4:] if len(raw) >= 4 else raw
    return f"•••• {tail}"


class B1Session(Base):
    __tablename__ = "b1_session"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    document_class: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # 'new' -> 'analysing' -> 'settled' (see app/storage/b1_repositories.py)
    status: Mapped[str] = mapped_column(String, default="new")


class B1Artefact(Base):
    __tablename__ = "b1_artefact"
    __table_args__ = (UniqueConstraint("session_id", "kind", name="uq_b1_artefact_session_kind"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("b1_session.id"), index=True)
    kind: Mapped[str] = mapped_column(String, index=True)
    filename: Mapped[str] = mapped_column(String)
    content_type: Mapped[str] = mapped_column(String)
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String)
    stored_path: Mapped[str] = mapped_column(String)
    captured_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    # ovd-sweep only; null for every other kind.
    duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    frame_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    angular_coverage_deg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class B1Run(Base):
    __tablename__ = "b1_run"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("b1_session.id"), index=True)
    started_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    verdict: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    coverage_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    # 'running' -> 'settled'. Distinct from `ended_at is not None` only in
    # that this is the field routes branch on; kept for readability.
    status: Mapped[str] = mapped_column(String, default="running")
    # The full ScreeningResult, camelCase-on-wire JSON, exactly as GET
    # /result returns it once settled -- avoids reconstructing the wire
    # object from normalized rows and risking a field-name drift from
    # app/api/schemas.py. The normalized signal/finding rows below exist for
    # B2's audit-chain assembly and future case history queries, not to
    # rebuild this response.
    result_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)


class B1StageEvent(Base):
    __tablename__ = "b1_stage_event"
    __table_args__ = (UniqueConstraint("run_id", "sequence", name="uq_b1_stage_event_run_seq"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("b1_run.id"), index=True)
    stage_id: Mapped[str] = mapped_column(String)
    state: Mapped[str] = mapped_column(String)
    detail: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    occurred_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    sequence: Mapped[int] = mapped_column(Integer)


class B1Signal(Base):
    __tablename__ = "b1_signal"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("b1_run.id"), index=True)
    stage_id: Mapped[str] = mapped_column(String)
    signal_id: Mapped[str] = mapped_column(String)
    modality: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)
    detail: Mapped[str] = mapped_column(String)
    region_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    model_pin: Mapped[str] = mapped_column(String)
    emitted_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))


class B1Finding(Base):
    __tablename__ = "b1_finding"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("b1_run.id"), index=True)
    finding_id: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(String)
    hypothesis: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String)
    region_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)


class B1FindingSignal(Base):
    __tablename__ = "b1_finding_signal"

    finding_id: Mapped[str] = mapped_column(String, ForeignKey("b1_finding.id"), primary_key=True)
    signal_id: Mapped[str] = mapped_column(String, ForeignKey("b1_signal.id"), primary_key=True)


class B1Case(Base):
    __tablename__ = "b1_case"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("b1_session.id"), index=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("b1_run.id"))
    case_id: Mapped[str] = mapped_column(String, unique=True)
    screened_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    document_class: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    issuing_state: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    document_number_masked: Mapped[Optional[str]] = mapped_column(String, nullable=True)
