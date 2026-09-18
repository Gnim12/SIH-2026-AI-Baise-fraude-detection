"""Milestone B2: decision submission, the case register, and the config
audit trail -- on the same SQLAlchemy 2.0 async `Base` as app/storage/
b1_models.py, table names prefixed `b2_` for the same reason B1's are
prefixed `b1_` (a distinct namespace from both the older BACKEND_BRIEF.md
`sessions` table and B1's own tables).

`B2AuditEntry` rows are append-only by convention, exactly like
`B1StageEvent` -- no route or repository function in this codebase updates
or deletes one. `app/audit/chain.py` is the only writer. `sequence` is
monotonic *per terminal_id* (one chain per terminal, not per case -- see
that module's docstring), assigned inside `chain.append()`'s per-terminal
lock.

`B2Decision.session_id` is the primary key, not just a foreign key: this is
the DB-level enforcement (on top of the app-level check in
app/api/b2_decisions.py) that a session can never have two decisions.
`B2Decision` rows are also never updated or deleted once inserted --
"a wrong decision is corrected by a new entry, never by editing the old
one" applies to the decision row too, which is why there is no per-session
uniqueness *violation recovery* path anywhere in this codebase; a 409 is the
only outcome of a second attempt.
"""
from __future__ import annotations

import datetime
from typing import Any, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class B2AuditEntry(Base):
    __tablename__ = "b2_audit_entry"
    __table_args__ = (UniqueConstraint("terminal_id", "sequence", name="uq_b2_audit_entry_terminal_seq"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)  # entryId, e.g. AUD-...
    terminal_id: Mapped[str] = mapped_column(String, index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    entry_type: Mapped[str] = mapped_column(String)  # 'decision' | 'config_change'
    session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    previous_hash: Mapped[str] = mapped_column(String)
    entry_hash: Mapped[str] = mapped_column(String, unique=True, index=True)
    sealed_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    officer_id: Mapped[str] = mapped_column(String)
    # The exact dict `chain.canonical_json` hashed -- re-hashed verbatim by
    # `chain.verify()`. Storing the payload rather than reconstructing it
    # from other tables is what makes tamper detection meaningful: a stored
    # field that's derived at read time can't be "altered since it was
    # sealed" because it's recomputed every time.
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON)


class B2Decision(Base):
    __tablename__ = "b2_decision"

    session_id: Mapped[str] = mapped_column(String, ForeignKey("b1_session.id"), primary_key=True)
    case_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("b1_run.id"))
    audit_entry_id: Mapped[str] = mapped_column(String, ForeignKey("b2_audit_entry.id"))
    officer_id: Mapped[str] = mapped_column(String)
    action: Mapped[str] = mapped_column(String)  # OfficerAction
    recommendation: Mapped[str] = mapped_column(String)  # Verdict at decision time
    divergence: Mapped[str] = mapped_column(String)  # server-computed, never client-supplied
    divergence_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    notes: Mapped[str] = mapped_column(String)
    attestation: Mapped[bool] = mapped_column(Boolean)
    decided_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    # [{findingId, disposition, reason}] -- the dispositions payload verbatim
    finding_dispositions_json: Mapped[list[Any]] = mapped_column(JSON)


class B2ConfigChange(Base):
    __tablename__ = "b2_config_change"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    audit_entry_id: Mapped[str] = mapped_column(String, ForeignKey("b2_audit_entry.id"))
    officer_id: Mapped[str] = mapped_column(String)
    changed_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    reason: Mapped[str] = mapped_column(String)
    # [{id, previousValue, newValue}] for every threshold in the batch --
    # "a batch seals as one entry containing all changes" (spec §4).
    changes_json: Mapped[list[Any]] = mapped_column(JSON)


class B2ThresholdValue(Base):
    """Current value of each of the six officer-tunable thresholds
    (frontend/src/lib/config/thresholdSeed.ts's THRESHOLD_SEED). Seeded to
    each threshold's `default` on first read if the row doesn't exist yet
    (app/storage/b2_repositories.py's `get_or_seed_threshold_values`) --
    this table only ever holds *current* values; history lives in
    `B2ConfigChange`/`B2AuditEntry`, never here."""

    __tablename__ = "b2_threshold_value"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # ThresholdId
    value: Mapped[float] = mapped_column(Float)
