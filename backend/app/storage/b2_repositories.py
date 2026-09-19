"""DB access for milestone B2: decision submission, case register, config
audit. Every write that touches `B2AuditEntry` goes through
`app/audit/chain.append()` inside the same transaction as its domain row
(`B2Decision`/`B2ConfigChange`) -- see `submit_decision`'s docstring for why
that ordering is load-bearing.
"""
from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit import chain
from app.storage import b1_repositories as b1_repo
from app.storage.b1_models import B1Case, B1Run, B1Session

from .b2_models import B2AuditEntry, B2ConfigChange, B2Decision, B2ThresholdValue

# frontend/src/lib/config/thresholdSeed.ts's THRESHOLD_SEED, mirrored
# server-side -- PUT /api/config/thresholds validates against these bounds
# rather than trusting the client's own copy of them.
THRESHOLD_DEFS: list[dict[str, Any]] = [
    {
        "id": "mrz-viz-field-tolerance", "label": "MRZ ↔ VIZ field tolerance", "gate": "gate-1",
        "unit": "ratio", "min": 0.7, "max": 1.0, "step": 0.01, "default": 0.92,
        "description": "How closely machine-readable and visual-zone fields must agree to pass Gate 1.",
    },
    {
        "id": "ovd-angular-coverage-min", "label": "Minimum sweep angular coverage", "gate": "capture",
        "unit": "degrees", "min": 10, "max": 60, "step": 1, "default": 25,
        "description": "The minimum angle a traveller must sweep the document through during OVD capture.",
    },
    {
        "id": "ovd-variance-min", "label": "Minimum OVD frame variance", "gate": "gate-2",
        "unit": "ratio", "min": 0.1, "max": 0.8, "step": 0.01, "default": 0.35,
        "description": "How much the OVD response must shift across the sweep to count as physically real.",
    },
    {
        "id": "face-match-min", "label": "Face match minimum similarity", "gate": "gate-2",
        "unit": "ratio", "min": 0.4, "max": 0.95, "step": 0.01, "default": 0.68,
        "description": "The minimum ArcFace cosine similarity between the document photo and the live capture.",
    },
    {
        "id": "tamper-confidence-min", "label": "Tamper detection confidence", "gate": "gate-2",
        "unit": "ratio", "min": 0.3, "max": 0.9, "step": 0.01, "default": 0.6,
        "description": "The confidence the tamper model must reach before a finding is raised.",
    },
    {
        "id": "graph-match-min", "label": "Identity graph match similarity", "gate": None,
        "unit": "ratio", "min": 0.5, "max": 0.95, "step": 0.01, "default": 0.75,
        "description": "The minimum face-embedding similarity for the identity graph to link two enrolments.",
    },
]
THRESHOLD_DEF_BY_ID = {d["id"]: d for d in THRESHOLD_DEFS}

# BACKEND_BRIEF.md §1.5-adjacent: model pins are read-only over HTTP (spec
# §4) -- there is deliberately no endpoint anywhere that writes this dict.
MODEL_PINS: dict[str, str] = {
    "mrz-crnn-slot": "1.2.0", "rapidocr": "1.3.1", "resnet": "0.7.2", "arcface": "1.1.0",
}


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


# --- Decisions ------------------------------------------------------------


class DecisionValidationError(Exception):
    """First unmet requirement, named -- see app/api/b2_decisions.py's raise
    site, which turns this into an ApiError(422, ...)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DecisionConflictError(Exception):
    """A decision already exists for this session -- 409, decisions are not
    editable."""


def compute_divergence(recommendation: str, action: str) -> str:
    """Server-side port of frontend/src/lib/decision/divergence.ts's
    computeDivergence -- same three named pairs, same 'none' for everything
    else. Kept in lockstep deliberately: same shape as
    BACKEND_BRIEF.md's own mirror-the-frontend-contract convention."""
    if recommendation == "flagged" and action == "clear":
        return "officer-cleared-flagged"
    if recommendation == "cleared" and action == "reject":
        return "officer-rejected-cleared"
    if recommendation == "review-required" and action == "reject":
        return "officer-escalated"
    return "none"


MIN_DISMISS_REASON_LENGTH = 10
MIN_DIVERGENCE_REASON_LENGTH = 20
VALID_ACTIONS = ("clear", "refer-secondary", "reject", "request-recapture")
NOTES_REQUIRED_ACTIONS = ("reject", "refer-secondary")


async def get_decision(db: AsyncSession, session_id: str) -> Optional[B2Decision]:
    return await db.get(B2Decision, session_id)


@dataclass
class SubmitDecisionResult:
    case_id: str
    session_id: str
    entry_hash: str
    entry_id: str


async def submit_decision(
    db: AsyncSession,
    *,
    session_id: str,
    officer_id: str,
    terminal_id: str,
    action: str,
    finding_dispositions: list[dict[str, Any]],
    all_finding_ids: list[str],
    all_signal_ids: list[str],
    evidence_hashes: list[dict[str, str]],
    recommendation: str,
    run_id: str,
    thresholds: dict[str, float],
    notes: str,
    divergence_reason: Optional[str],
    attestation: bool,
) -> SubmitDecisionResult:
    """Validates server-side (never trusting the client, spec §2), computes
    divergence, and -- only if every requirement is met -- inserts the
    `B2Decision` row and the `B2AuditEntry` it seals into, in the same
    transaction: `chain.append()` only `flush()`es, it never commits, so
    either both rows land in the single `db.commit()` at the end of this
    function or (on any exception before that point, including the
    `session.refresh` after) neither does. A decision without its audit
    record is exactly the failure mode this function exists to prevent.

    Raises `DecisionConflictError` if a decision already exists for this
    session (a `UniqueConstraint`/primary-key violation on insert would also
    catch a race between two concurrent submissions for the same session,
    but the explicit `get_decision` check up front is what produces the
    clean 409 in the common, non-racing case)."""
    existing = await get_decision(db, session_id)
    if existing is not None:
        raise DecisionConflictError()

    disposition_ids = {d["findingId"] for d in finding_dispositions}
    missing = [fid for fid in all_finding_ids if fid not in disposition_ids]
    if missing:
        raise DecisionValidationError(f"Finding {missing[0]} has not been given a disposition.")

    for d in finding_dispositions:
        if d["disposition"] == "dismissed":
            reason = (d.get("reason") or "").strip()
            if len(reason) < MIN_DISMISS_REASON_LENGTH:
                raise DecisionValidationError(
                    f"Dismissing finding {d['findingId']} requires a reason of at least "
                    f"{MIN_DISMISS_REASON_LENGTH} characters."
                )

    if action not in VALID_ACTIONS:
        raise DecisionValidationError(f"action must be one of {', '.join(VALID_ACTIONS)}.")

    divergence = compute_divergence(recommendation, action)
    if divergence != "none":
        reason = (divergence_reason or "").strip()
        if len(reason) < MIN_DIVERGENCE_REASON_LENGTH:
            raise DecisionValidationError(
                f"This decision diverges from the system recommendation and requires a reason of at least "
                f"{MIN_DIVERGENCE_REASON_LENGTH} characters."
            )

    if action in NOTES_REQUIRED_ACTIONS and len((notes or "").strip()) == 0:
        raise DecisionValidationError(f"Officer notes are required for {action}.")

    if not attestation:
        raise DecisionValidationError("Attestation is required to seal the record.")

    # `settle_run` (app/storage/b1_repositories.py) already created a
    # `B1Case` row the moment this run settled -- reuse its case_id rather
    # than minting a second one. Scoped by run_id (not session_id): a
    # session can have more than one settled run (re-analysis), each with
    # its own case row, and this decision is sealed against one specific run.
    existing_case = await b1_repo.get_case_by_run_id(db, run_id)
    case_id = existing_case.case_id if existing_case is not None else f"CASE-{uuid.uuid4().hex[:8].upper()}"
    decided_at = _now()

    seal = await chain.append(
        db, entry_type="decision", terminal_id=terminal_id, officer_id=officer_id,
        session_id=session_id, run_id=run_id, evidence_hashes=evidence_hashes,
        signal_ids=all_signal_ids, finding_dispositions=finding_dispositions,
        model_pins=MODEL_PINS, thresholds=thresholds, recommendation=recommendation,
        officer_action=action, divergence=divergence,
        divergence_reason=divergence_reason if divergence != "none" else None, notes=notes,
    )

    db.add(B2Decision(
        session_id=session_id, case_id=case_id, run_id=run_id, audit_entry_id=seal.entry_id,
        officer_id=officer_id, action=action, recommendation=recommendation, divergence=divergence,
        divergence_reason=divergence_reason if divergence != "none" else None, notes=notes,
        attestation=attestation, decided_at=decided_at, finding_dispositions_json=finding_dispositions,
    ))

    await db.commit()
    return SubmitDecisionResult(case_id=case_id, session_id=session_id, entry_hash=seal.entry_hash,
                                 entry_id=seal.entry_id)


async def get_audit_entry(db: AsyncSession, entry_id: str) -> Optional[B2AuditEntry]:
    return await db.get(B2AuditEntry, entry_id)


async def get_audit_entries_for_session(db: AsyncSession, session_id: str) -> list[B2AuditEntry]:
    result = await db.execute(
        select(B2AuditEntry).where(B2AuditEntry.session_id == session_id).order_by(B2AuditEntry.sequence.asc())
    )
    return list(result.scalars().all())


async def get_decision_by_case_id(db: AsyncSession, case_id: str) -> Optional[B2Decision]:
    result = await db.execute(select(B2Decision).where(B2Decision.case_id == case_id))
    return result.scalar_one_or_none()


# --- Case register ---------------------------------------------------------


@dataclass
class CaseListResult:
    rows: list[tuple[B1Case, Optional[B2Decision]]]
    total: int


PAGE_SIZE = 20


async def list_cases(
    db: AsyncSession,
    *,
    verdict: Optional[str] = None,
    document_class: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    coverage: Optional[str] = None,  # 'partial' | None
    divergence: Optional[str] = None,  # 'only' | None
    linked: Optional[str] = None,  # 'only' | None -- always empty, see spec §3
    query: Optional[str] = None,
    page: int = 1,
) -> CaseListResult:
    if linked == "only":
        return CaseListResult(rows=[], total=0)

    stmt = select(B1Case, B2Decision).outerjoin(B2Decision, B2Decision.case_id == B1Case.case_id)

    if verdict is not None:
        stmt = stmt.where(B2Decision.recommendation == verdict)
    if document_class is not None:
        stmt = stmt.where(B1Case.document_class == document_class)
    if date_from is not None:
        stmt = stmt.where(B1Case.screened_at >= datetime.datetime.fromisoformat(date_from))
    if date_to is not None:
        stmt = stmt.where(B1Case.screened_at <= datetime.datetime.fromisoformat(date_to))
    if coverage == "partial":
        stmt = stmt.join(B1Run, B1Run.id == B1Case.run_id).where(B1Run.coverage_complete.is_(False))
    if divergence == "only":
        stmt = stmt.where(B2Decision.divergence.isnot(None), B2Decision.divergence != "none")
    if query:
        like = f"%{query}%"
        stmt = stmt.where((B1Case.case_id.ilike(like)) | (B1Case.document_number_masked.ilike(like)))

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar_one()

    stmt = stmt.order_by(B1Case.screened_at.desc()).offset((page - 1) * PAGE_SIZE).limit(PAGE_SIZE)
    rows = list((await db.execute(stmt)).all())
    return CaseListResult(rows=[(r[0], r[1]) for r in rows], total=total)


async def get_case_by_id(db: AsyncSession, case_id: str) -> Optional[tuple[B1Case, Optional[B2Decision]]]:
    result = await db.execute(
        select(B1Case, B2Decision).outerjoin(B2Decision, B2Decision.case_id == B1Case.case_id)
        .where(B1Case.case_id == case_id)
    )
    row = result.first()
    return (row[0], row[1]) if row is not None else None


async def get_run_coverage_complete(db: AsyncSession, run_id: str) -> bool:
    run = await db.get(B1Run, run_id)
    return bool(run.coverage_complete) if run is not None else False


async def get_session_row(db: AsyncSession, session_id: str) -> Optional[B1Session]:
    return await db.get(B1Session, session_id)


# --- Config / thresholds ---------------------------------------------------


async def get_threshold_values(db: AsyncSession) -> dict[str, float]:
    """Seeds any missing threshold to its default on first read -- this
    table only ever holds *current* values (see B2ThresholdValue's
    docstring)."""
    result = await db.execute(select(B2ThresholdValue))
    existing = {row.id: row.value for row in result.scalars().all()}
    missing = [d for d in THRESHOLD_DEFS if d["id"] not in existing]
    for d in missing:
        db.add(B2ThresholdValue(id=d["id"], value=float(d["default"])))
        existing[d["id"]] = float(d["default"])
    if missing:
        await db.commit()
    return existing


class ThresholdValidationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


async def update_thresholds(
    db: AsyncSession, *, officer_id: str, terminal_id: str, changes: list[dict[str, Any]], reason: str,
) -> tuple[str, str]:
    """Validates every change against its threshold's [min, max] bound
    before writing anything, then applies the whole batch and seals it as a
    single audit entry (spec §4: "a batch seals as one entry containing all
    changes"). Returns (entry_id, entry_hash)."""
    if len((reason or "").strip()) < MIN_DIVERGENCE_REASON_LENGTH:
        raise ThresholdValidationError(
            f"A threshold change requires a reason of at least {MIN_DIVERGENCE_REASON_LENGTH} characters."
        )

    current = await get_threshold_values(db)
    change_rows: list[dict[str, Any]] = []
    for change in changes:
        threshold_id = change["id"]
        definition = THRESHOLD_DEF_BY_ID.get(threshold_id)
        if definition is None:
            raise ThresholdValidationError(f"Unknown threshold {threshold_id!r}.")
        new_value = float(change["value"])
        if not (definition["min"] <= new_value <= definition["max"]):
            raise ThresholdValidationError(
                f"{threshold_id} must be between {definition['min']} and {definition['max']} "
                f"(got {new_value})."
            )
        change_rows.append({
            "id": threshold_id, "previousValue": current.get(threshold_id, definition["default"]),
            "newValue": new_value,
        })

    seal = await chain.append(
        db, entry_type="config_change", terminal_id=terminal_id, officer_id=officer_id,
        session_id=None, run_id=None, evidence_hashes=[], signal_ids=[], finding_dispositions=[],
        model_pins=MODEL_PINS, thresholds={c["id"]: c["newValue"] for c in change_rows},
        recommendation=None, officer_action=None, divergence=None, divergence_reason=None,
        notes=reason,
    )

    for change in changes:
        row = await db.get(B2ThresholdValue, change["id"])
        if row is None:
            db.add(B2ThresholdValue(id=change["id"], value=float(change["value"])))
        else:
            # A `B2ThresholdValue` row IS mutated in place -- this is the
            # current-value cache, not the audit trail (spec's append-only
            # requirement is scoped to audit/case tables, §7's acceptance
            # criterion). History of the change lives immutably in
            # `B2ConfigChange`/`B2AuditEntry` above, which is what
            # GET /api/config/audit reads from.
            row.value = float(change["value"])

    db.add(B2ConfigChange(
        id=f"CFG-{uuid.uuid4().hex[:8].upper()}", audit_entry_id=seal.entry_id, officer_id=officer_id,
        changed_at=_now(), reason=reason, changes_json=change_rows,
    ))

    await db.commit()
    return seal.entry_id, seal.entry_hash


async def list_config_changes(db: AsyncSession) -> list[B2ConfigChange]:
    result = await db.execute(select(B2ConfigChange).order_by(B2ConfigChange.changed_at.asc()))
    return list(result.scalars().all())
