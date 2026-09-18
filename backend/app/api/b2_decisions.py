"""Milestone B2: POST /api/sessions/{id}/decision, GET /api/cases/{caseId}/audit,
GET /api/audit/verify. Same no-/v1-prefix, no-auth namespace as app/api/b1_*.py
(officerId/terminalId travel in the request body instead, mirroring
frontend/src/lib/decision/officerContext.ts's hardcoded stand-ins -- real
auth is a stated non-goal)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import ScreeningResult
from app.audit import chain
from app.storage import b1_repositories as b1_repo
from app.storage import b2_repositories as repo
from app.storage.b2_models import B2AuditEntry
from app.storage.db import get_session

from .b1_errors import ApiError
from .b2_wire import (
    AuditEntryOut, ChainHeadOut, ChainVerifyOut, DecisionIn, DecisionOut, EvidenceHashIn, FindingDispositionIn,
)

router = APIRouter(prefix="/api", tags=["b2-decisions"])


async def _get_settled_result(db: AsyncSession, session_id: str) -> tuple[str, ScreeningResult]:
    run = await b1_repo.get_latest_run(db, session_id)
    if run is None or run.status != "settled" or run.result_json is None:
        raise ApiError(
            409, "no_settled_run",
            "A screening must complete before a decision can be recorded for this session.",
        )
    return run.id, ScreeningResult.model_validate(run.result_json)


@router.post("/sessions/{session_id}/decision")
async def submit_decision_route(
    session_id: str, body: DecisionIn, db: AsyncSession = Depends(get_session),
) -> DecisionOut:
    session = await b1_repo.get_session(db, session_id)
    if session is None:
        raise ApiError(404, "session_not_found", f"No session found with id {session_id!r}.")

    run_id, result = await _get_settled_result(db, session_id)

    all_finding_ids = [f.finding_id for f in result.findings]
    all_signal_ids = [s.signal_id for f in result.findings for s in f.signals]

    artefacts = await b1_repo.list_artefacts(db, session_id)
    evidence_hashes = [{"kind": a.kind, "sha256": f"sha256:{a.sha256}"} for a in artefacts]

    thresholds = await repo.get_threshold_values(db)

    finding_dispositions = [
        {"findingId": d.finding_id, "disposition": d.disposition, "reason": d.reason}
        for d in body.finding_dispositions
    ]

    try:
        outcome = await repo.submit_decision(
            db, session_id=session_id, officer_id=body.officer_id, terminal_id=body.terminal_id,
            action=body.action, finding_dispositions=finding_dispositions, all_finding_ids=all_finding_ids,
            all_signal_ids=all_signal_ids, evidence_hashes=evidence_hashes,
            recommendation=result.verdict.value, run_id=run_id, thresholds=thresholds, notes=body.notes,
            divergence_reason=body.divergence_reason, attestation=body.attestation,
        )
    except repo.DecisionConflictError:
        raise ApiError(
            409, "decision_already_recorded",
            "A decision has already been recorded for this session. Decisions cannot be edited.",
        ) from None
    except repo.DecisionValidationError as exc:
        raise ApiError(422, "invalid_decision", exc.message) from None

    return DecisionOut(
        case_id=outcome.case_id, session_id=outcome.session_id, entry_hash=outcome.entry_hash,
        entry_id=outcome.entry_id,
    )


def _entry_to_wire(entry_row: B2AuditEntry) -> AuditEntryOut:
    payload = entry_row.payload_json
    finding_dispositions = [FindingDispositionIn.model_validate(d) for d in payload["findingDispositions"]]
    return AuditEntryOut(
        entry_id=entry_row.id,
        session_id=payload["sessionId"],
        run_id=payload["runId"],
        previous_hash=payload["previousHash"],
        entry_hash=entry_row.entry_hash,
        sealed_at=payload["sealedAt"],
        officer_id=payload["officerId"],
        terminal_id=payload["terminalId"],
        evidence_hashes=[EvidenceHashIn.model_validate(e) for e in payload["evidenceHashes"]],
        signal_ids=payload["signalIds"],
        finding_ids=[d.finding_id for d in finding_dispositions],
        finding_dispositions=finding_dispositions,
        model_pins=payload["modelPins"],
        thresholds=payload["thresholds"],
        recommendation=payload["recommendation"],
        officer_action=payload["officerAction"],
        divergence=payload["divergence"],
        divergence_reason=payload["divergenceReason"],
        notes=payload["notes"],
        entry_type=entry_row.entry_type,
    )


@router.get("/cases/{case_id}/audit")
async def get_case_audit_route(case_id: str, db: AsyncSession = Depends(get_session)) -> list[AuditEntryOut]:
    case_row = await repo.get_case_by_id(db, case_id)
    if case_row is None:
        raise ApiError(404, "case_not_found", f"No case found with id {case_id!r}.")
    case, _decision = case_row
    entries = await repo.get_audit_entries_for_session(db, case.session_id)
    return [_entry_to_wire(e) for e in entries]


@router.get("/audit/verify")
async def verify_audit_route(
    terminal_id: Optional[str] = Query(default=None, alias="terminalId"),
    db: AsyncSession = Depends(get_session),
) -> ChainVerifyOut:
    result = await chain.verify(db, terminal_id=terminal_id)
    return ChainVerifyOut(
        intact=result.intact, broken_entry_id=result.broken_entry_id, reason=result.reason,
        entries_checked=result.entries_checked,
    )


@router.get("/audit/chain-head")
async def get_chain_head_route(
    terminal_id: str = Query(alias="terminalId"),
    db: AsyncSession = Depends(get_session),
) -> ChainHeadOut:
    """Backs the Decision screen's pre-seal audit preview (spec §5): the real
    previousHash a new entry on this terminal would reference right now,
    read without appending anything."""
    previous_hash = await chain.peek_latest_hash(db, terminal_id)
    return ChainHeadOut(terminal_id=terminal_id, previous_hash=previous_hash)
