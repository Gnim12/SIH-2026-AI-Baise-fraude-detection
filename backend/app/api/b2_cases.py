"""Milestone B2: GET /api/cases (register, filtered/sorted/paginated
server-side) and GET /api/cases/{caseId} (full record)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage import b1_repositories as b1_repo
from app.storage import b2_repositories as repo
from app.storage.b1_models import B1Case
from app.storage.b2_models import B2Decision
from app.storage.db import get_session

from .b1_errors import ApiError
from .b2_decisions import _entry_to_wire
from .b2_wire import CaseListOut, CaseRecordOut, CaseSummaryOut

router = APIRouter(prefix="/api/cases", tags=["b2-cases"])


async def _case_summary(
    db: AsyncSession, case: B1Case, decision: Optional[B2Decision], coverage_complete: bool,
) -> CaseSummaryOut:
    entry_hash = None
    if decision is not None:
        entry = await repo.get_audit_entry(db, decision.audit_entry_id)
        entry_hash = entry.entry_hash if entry is not None else None
    return CaseSummaryOut(
        case_id=case.case_id, session_id=case.session_id, document_class=case.document_class,
        issuing_state=case.issuing_state, document_number_masked=case.document_number_masked,
        screened_at=case.screened_at, officer_id=decision.officer_id if decision else None,
        # No officer directory exists in this no-auth namespace (real auth
        # is a stated non-goal) -- officerName mirrors officerId rather than
        # being fabricated or omitted from a frontend type that requires it.
        officer_name=decision.officer_id if decision else None,
        recommendation=decision.recommendation if decision else None,
        officer_action=decision.action if decision else None,
        divergence=decision.divergence if decision else None,
        divergence_reason=decision.divergence_reason if decision else None,
        coverage_complete=coverage_complete,
        notes=decision.notes if decision else None,
        entry_hash=entry_hash,
    )


@router.get("")
async def list_cases_route(
    verdict: Optional[str] = None,
    document_class: Optional[str] = Query(default=None, alias="documentClass"),
    date_from: Optional[str] = Query(default=None, alias="from"),
    date_to: Optional[str] = Query(default=None, alias="to"),
    coverage: Optional[str] = None,
    divergence: Optional[str] = None,
    linked: Optional[str] = None,
    q: Optional[str] = None,
    page: int = 1,
    db: AsyncSession = Depends(get_session),
) -> CaseListOut:
    result = await repo.list_cases(
        db, verdict=verdict, document_class=document_class, date_from=date_from, date_to=date_to,
        coverage=coverage, divergence=divergence, linked=linked, query=q, page=page,
    )
    summaries = []
    for case, decision in result.rows:
        coverage_complete = await repo.get_run_coverage_complete(db, case.run_id)
        summaries.append(await _case_summary(db, case, decision, coverage_complete))

    return CaseListOut(
        cases=summaries, total=result.total, page=page, page_size=repo.PAGE_SIZE,
        # spec §3: "linked=only returns zero results ... with an explicit
        # indicator that the check is unavailable, not a silent empty" --
        # this flag is set whenever the caller asked for `linked=only`,
        # regardless of how many (zero) rows came back, so the frontend
        # never has to infer "unavailable" from an empty list alone.
        linked_unavailable=(linked == "only"),
    )


@router.get("/{case_id}")
async def get_case_record_route(case_id: str, db: AsyncSession = Depends(get_session)) -> CaseRecordOut:
    case_row = await repo.get_case_by_id(db, case_id)
    if case_row is None:
        raise ApiError(404, "case_not_found", f"No case found with id {case_id!r}.")
    case, decision = case_row

    coverage_complete = await repo.get_run_coverage_complete(db, case.run_id)
    run = await b1_repo.get_run(db, case.run_id)
    result_json = run.result_json if run is not None else None

    decision_entry = None
    if decision is not None:
        entry = await repo.get_audit_entry(db, decision.audit_entry_id)
        if entry is not None:
            decision_entry = _entry_to_wire(entry)

    return CaseRecordOut(
        case=await _case_summary(db, case, decision, coverage_complete), result=result_json,
        decision=decision_entry,
    )
