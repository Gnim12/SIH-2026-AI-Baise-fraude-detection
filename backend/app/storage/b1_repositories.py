"""DB + on-disk access for the B1b HTTP surface. Every stage-event write goes
through `append_stage_event`, which assigns the monotonic-per-run `sequence`
column -- the only writer of `B1StageEvent` rows in the codebase, and it never
updates or deletes one (see app/storage/b1_models.py's docstring).
"""
from __future__ import annotations

import datetime
import hashlib
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import ScreeningResult
from app.config import settings

from .b1_models import (
    B1Artefact, B1Case, B1Finding, B1FindingSignal, B1Run, B1Session, B1Signal, B1StageEvent,
    mask_document_number,
)

_EXT_BY_CONTENT_TYPE = {
    "image/jpeg": ".jpg", "image/png": ".png", "application/pdf": ".pdf",
    "video/mp4": ".mp4", "video/webm": ".webm",
}


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _artefact_dir(session_id: str) -> Path:
    d = settings.b1_artefact_root / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _artefact_path(session_id: str, kind: str, content_type: str) -> Path:
    ext = _EXT_BY_CONTENT_TYPE.get(content_type, mimetypes.guess_extension(content_type) or "")
    return _artefact_dir(session_id) / f"{kind}{ext}"


async def create_session(db: AsyncSession) -> B1Session:
    row = B1Session(id=str(uuid.uuid4()), created_at=_now(), document_class=None, status="new")
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def get_session(db: AsyncSession, session_id: str) -> Optional[B1Session]:
    return await db.get(B1Session, session_id)


async def list_artefacts(db: AsyncSession, session_id: str) -> list[B1Artefact]:
    result = await db.execute(select(B1Artefact).where(B1Artefact.session_id == session_id))
    return list(result.scalars().all())


async def get_artefact(db: AsyncSession, session_id: str, kind: str) -> Optional[B1Artefact]:
    result = await db.execute(
        select(B1Artefact).where(B1Artefact.session_id == session_id, B1Artefact.kind == kind)
    )
    return result.scalar_one_or_none()


async def _delete_artefact_row_and_bytes(db: AsyncSession, artefact: B1Artefact) -> None:
    path = Path(artefact.stored_path)
    if path.exists():
        path.unlink()
    await db.delete(artefact)


async def save_artefact(
    db: AsyncSession,
    session_id: str,
    kind: str,
    filename: str,
    content_type: str,
    data: bytes,
    *,
    duration_ms: Optional[float] = None,
    frame_count: Optional[int] = None,
    angular_coverage_deg: Optional[float] = None,
) -> tuple[B1Artefact, bool]:
    """Write bytes to disk, upsert the artefact row, and enforce the sweep
    invariant server-side: attaching a new document-still deletes any
    existing ovd-sweep for the same session (its bytes too). Returns
    (artefact, sweep_was_deleted)."""
    existing = await get_artefact(db, session_id, kind)
    if existing is not None:
        await _delete_artefact_row_and_bytes(db, existing)
        await db.flush()

    path = _artefact_path(session_id, kind, content_type)
    path.write_bytes(data)
    sha256 = hashlib.sha256(data).hexdigest()

    row = B1Artefact(
        id=str(uuid.uuid4()), session_id=session_id, kind=kind, filename=filename,
        content_type=content_type, size_bytes=len(data), sha256=sha256, stored_path=str(path),
        captured_at=_now(), duration_ms=duration_ms, frame_count=frame_count,
        angular_coverage_deg=angular_coverage_deg,
    )
    db.add(row)

    sweep_deleted = False
    if kind == "document-still":
        sweep = await get_artefact(db, session_id, "ovd-sweep")
        if sweep is not None:
            await _delete_artefact_row_and_bytes(db, sweep)
            sweep_deleted = True

    await db.commit()
    await db.refresh(row)
    return row, sweep_deleted


async def delete_artefact(db: AsyncSession, session_id: str, kind: str) -> tuple[Optional[B1Artefact], bool]:
    """Returns (deleted_artefact_or_None, sweep_was_cascaded)."""
    existing = await get_artefact(db, session_id, kind)
    if existing is None:
        return None, False

    await _delete_artefact_row_and_bytes(db, existing)

    sweep_deleted = False
    if kind == "document-still":
        sweep = await get_artefact(db, session_id, "ovd-sweep")
        if sweep is not None:
            await _delete_artefact_row_and_bytes(db, sweep)
            sweep_deleted = True

    await db.commit()
    return existing, sweep_deleted


async def get_case_by_run_id(db: AsyncSession, run_id: str) -> Optional[B1Case]:
    """The `B1Case` row `settle_run` creates automatically for every settled
    run -- B2's decision submission (app/storage/b2_repositories.py) reuses
    this row's `case_id` rather than minting a second one. Scoped by run_id,
    not session_id: a session may be analysed more than once (B1e's re-run,
    or simply re-triggering analysis), and `settle_run` inserts a fresh
    `B1Case` row for every settled run, so a session can genuinely have
    several case rows. The decision being sealed is always for one specific
    run, so that is what must key this lookup -- looking it up by
    session_id alone raised `MultipleResultsFound` the moment a session had
    more than one settled run, which a real capture->analyse->re-analyse
    flow reaches easily."""
    result = await db.execute(select(B1Case).where(B1Case.run_id == run_id))
    return result.scalar_one_or_none()


async def get_in_flight_run(db: AsyncSession, session_id: str) -> Optional[B1Run]:
    result = await db.execute(
        select(B1Run).where(B1Run.session_id == session_id, B1Run.status == "running")
        .order_by(B1Run.started_at.desc())
    )
    return result.scalars().first()


async def get_latest_run(db: AsyncSession, session_id: str) -> Optional[B1Run]:
    result = await db.execute(
        select(B1Run).where(B1Run.session_id == session_id).order_by(B1Run.started_at.desc())
    )
    return result.scalars().first()


async def get_run(db: AsyncSession, run_id: str) -> Optional[B1Run]:
    return await db.get(B1Run, run_id)


async def create_run(db: AsyncSession, session_id: str) -> B1Run:
    row = B1Run(id=str(uuid.uuid4()), session_id=session_id, started_at=_now(), status="running")
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def append_stage_event(
    db: AsyncSession, run_id: str, stage_id: str, state: str, detail: Optional[str], occurred_at: str,
) -> int:
    """Assigns and returns the next monotonic sequence number for this run."""
    max_seq = await db.scalar(select(func.max(B1StageEvent.sequence)).where(B1StageEvent.run_id == run_id))
    sequence = (max_seq or 0) + 1
    row = B1StageEvent(
        id=str(uuid.uuid4()), run_id=run_id, stage_id=stage_id, state=state, detail=detail,
        occurred_at=datetime.datetime.fromisoformat(occurred_at), sequence=sequence,
    )
    db.add(row)
    await db.commit()
    return sequence


async def list_stage_events(db: AsyncSession, run_id: str) -> list[B1StageEvent]:
    result = await db.execute(
        select(B1StageEvent).where(B1StageEvent.run_id == run_id).order_by(B1StageEvent.sequence.asc())
    )
    return list(result.scalars().all())


def _region_json(region: Any) -> Optional[dict[str, Any]]:
    return region.model_dump(mode="json") if region is not None else None


async def settle_run(db: AsyncSession, run_id: str, session_id: str, result: ScreeningResult) -> None:
    """Marks the run settled, stores the full wire result, and populates the
    normalized signal/finding/finding_signal/case rows B2's audit chain and
    future case-history queries read from."""
    run = await db.get(B1Run, run_id)
    if run is None:
        return

    # B2 fix: the original formula (`ran or blocked_by is not None or
    # unavailable`) is a tautology -- every StageCoverage entry is always
    # exactly one of those three, so it was always True and useless as a
    # "did everything that could run actually run" signal. What B2's case
    # register needs (frontend/src/lib/cases/types.ts's own comment:
    # "blockedBy? 'gate-1' when coverage incomplete") is specifically
    # whether a *gate* blocked a stage that could otherwise have run --
    # `unavailable` (every Wave 2 stage, always, in this build) is a
    # separate, always-true-in-this-build condition that would make this
    # field meaningless as a case-register filter if counted the same way.
    coverage_complete = not any(c.blocked_by is not None for c in result.coverage)

    run.ended_at = _now()
    run.verdict = result.verdict.value
    run.coverage_complete = coverage_complete
    run.status = "settled"
    run.result_json = result.model_dump(mode="json", by_alias=True)

    signal_row_id_by_signal_id: dict[str, str] = {}
    for finding in result.findings:
        for signal in finding.signals:
            if signal.signal_id in signal_row_id_by_signal_id:
                continue
            row_id = str(uuid.uuid4())
            signal_row_id_by_signal_id[signal.signal_id] = row_id
            db.add(B1Signal(
                id=row_id, run_id=run_id, stage_id="", signal_id=signal.signal_id,
                modality=signal.modality.value, label=signal.label, detail=signal.detail,
                region_json=_region_json(signal.region), model_pin=signal.model_pin,
                emitted_at=datetime.datetime.fromisoformat(signal.emitted_at),
            ))

    finding_row_ids = []
    for finding in result.findings:
        finding_row_id = str(uuid.uuid4())
        finding_row_ids.append(finding_row_id)
        db.add(B1Finding(
            id=finding_row_id, run_id=run_id, finding_id=finding.finding_id, title=finding.title,
            hypothesis=finding.hypothesis, severity=finding.severity.value,
            region_json=_region_json(finding.region),
        ))

    # B1Signal and B1Finding rows must be persisted before B1FindingSignal
    # references them -- without this flush, SQLAlchemy's autoflush ordering
    # does not reliably insert them first, and the join-row insert fails a
    # foreign-key check against rows added earlier in this same session.
    await db.flush()

    for finding, finding_row_id in zip(result.findings, finding_row_ids):
        for signal in finding.signals:
            signal_row_id = signal_row_id_by_signal_id.get(signal.signal_id)
            if signal_row_id is not None:
                db.add(B1FindingSignal(finding_id=finding_row_id, signal_id=signal_row_id))

    doc_number_raw = None
    if result.mrz is not None:
        for field in result.mrz.fields:
            if field.field == "doc_number":
                doc_number_raw = field.value
                break

    session = await db.get(B1Session, session_id)
    session_status = "settled"
    if session is not None:
        session.status = session_status

    db.add(B1Case(
        id=str(uuid.uuid4()), session_id=session_id, run_id=run_id,
        case_id=f"CASE-{run_id[:8].upper()}", screened_at=_now(),
        document_class=session.document_class if session is not None else None,
        issuing_state=None,
        document_number_masked=mask_document_number(doc_number_raw),
    ))

    await db.commit()
