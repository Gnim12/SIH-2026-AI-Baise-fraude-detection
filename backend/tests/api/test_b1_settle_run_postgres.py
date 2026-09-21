"""settle_run must give every emitted signal its own b1_signal row and link
findings by row id, never by signal code. Runs against Postgres (a throwaway
schema) because the duplicate-pair failure is a unique-constraint violation
the SQLite test DB does not reproduce the same way."""
from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.api import schemas as s
from app.storage import b1_repositories as repo
from app.storage.b1_models import B1FindingSignal, B1Signal
from app.storage.db import Base

PG_URL = "postgresql+asyncpg://screening:screening@localhost:5433/screening"


def _signal(code: str, n: int) -> s.Signal:
    return s.Signal(
        signal_id=code, modality=list(s.Modality)[0], label=code, detail=f"instance {n}",
        emitted_at="2026-01-01T00:00:00+00:00", model_pin="test",
    )


def _finding(fid: str, signals: list[s.Signal]) -> s.Finding:
    return s.Finding(
        finding_id=fid, title=fid, hypothesis="h", severity=list(s.Severity)[0],
        signals=signals, converged_modalities=[],
    )


def _result(session_id: str, findings: list[s.Finding]) -> s.ScreeningResult:
    return s.ScreeningResult(
        session_id=session_id, verdict=list(s.Verdict)[0], reason="r", findings=findings, coverage=[],
        cross_check=[], identity_graph=s.GraphResult(ran=False, has_prior_encounter=False, enrolment_count=0),
        completed_at="2026-01-01T00:00:00+00:00",
    )


async def _run(findings_factory):
    schema = f"t_{uuid.uuid4().hex[:8]}"
    engine = create_async_engine(PG_URL, connect_args={"server_settings": {"search_path": schema}})
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
            await conn.run_sync(Base.metadata.create_all)
        async with AsyncSession(engine, expire_on_commit=False) as db:
            session = await repo.create_session(db)
            run = await repo.create_run(db, session.id)
            await repo.settle_run(db, run.id, session.id, _result(session.id, findings_factory()))
            signals = (await db.execute(select(func.count()).select_from(B1Signal))).scalar_one()
            links = (await db.execute(select(func.count()).select_from(B1FindingSignal))).scalar_one()
        return signals, links
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()


def _skip_without_postgres():
    async def _ping():
        e = create_async_engine(PG_URL)
        try:
            async with e.connect() as c:
                await c.execute(text("SELECT 1"))
        finally:
            await e.dispose()
    try:
        asyncio.run(_ping())
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Postgres unavailable: {exc}")


def test_five_same_code_signals_in_one_finding():
    _skip_without_postgres()
    signals, links = asyncio.run(_run(lambda: [
        _finding("F1", [_signal("VIZ_FIELD_NOT_READABLE", i) for i in range(5)])]))
    assert (signals, links) == (5, 5)


def test_two_findings_sharing_a_signal_code():
    _skip_without_postgres()
    signals, links = asyncio.run(_run(lambda: [
        _finding("F1", [_signal("VIZ_FIELD_NOT_READABLE", 1), _signal("VIZ_FIELD_NOT_READABLE", 2)]),
        _finding("F2", [_signal("VIZ_FIELD_NOT_READABLE", 3)])]))
    assert (signals, links) == (3, 3)


def test_same_instance_listed_twice_is_deduped():
    _skip_without_postgres()

    def make():
        sig = _signal("VIZ_FIELD_NOT_READABLE", 1)
        return [_finding("F1", [sig, sig])]

    signals, links = asyncio.run(_run(make))
    assert (signals, links) == (1, 1)
