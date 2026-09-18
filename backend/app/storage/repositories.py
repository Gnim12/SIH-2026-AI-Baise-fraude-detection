"""Repository functions over the `sessions` table (see models.py docstring
for the M1-sized schema decision)."""
from __future__ import annotations

import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts import ScreeningSession

from .models import SessionRow


def _is_sqlite(db: AsyncSession) -> bool:
    return db.bind.dialect.name == "sqlite"


async def upsert_session(db: AsyncSession, session: ScreeningSession) -> None:
    payload = session.model_dump(mode="json")
    now = datetime.datetime.now(datetime.timezone.utc)

    if _is_sqlite(db):
        # sqlite (used only by the local test suite) has no native upsert
        # helper as convenient as Postgres's ON CONFLICT; a plain
        # merge-by-select is fine at test scale.
        existing = await db.get(SessionRow, session.session_id)
        if existing is None:
            db.add(SessionRow(
                session_id=session.session_id, lane_id=session.lane_id,
                officer_id=session.officer_id, started_at=session.started_at,
                band=session.band, sealed=session.sealed, updated_at=now, payload=payload,
            ))
        else:
            existing.band = session.band
            existing.sealed = session.sealed
            existing.updated_at = now
            existing.payload = payload
        await db.commit()
        return

    stmt = pg_insert(SessionRow).values(
        session_id=session.session_id, lane_id=session.lane_id,
        officer_id=session.officer_id, started_at=session.started_at,
        band=session.band, sealed=session.sealed, updated_at=now, payload=payload,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[SessionRow.session_id],
        set_={"band": stmt.excluded.band, "sealed": stmt.excluded.sealed,
              "updated_at": stmt.excluded.updated_at, "payload": stmt.excluded.payload},
    )
    await db.execute(stmt)
    await db.commit()


async def get_session(db: AsyncSession, session_id: str) -> Optional[ScreeningSession]:
    row = await db.get(SessionRow, session_id)
    if row is None:
        return None
    return ScreeningSession.model_validate(row.payload)


async def list_history(
    db: AsyncSession, *, lane_id: Optional[str] = None, since: Optional[str] = None,
) -> list[ScreeningSession]:
    stmt = select(SessionRow).where(SessionRow.sealed.is_(True)).order_by(SessionRow.updated_at.desc())
    if lane_id:
        stmt = stmt.where(SessionRow.lane_id == lane_id)
    if since:
        stmt = stmt.where(SessionRow.started_at >= since)
    result = await db.execute(stmt)
    return [ScreeningSession.model_validate(row.payload) for row in result.scalars().all()]
