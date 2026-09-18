"""BACKEND_BRIEF.md §7: GET /api/v1/history?lane=&since= -- sealed sessions,
backed by real Postgres rows (see app/storage/models.py)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import require_officer
from app.auth.models import Officer
from app.contracts.wire import to_wire
from app.storage.db import get_session
from app.storage.repositories import get_session as repo_get_session
from app.storage.repositories import list_history

router = APIRouter(prefix="/api/v1/history", tags=["history"])


@router.get("")
async def get_history(
    lane: Optional[str] = None, since: Optional[str] = None,
    db: AsyncSession = Depends(get_session),
    officer: Officer = Depends(require_officer),
):
    sessions = await list_history(db, lane_id=lane, since=since)
    return [to_wire(s) for s in sessions]


@router.get("/{session_id}")
async def get_history_entry(
    session_id: str,
    db: AsyncSession = Depends(get_session),
    officer: Officer = Depends(require_officer),
):
    session = await repo_get_session(db, session_id)
    if session is None or not session.sealed:
        raise HTTPException(404, "sealed session not found")
    return to_wire(session)
