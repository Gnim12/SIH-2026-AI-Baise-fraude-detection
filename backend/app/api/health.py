"""BACKEND_BRIEF.md §7: GET /api/v1/health -- model manifest, watchlist age,
db state."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.registry import load_manifest
from app.storage.db import get_session

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
async def health(db: AsyncSession = Depends(get_session)):
    try:
        await db.execute(text("SELECT 1"))
        db_state = "ok"
    except Exception as exc:  # noqa: BLE001
        db_state = f"unreachable: {exc}"

    manifest = load_manifest()
    model_versions = {k: v.get("version") for k, v in manifest.items()}

    return {
        "status": "ok",
        "db": db_state,
        "modelVersions": model_versions,
        # M5 (watchlist sync) not built yet -- explicit null, not a fabricated timestamp.
        "watchlistSyncedAt": None,
    }
