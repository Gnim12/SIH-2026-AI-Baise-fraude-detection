"""BACKEND_BRIEF.md §7: GET /api/v1/health -- model manifest, watchlist age,
db state."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ocr.mrz.runtime import get_mrz_runtime
from app.registry import load_manifest
from app.storage.db import get_session

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
async def health(db: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    try:
        await db.execute(text("SELECT 1"))
        db_state = "ok"
    except Exception as exc:  # noqa: BLE001
        db_state = f"unreachable: {exc}"

    manifest = load_manifest()
    model_versions = {k: v.get("version") for k, v in manifest.items()}
    # The manifest version of a placeholder/unloaded MRZ model is not a running model.
    model_versions["mrz_crnn"] = get_mrz_runtime().status.version

    return {
        "status": "ok",
        "db": db_state,
        "modelVersions": model_versions,
        # M5 (watchlist sync) not built yet -- explicit null, not a fabricated timestamp.
        "watchlistSyncedAt": None,
    }
