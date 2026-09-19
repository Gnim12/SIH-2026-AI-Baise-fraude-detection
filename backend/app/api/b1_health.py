"""B1b: GET /api/health (six-node status strip + model pins) and
GET /api/config/thresholds (read-only). B2 adds the officer-tunable
threshold list (frontend/src/lib/config/thresholdSeed.ts's six decision
thresholds) to the same GET response under `officerThresholds`, plus
PUT /api/config/thresholds and GET /api/config/audit -- see
app/storage/b2_repositories.py's THRESHOLD_DEFS and this module's own
`update_thresholds_route` for why this is a genuinely different resource
from the flat internal-settings dict this endpoint already returned, kept
alongside it rather than replacing it (no test or frontend code relies on
the old shape disappearing; see the B2 implementation report)."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.ocr.mrz.runtime import get_mrz_runtime
from app.registry import load_manifest
from app.storage import b2_repositories as repo
from app.storage.db import get_session

from .b1_errors import ApiError
from .b2_wire import ConfigChangeOut, ThresholdOut, ThresholdsUpdateIn

router = APIRouter(prefix="/api", tags=["b1-health"])


def _pin(manifest: dict[str, Any], key: str) -> Optional[str]:
    entry = manifest.get(key)
    return entry.get("version") if entry else None


@router.get("/health")
async def health_route(db: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False

    manifest = load_manifest()

    # Six nodes for the dashboard status strip. MRZ is `live` only when its
    # weights loaded and verified (app/ocr/mrz/runtime.py), and `unavailable`
    # when they are absent, mismatched or still a placeholder -- never `live`
    # on a placeholder. OVD is `hold` since no sweep
    # scanner is wired up; tamper/face/identity-graph are the three unbuilt
    # models -- `unavailable`, not silently omitted.
    mrz = get_mrz_runtime().status
    nodes = [
        {
            "node": "mrz", "status": mrz.state,
            "detail": (
                "MRZ recogniser weights loaded and verified." if mrz.state == "live"
                else f"MRZ recogniser unavailable: {mrz.reason}"
            ),
            "modelPin": mrz.pin, "modelVersion": mrz.version,
        },
        {
            "node": "viz", "status": "ok",
            "detail": "RapidOCR (PP-OCRv5, ONNX Runtime) is live.", "modelPin": _pin(manifest, "rapidocr_rec"),
        },
        {
            "node": "tamper", "status": "unavailable",
            "detail": "Tamper forensics is not implemented in this build.", "modelPin": _pin(manifest, "tamper_unet"),
        },
        {
            "node": "ovd", "status": "hold",
            "detail": "OVD sweep scanner offline.", "modelPin": None,
        },
        {
            "node": "face", "status": "unavailable",
            "detail": "Face verification is not implemented in this build.", "modelPin": _pin(manifest, "scrfd"),
        },
        {
            "node": "identity-graph", "status": "unavailable",
            "detail": "Identity graph lookup is not implemented in this build.", "modelPin": None,
        },
    ]

    return {"status": "ok" if db_ok else "degraded", "db": "ok" if db_ok else "unreachable", "nodes": nodes}


@router.get("/config/thresholds")
async def thresholds_route(db: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    current = await repo.get_threshold_values(db)
    officer_thresholds = [
        ThresholdOut(
            id=d["id"], label=d["label"], gate=d["gate"], value=current[d["id"]], unit=d["unit"],
            min=d["min"], max=d["max"], step=d["step"], default=d["default"], description=d["description"],
        )
        for d in repo.THRESHOLD_DEFS
    ]
    return {
        "ovdAngularCoverageMinDeg": settings.b1_ovd_angular_coverage_min_deg,
        "stageTimeoutS": settings.b1_stage_timeout_s,
        "maxImagePdfUploadMb": float(settings.b1_max_image_pdf_upload_mb),
        "maxVideoUploadMb": float(settings.b1_max_video_upload_mb),
        "officerThresholds": [t.model_dump(mode="json", by_alias=True) for t in officer_thresholds],
    }


@router.put("/config/thresholds")
async def update_thresholds_route(
    body: ThresholdsUpdateIn, db: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    try:
        entry_id, entry_hash = await repo.update_thresholds(
            db, officer_id=body.officer_id, terminal_id=body.terminal_id,
            changes=[c.model_dump(by_alias=False) for c in body.changes], reason=body.reason,
        )
    except repo.ThresholdValidationError as exc:
        raise ApiError(422, "invalid_threshold_change", exc.message) from None
    return {"entryId": entry_id, "entryHash": entry_hash}


@router.get("/config/audit")
async def config_audit_route(db: AsyncSession = Depends(get_session)) -> list[ConfigChangeOut]:
    changes = await repo.list_config_changes(db)
    out = []
    for c in changes:
        entry = await repo.get_audit_entry(db, c.audit_entry_id)
        out.append(ConfigChangeOut(
            change_id=c.id, changed_at=c.changed_at.isoformat(), changed_by=c.officer_id, reason=c.reason,
            changes=c.changes_json, audit_entry_hash=entry.entry_hash if entry else "",
        ))
    return out
