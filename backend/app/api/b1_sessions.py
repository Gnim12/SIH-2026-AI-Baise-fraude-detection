"""B1b: POST/GET /api/sessions, POST/DELETE /api/sessions/{id}/artefacts.

Separate namespace from BACKEND_BRIEF.md's /api/v1/screening (app/api/
screening.py) -- no /v1 prefix, no officer auth, per the B1b milestone
decision (real auth is explicitly a non-goal; see app/main.py wiring).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.storage import b1_repositories as repo
from app.storage.b1_models import B1Artefact
from app.storage.db import get_session

from .b1_errors import ApiError
from .b1_magic import ACCEPTED_TYPES, is_video, sniff_content_type
from .b1_wire import ArtefactOut, ArtefactUploadOut, SessionOut

router = APIRouter(prefix="/api/sessions", tags=["b1-sessions"])

VALID_KINDS = ("document-still", "ovd-sweep", "live-face", "secondary-still")
SWEEP_INVALIDATED_NOTICE = (
    "Document replaced. Re-capture the tilt sweep so both refer to the same document."
)


def _artefact_out(row: B1Artefact) -> ArtefactOut:
    return ArtefactOut(
        kind=row.kind, filename=row.filename, content_type=row.content_type, size_bytes=row.size_bytes,
        sha256=row.sha256, captured_at=row.captured_at, duration_ms=row.duration_ms,
        frame_count=row.frame_count, angular_coverage_deg=row.angular_coverage_deg,
    )


def _ext_for(content_type: str) -> str:
    return {
        "image/jpeg": ".jpg", "image/png": ".png", "application/pdf": ".pdf",
        "video/mp4": ".mp4", "video/webm": ".webm",
    }.get(content_type, "")


def _extract_sweep_metadata(path: Path) -> tuple[Optional[float], Optional[int], Optional[float]]:
    """(duration_ms, frame_count, angular_coverage_deg). Angular-coverage
    estimation is not implemented in this build: that element of the tuple
    is always None -- an unimplemented measurement is reported as absent,
    never invented. Duration and frame count come from real container
    metadata (cv2.VideoCapture), which is a genuinely available measurement."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None, None, None
        frame_count_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(frame_count_raw) if frame_count_raw and frame_count_raw > 0 else None
        duration_ms = (
            (frame_count_raw / fps) * 1000.0 if frame_count_raw and frame_count_raw > 0 and fps and fps > 0
            else None
        )
        return duration_ms, frame_count, None
    finally:
        cap.release()


@router.post("", status_code=201)
async def create_session_route(db: AsyncSession = Depends(get_session)) -> dict[str, str]:
    session = await repo.create_session(db)
    return {"sessionId": session.id}


@router.get("/{session_id}")
async def get_session_route(session_id: str, db: AsyncSession = Depends(get_session)) -> SessionOut:
    session = await repo.get_session(db, session_id)
    if session is None:
        raise ApiError(404, "session_not_found", f"No session found with id {session_id!r}.")
    artefacts = await repo.list_artefacts(db, session_id)
    return SessionOut(
        session_id=session.id, created_at=session.created_at, document_class=session.document_class,
        status=session.status, artefacts=[_artefact_out(a) for a in artefacts],
    )


@router.post("/{session_id}/artefacts")
async def upload_artefact_route(
    session_id: str,
    kind: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_session),
) -> ArtefactUploadOut:
    session = await repo.get_session(db, session_id)
    if session is None:
        raise ApiError(404, "session_not_found", f"No session found with id {session_id!r}.")
    if kind not in VALID_KINDS:
        raise ApiError(422, "invalid_kind", f"kind must be one of {', '.join(VALID_KINDS)}.")

    data = await file.read()
    sniffed = sniff_content_type(data[:32])
    if sniffed is None:
        raise ApiError(
            415, "unsupported_media_type",
            f"Unsupported file type. Accepted types: {', '.join(ACCEPTED_TYPES)}.",
        )

    max_mb = settings.b1_max_video_upload_mb if is_video(sniffed) else settings.b1_max_image_pdf_upload_mb
    max_bytes = max_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise ApiError(
            413, "file_too_large",
            f"File is {len(data)} bytes, which exceeds the {max_mb}MB limit for {kind} uploads.",
        )

    duration_ms = frame_count = angular_coverage_deg = None
    if kind == "ovd-sweep":
        if not is_video(sniffed):
            raise ApiError(422, "invalid_kind_for_type", "ovd-sweep artefacts must be a video (mp4 or webm).")
        with tempfile.NamedTemporaryFile(suffix=_ext_for(sniffed), delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            duration_ms, frame_count, angular_coverage_deg = _extract_sweep_metadata(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        if angular_coverage_deg is not None and angular_coverage_deg < settings.b1_ovd_angular_coverage_min_deg:
            raise ApiError(
                422, "insufficient_tilt_range", "Insufficient tilt range. Sweep again through a wider angle.",
            )

    artefact, sweep_deleted = await repo.save_artefact(
        db, session_id, kind, file.filename or kind, sniffed, data,
        duration_ms=duration_ms, frame_count=frame_count, angular_coverage_deg=angular_coverage_deg,
    )

    notice = SWEEP_INVALIDATED_NOTICE if sweep_deleted else None
    return ArtefactUploadOut(artefact=_artefact_out(artefact), notice=notice)


@router.delete("/{session_id}/artefacts/{kind}")
async def delete_artefact_route(
    session_id: str, kind: str, db: AsyncSession = Depends(get_session),
) -> dict[str, Optional[str]]:
    session = await repo.get_session(db, session_id)
    if session is None:
        raise ApiError(404, "session_not_found", f"No session found with id {session_id!r}.")

    deleted, sweep_deleted = await repo.delete_artefact(db, session_id, kind)
    if deleted is None:
        raise ApiError(404, "artefact_not_found", f"No {kind} artefact exists for this session.")

    notice = SWEEP_INVALIDATED_NOTICE if sweep_deleted else None
    return {"deletedKind": kind, "notice": notice}
