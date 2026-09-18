"""BACKEND_BRIEF.md §7: POST /api/v1/screening (multipart), POST
/api/v1/screening/{id}/decision. Field names for the multipart body match
the already-built frontend's src/api/capture.ts `submitCapture` exactly
(documentCount, document_{i}_type, document_{i}, liveFrame,
liveFrameCaptureMethod, videoSweep?, checkpointId, officerId) -- that file
is the one place the wire shape was already decided, since no real backend
existed yet to dictate one.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.datastructures import UploadFile

from app.auth.dependencies import require_officer
from app.auth.models import Officer
from app.contracts import ScreeningSession
from app.contracts.wire import to_wire
from app.pipeline import orchestrator
from app.pipeline.context import DocumentInput
from app.storage.db import get_session
from app.storage.repositories import get_session as repo_get_session
from app.storage.repositories import upsert_session

from .media import save_document_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/screening", tags=["screening"])


def _decode_image(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(422, "uploaded file is not a decodable image")
    return image


@router.post("", status_code=201)
async def start_screening(
    request: Request,
    db: AsyncSession = Depends(get_session),
    officer: Officer = Depends(require_officer),
):
    form = await request.form()

    document_count = int(form.get("documentCount", "0"))
    if document_count < 1:
        raise HTTPException(422, "at least one document is required")

    checkpoint_id = str(form.get("checkpointId", "unknown-lane"))
    # The authenticated officer (from the session cookie) is the source of
    # truth, not the form field the frontend also sends -- a logged-in
    # officer cannot submit a screening attributed to someone else.
    officer_id = officer.officer_id
    session_id = str(uuid.uuid4())

    documents: list[DocumentInput] = []
    document_image_urls: dict[str, str] = {}
    for i in range(document_count):
        upload = form.get(f"document_{i}")
        if upload is None or not isinstance(upload, UploadFile):
            raise HTTPException(422, f"document_{i} is missing")
        doc_type = str(form.get(f"document_{i}_type", "UNKNOWN"))
        raw = await upload.read()
        image = _decode_image(raw)

        document_id = f"{session_id}-doc{i}"
        image_url = save_document_image(document_id, raw)
        document_image_urls[document_id] = image_url

        # Dev/test-only affordance -- see app/pipeline/context.py's
        # DocumentInput.mrz_ground_truth docstring. The real frontend never
        # sends this field; it exists so an integration test can exercise a
        # genuine checksum-constrained MRZ decode against a real image
        # without a trained CRNN checkpoint (§1.3).
        ground_truth_raw = form.get(f"document_{i}_mrz_ground_truth")
        mrz_ground_truth = json.loads(ground_truth_raw) if ground_truth_raw else None

        documents.append(DocumentInput(
            document_id=document_id, doc_type=doc_type, original_bytes=raw,
            rectified=image, mrz_ground_truth=mrz_ground_truth,
        ))

    live_frame_upload = form.get("liveFrame")
    live_frame = await live_frame_upload.read() if isinstance(live_frame_upload, UploadFile) else None
    # LiveFaceCapture.tsx's dev/test file-upload fallback (used when
    # getUserMedia fails/is unavailable, or picked manually) sends this so
    # a still photo is never conflated with an actual live capture --
    # see FaceResult.capture_method.
    live_frame_capture_method = str(form.get("liveFrameCaptureMethod", "live"))
    if live_frame_capture_method not in ("live", "upload"):
        live_frame_capture_method = "live"
    sweep_upload = form.get("videoSweep")
    video_sweep = await sweep_upload.read() if isinstance(sweep_upload, UploadFile) else None

    inputs = orchestrator.PipelineInputs(
        session_id=session_id, lane_id=checkpoint_id, officer_id=officer_id,
        documents=documents, live_frame=live_frame,
        live_frame_capture_method=live_frame_capture_method, video_sweep=video_sweep,
        document_image_urls=document_image_urls,
    )
    ctx = orchestrator.build_context(session_id, checkpoint_id, officer_id)

    async def _run_and_persist() -> None:
        # Own DB session: this runs after the request's session is closed.
        from app.storage.db import async_session_factory

        try:
            session = await orchestrator.run(inputs, ctx)
        except Exception:
            logger.exception("pipeline run failed for session %s", session_id)
            return
        async with async_session_factory() as persist_db:
            await upsert_session(persist_db, session)

    asyncio.create_task(_run_and_persist())

    return {"sessionId": session_id, "status": "processing"}


@router.get("/{session_id}")
async def get_screening(
    session_id: str,
    db: AsyncSession = Depends(get_session),
    officer: Officer = Depends(require_officer),
):
    session = await repo_get_session(db, session_id)
    if session is None:
        bus = orchestrator.event_bus.get(session_id)
        if bus is None:
            raise HTTPException(404, "session not found")
        # Still processing: no persisted row yet, but events exist.
        raise HTTPException(202, "session still processing")
    return to_wire(session)


@router.post("/{session_id}/decision")
async def submit_decision(
    session_id: str,
    request: Request,
    db: AsyncSession = Depends(get_session),
    officer: Officer = Depends(require_officer),
):
    body = await request.json()
    decision = body.get("decision")
    note = (body.get("note") or "").strip()
    if decision not in ("CLEAR", "SECONDARY", "HOLD", "REFER"):
        raise HTTPException(422, "decision must be one of CLEAR, SECONDARY, HOLD, REFER")

    session = await repo_get_session(db, session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    if session.sealed:
        raise HTTPException(409, "session is already sealed")

    override = decision != session.band
    if (override or decision in ("HOLD", "REFER")) and not note:
        raise HTTPException(422, "a note is required when overriding the recommendation, or for HOLD/REFER")

    from app.contracts import OfficerDecision

    sealed_session = session.model_copy(update={
        "officer_decision": OfficerDecision(
            decision=decision, note=note,
            decided_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            override=override,
        ),
        "sealed": True,
    })
    await upsert_session(db, sealed_session)
    return to_wire(sealed_session)
