"""THE DAG. BACKEND_BRIEF.md §5.3/§5.4: gates, wave 1, wave 2, cross-document
join, correlation, fusion, timeouts, degraded mode.

M1 scope (this milestone): the skeleton is real -- gates, wave-1 concurrency,
per-branch timeout handling via `Degraded`, event publication as branches
complete, and a stub-but-structurally-real wave 2/crossdoc/fusion. The OCR
branch is the one real analysis branch (app/ocr/ already built and tested);
forensics/face/ovd/template stay stubs until M4. Fusion is a placeholder risk
sum, not the real three-layer system in §6.9 -- that is explicitly M3/M7
work; do not read the band thresholds below as calibrated.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.config import settings
from app.contracts import (
    ClassifiedEvent,
    CrossdocEvent,
    DatabaseEvent,
    DecisionEvent,
    ExtractedField,
    FaceEvent,
    FaceResult,
    ForensicsEvent,
    OcrEvent,
    QualityEvent,
    ReceivedEvent,
    ScreenedDocument,
    ScreeningSession,
    Signal,
)
from app.registry import verify_all

from .branches.face import BUDGET_MS as FACE_BUDGET_MS
from .branches.face import branch_face
from .branches.forensics import BUDGET_MS as FORENSICS_BUDGET_MS
from .branches.forensics import branch_forensics
from .branches.ocr import branch_ocr
from .branches.ovd import BUDGET_MS as OVD_BUDGET_MS
from .branches.ovd import branch_ovd
from .branches.template import BUDGET_MS as TEMPLATE_BUDGET_MS
from .branches.template import branch_template
from .bus import event_bus
from .context import DocumentInput, PipelineContext
from .degraded import Degraded
from .gates import classify, quality_gate
from .wave2.database import branch_database
from .wave2.rules import branch_rules

logger = logging.getLogger(__name__)

BUDGET_MS = {
    "ocr": settings.ocr_budget_ms, "forensics": FORENSICS_BUDGET_MS, "face": FACE_BUDGET_MS,
    "ovd": OVD_BUDGET_MS, "template": TEMPLATE_BUDGET_MS,
}

if not settings.mrz_require_trained_weights:
    # BACKEND_BRIEF.md §1.3: this is an operator-visible, pipeline-level fact
    # (no trained CRNN checkpoint exists), not just something buried in a
    # unit test or the OCR module's own logs -- log it loudly here too, at
    # orchestrator import time, so it shows up once per process start.
    logger.warning(
        "=" * 78 + "\n"
        "ORCHESTRATOR STARTUP: MRZReader has NO TRAINED CRNN WEIGHTS (BACKEND_BRIEF.md "
        "§1.3).\nThe OCR branch runs the real reader end to end, but MRZ decoding only "
        "works\nwhen the caller supplies known ground-truth MRZ lines (stub mode "
        "fabricates\nrecognition confidence from them; it does not read the image). "
        "Unlabelled\nreal captures will report MRZ_MODEL_UNAVAILABLE instead of a "
        "decoded MRZ.\nTrain the CRNN (app/ocr/mrz/train.py) before relying on MRZ "
        "results in\nproduction.\n" + "=" * 78
    )


@dataclass
class PipelineInputs:
    session_id: str
    lane_id: str
    officer_id: str
    documents: list[DocumentInput]
    live_frame: Optional[bytes] = None
    live_frame_capture_method: str = "live"  # "live" | "upload" -- see FaceResult.capture_method
    video_sweep: Optional[bytes] = None
    document_image_urls: dict[str, str] = field(default_factory=dict)


async def _run_wave1_branch(name: str, coro) -> object:
    try:
        return await asyncio.wait_for(coro, BUDGET_MS[name] / 1000)
    except asyncio.TimeoutError:
        logger.warning("branch %s timed out after %dms", name, BUDGET_MS[name])
        return Degraded(name, "timeout")
    except Exception as exc:  # noqa: BLE001 -- failure isolation is the point (§5.3)
        logger.exception("branch %s failed", name)
        return Degraded(name, str(exc))


async def wave_1(
    doc: DocumentInput, ctx: PipelineContext, *, live_frame: Optional[bytes], sweep,
    run_face: bool, live_frame_capture_method: str = "live",
) -> dict[str, object]:
    tasks: dict[str, asyncio.Task] = {
        "ocr": asyncio.create_task(branch_ocr(doc, ctx)),
        "forensics": asyncio.create_task(branch_forensics(doc.original_bytes, ctx)),
        "template": asyncio.create_task(branch_template(doc.rectified, ctx)),
    }
    if run_face:
        tasks["face"] = asyncio.create_task(
            branch_face(doc.rectified, live_frame, ctx, capture_method=live_frame_capture_method)
        )
    if sweep is not None:
        tasks["ovd"] = asyncio.create_task(branch_ovd(sweep, ctx))

    results: dict[str, object] = {}
    for name, task in tasks.items():
        results[name] = await _run_wave1_branch(name, task)
    return results


def _degraded_signal(branch: str, reason: str) -> Signal:
    return Signal(
        id=str(uuid.uuid4()), code=f"{branch.upper()}_UNAVAILABLE", module="system",
        severity="info", weight=0.0,
        detail=f"The {branch} module did not complete ({reason}). Reported as a coverage gap.",
    )


async def _publish(session_id: str, event) -> None:
    await event_bus.publish(session_id, event)


async def run_document(
    doc: DocumentInput, ctx: PipelineContext, session_id: str, *,
    live_frame: Optional[bytes], sweep, run_face: bool, image_url: str,
    live_frame_capture_method: str = "live",
) -> dict[str, object]:
    q = await quality_gate(doc.document_id, doc.original_bytes)
    await _publish(session_id, QualityEvent(
        document_id=doc.document_id, ok=q.ok, dpi=q.dpi, reason=q.reason, hint=q.hint,
    ))
    if not q.ok:
        return {"quality": q}

    cls = await classify(doc.document_id, doc.original_bytes)
    await _publish(session_id, ClassifiedEvent(
        document_id=doc.document_id, type=cls.doc_type, country=cls.country,
        version=cls.version, confidence=cls.confidence, image_url=image_url,
    ))

    wave1 = await wave_1(
        doc, ctx, live_frame=live_frame, sweep=sweep, run_face=run_face,
        live_frame_capture_method=live_frame_capture_method,
    )

    coverage_flags: list[str] = []

    ocr_result = wave1["ocr"]
    if isinstance(ocr_result, Degraded):
        ocr_signals = [_degraded_signal("ocr", ocr_result.reason)]
        ocr_fields: list[ExtractedField] = []
        mrz = None
        coverage_flags.append("no_ocr")
    else:
        ocr_signals = list(ocr_result.signals)
        ocr_fields = list(ocr_result.fields.values())
        mrz = ocr_result.mrz
    await _publish(session_id, OcrEvent(
        document_id=doc.document_id, fields=ocr_fields, mrz=mrz, signals=ocr_signals,
    ))

    forensics_result = wave1["forensics"]
    if isinstance(forensics_result, Degraded):
        forensics_signals = [_degraded_signal("forensics", forensics_result.reason)]
        views: dict[str, str] = {}
        coverage_flags.append("no_forensics")
    else:
        forensics_signals = list(forensics_result.signals)
        views = dict(forensics_result.views)
    views.setdefault("rgb", image_url)
    await _publish(session_id, ForensicsEvent(
        document_id=doc.document_id, views=views, signals=forensics_signals,
    ))

    template_result = wave1.get("template")
    template_signals: list[Signal] = []
    if isinstance(template_result, Degraded):
        template_signals = [_degraded_signal("template", template_result.reason)]
        coverage_flags.append("no_template")
    elif template_result is not None:
        template_signals = list(template_result.signals)

    # Face is published once per session (the FaceEvent wire shape carries no
    # documentId), by the caller, once run_document returns -- not here.
    face_result = wave1.get("face")
    face: Optional[FaceResult] = None
    face_signals: list[Signal] = []
    if run_face:
        if isinstance(face_result, Degraded):
            face_signals = [_degraded_signal("face", face_result.reason)]
            coverage_flags.append("no_biometric")
        elif face_result is not None:
            face = face_result.face
            face_signals = list(face_result.signals)

    ovd_result = wave1.get("ovd")
    ovd_signals: list[Signal] = []
    if isinstance(ovd_result, Degraded):
        ovd_signals = [_degraded_signal("ovd", ovd_result.reason)]
    elif ovd_result is not None:
        ovd_signals = list(ovd_result.signals)
    elif sweep is None:
        coverage_flags.append("no_ovd")

    all_signals = ocr_signals + forensics_signals + template_signals + ovd_signals

    document = ScreenedDocument(
        id=doc.document_id, type=cls.doc_type, country=cls.country, version=cls.version,
        image_url=image_url, views=views, mrz=mrz, fields=ocr_fields,
        risk=None,
    )

    return {
        "quality": q, "document": document, "signals": all_signals,
        "face": face, "face_signals": face_signals, "coverage_flags": coverage_flags,
    }


async def run(inputs: PipelineInputs, ctx: PipelineContext) -> ScreeningSession:
    start = time.monotonic()
    session_id = inputs.session_id
    bus = event_bus.open(session_id)

    await _publish(session_id, ReceivedEvent(
        session_id=session_id, lane_id=inputs.lane_id, officer_id=inputs.officer_id,
    ))

    documents: list[ScreenedDocument] = []
    all_signals: list[Signal] = []
    coverage_flags: list[str] = []
    face: Optional[FaceResult] = None
    face_signals: list[Signal] = []

    for i, doc in enumerate(inputs.documents):
        image_url = inputs.document_image_urls.get(doc.document_id, f"/media/{doc.document_id}.jpg")
        result = await run_document(
            doc, ctx, session_id, live_frame=inputs.live_frame, sweep=inputs.video_sweep,
            run_face=(i == 0 and inputs.live_frame is not None), image_url=image_url,
            live_frame_capture_method=inputs.live_frame_capture_method,
        )
        if not result["quality"].ok:
            # BACKEND_BRIEF.md §5.1: quality gate is blocking. Fail =
            # RECAPTURE, no score at all for the whole session.
            session = ScreeningSession(
                session_id=session_id, lane_id=inputs.lane_id, officer_id=inputs.officer_id,
                started_at=_now_iso(), band="RECAPTURE", risk=None, confidence=None,
                abstained=False, recapture_reason=result["quality"].reason,
                recapture_hint=result["quality"].hint, documents=[], signals=[],
                face=None, graph=None, cross_document_signals=[], coverage_flags=[],
                timing_ms={"total": _elapsed_ms(start)}, sealed=False,
                model_versions=ctx.model_versions,
            )
            await _publish(session_id, DecisionEvent(
                band="RECAPTURE", risk=None, confidence=0.0, abstained=False,
                coverage_flags=[], timing_ms=session.timing_ms,
            ))
            bus.close()
            return session

        documents.append(result["document"])
        all_signals.extend(result["signals"])
        coverage_flags.extend(result["coverage_flags"])
        if result["face"] is not None or result["face_signals"]:
            face = result["face"]
            face_signals = result["face_signals"]

    if face is not None or face_signals:
        await _publish(session_id, FaceEvent(
            face=face or FaceResult(status="UNAVAILABLE", similarity=None, threshold=0.38, pad_verdict="not_run"),
            signals=face_signals,
        ))
        all_signals.extend(face_signals)

    # Wave 2 -- stub (M2/M3/M5 land the real rules engine + database queries).
    all_ocr_fields = [f for d in documents for f in d.fields]
    rules_result = await branch_rules(all_ocr_fields, ctx)
    database_result = await branch_database(all_ocr_fields, ctx)
    await _publish(session_id, DatabaseEvent(graph=database_result.graph, signals=database_result.signals))
    all_signals.extend(rules_result.signals)
    all_signals.extend(database_result.signals)

    # Cross-document join (Stage 5.5) -- stub, no real cross-doc rules yet.
    cross_document_signals: list[Signal] = []
    await _publish(session_id, CrossdocEvent(signals=cross_document_signals))

    # Fusion (Stage 7) -- STUB. Real three-layer fusion (deterministic
    # overrides, learned model, calibrated confidence/abstention) is M3/M7
    # work (§6.9). This is a placeholder risk sum so the DAG has a real
    # terminal decision event to publish.
    risk = min(100.0, sum(s.weight for s in all_signals) + sum(s.weight for s in cross_document_signals))
    if risk >= 60:
        band = "HOLD"
    elif risk >= 25:
        band = "SECONDARY"
    else:
        band = "CLEAR"
    confidence = max(0.3, 0.95 - 0.1 * len(coverage_flags))
    timing_ms = {"total": _elapsed_ms(start)}

    await _publish(session_id, DecisionEvent(
        band=band, risk=risk, confidence=confidence, abstained=False,
        coverage_flags=sorted(set(coverage_flags)), timing_ms=timing_ms,
    ))

    session = ScreeningSession(
        session_id=session_id, lane_id=inputs.lane_id, officer_id=inputs.officer_id,
        started_at=_now_iso(), band=band, risk=risk, confidence=confidence,
        abstained=False, documents=documents, signals=all_signals, face=face,
        graph=database_result.graph, cross_document_signals=cross_document_signals,
        coverage_flags=sorted(set(coverage_flags)), timing_ms=timing_ms, sealed=False,
        model_versions=ctx.model_versions,
    )
    bus.close()
    return session


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _elapsed_ms(start: float) -> float:
    return round((time.monotonic() - start) * 1000, 1)


def build_context(session_id: str, lane_id: str, officer_id: str) -> PipelineContext:
    model_versions = verify_all()
    return PipelineContext(
        session_id=session_id, lane_id=lane_id, officer_id=officer_id,
        model_versions=model_versions,
        mrz_require_trained_weights=settings.mrz_require_trained_weights,
    )
