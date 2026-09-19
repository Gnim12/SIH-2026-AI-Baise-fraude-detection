"""OCR branch (BACKEND_BRIEF.md §5.3/§6.2): the REAL OCRReader, not a stub.

The MRZ reader is whatever app/ocr/mrz/runtime.py resolved at startup: the
real ONNX recogniser with weights verified against the manifest, or nothing.
There is no stub fallback on this path. When the model is missing,
hash-mismatched or still a placeholder, the branch reports
MRZ_MODEL_UNAVAILABLE and carries on with real VIZ extraction -- a check that
cannot run is reported as such, never as a pass or a fabricated reading.
"""
from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Optional

from app.contracts import Signal
from app.ocr.mrz.infer import MRZReader
from app.ocr.mrz.runtime import get_mrz_runtime
from app.ocr.reader import OCRReader, OCRResult
from app.ocr.viz import VizReader
from app.pipeline.context import DocumentInput, PipelineContext

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _shared_viz_reader() -> VizReader:
    # Loading RapidOCR's ONNX sessions is expensive (§2: "one shared session
    # per model") -- construct once per process, not once per request.
    return VizReader()


def _shared_mrz_reader() -> Optional[MRZReader]:
    """The process-wide MRZ reader, or None while the model is unavailable.

    Production weights are always required: runtime.load_mrz_runtime builds
    the reader with require_trained_weights=True. Stub mode is reachable only
    through runtime.install_reader_for_tests.
    """
    return get_mrz_runtime().reader


def warm_up() -> None:
    """Force-construct both shared readers now, synchronously.

    §2's "one shared session per model" only guarantees the ONNX sessions are
    built *once* -- it says nothing about *when*. Left to their lazy
    @lru_cache default, that "once" happens on the first real branch_ocr()
    call, which pays RapidOCR's session-load cost (measured ~2-7s on dev
    hardware) inside that request's BUDGET_MS["ocr"] window and times it out.
    Call this from app startup (app/main.py's lifespan) so the cost is paid
    once at boot instead of on whichever request happens to run first.
    """
    _shared_viz_reader()
    _shared_mrz_reader()


def _viz_only_result(doc: DocumentInput, viz_reader: VizReader, detail: str) -> OCRResult:
    from app.contracts import ExtractedField
    from app.ocr.viz import classify_fields

    viz_boxes = viz_reader.read_boxes(doc.rectified, mrz_band=None, document_id=doc.document_id)
    viz_fields = classify_fields(viz_boxes)
    fields = {
        f"viz.{vf.name}": ExtractedField(
            key=vf.name, label=vf.name.replace("_", " ").capitalize(),
            value=vf.value, confidence=vf.confidence, source="VIZ",
        )
        for vf in viz_fields
    }
    signal = Signal(
        id=str(uuid.uuid4()), code="MRZ_MODEL_UNAVAILABLE", module="ocr",
        severity="info", weight=0.0,
        detail=f"The machine-readable zone could not be decoded: {detail} This is a coverage gap, not a clean result.",
    )
    return OCRResult(
        fields=fields, signals=[signal], mrz=None,
        confidence=0.5 if viz_fields else 0.0,
        mrz_present=False, viz_field_count=len(viz_fields),
    )


async def branch_ocr(doc: DocumentInput, ctx: PipelineContext) -> OCRResult:
    import asyncio

    viz_reader = _shared_viz_reader()
    mrz_reader = _shared_mrz_reader()
    if mrz_reader is None:
        reason = get_mrz_runtime().status.reason or "the MRZ model is unavailable."
        logger.warning("OCR branch: MRZ unavailable on document %s -- %s", doc.document_id, reason)
        return await asyncio.to_thread(_viz_only_result, doc, viz_reader, reason)

    reader = OCRReader(mrz_reader=mrz_reader, viz_reader=viz_reader)

    def _read() -> OCRResult:
        try:
            return reader.read(
                doc.rectified,
                document_id=doc.document_id,
                # Only a stub-mode reader (tests) can act on ground truth; a
                # real reader refuses it, so it must never be forwarded to one.
                stub_mrz_ground_truth=doc.mrz_ground_truth if mrz_reader.stub_mode else None,
            )
        except ValueError:
            # Stub-mode reader (tests only) with no ground truth supplied.
            return _viz_only_result(
                doc, viz_reader, "a stub reader was given no ground truth to decode against."
            )

    return await asyncio.to_thread(_read)
