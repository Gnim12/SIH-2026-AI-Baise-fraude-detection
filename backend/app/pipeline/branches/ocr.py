"""OCR branch (BACKEND_BRIEF.md §5.3/§6.2): the REAL OCRReader, not a stub.

Task brief for this milestone: the OCR branch calls the real MRZReader +
VizReader + cross-check pipeline built in app/ocr/. `require_trained_weights`
is False, since no trained CRNN checkpoint exists yet (§1.3) -- see
app/pipeline/orchestrator.py's module-level log for the pipeline-level
(operator-visible) version of that warning; MRZReader itself also logs it.

Because stub-mode MRZ decoding requires foreknowledge of the ground-truth
line strings (it fabricates logprobs from them, it does not read the image),
a real unlabelled capture's MRZ band cannot be decoded until the CRNN is
trained. That is a real coverage gap, not an implementation shortcut, and is
reported as MRZ_MODEL_UNAVAILABLE rather than silently skipped or faked.
VIZ extraction (RapidOCR) has no such limitation and always runs for real.
"""
from __future__ import annotations

import logging
import uuid
from functools import lru_cache

from app.contracts import Signal
from app.ocr.mrz.infer import MRZReader
from app.ocr.reader import OCRReader, OCRResult
from app.ocr.viz import VizReader
from app.pipeline.context import DocumentInput, PipelineContext

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _shared_viz_reader() -> VizReader:
    # Loading RapidOCR's ONNX sessions is expensive (§2: "one shared session
    # per model") -- construct once per process, not once per request.
    return VizReader()


@lru_cache(maxsize=1)
def _shared_mrz_reader() -> MRZReader:
    return MRZReader(require_trained_weights=False)


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


async def branch_ocr(doc: DocumentInput, ctx: PipelineContext) -> OCRResult:
    import asyncio

    reader = OCRReader(mrz_reader=_shared_mrz_reader(), viz_reader=_shared_viz_reader())

    def _read() -> OCRResult:
        try:
            return reader.read(
                doc.rectified,
                document_id=doc.document_id,
                stub_mrz_ground_truth=doc.mrz_ground_truth,
            )
        except ValueError:
            # MRZReader is in stub mode and this capture's MRZ band was
            # located but no ground truth was supplied -- see module
            # docstring. Re-run VIZ-only extraction so the branch still
            # returns real VIZ fields instead of failing the whole branch.
            logger.warning(
                "OCR branch: MRZ band located on document %s but MRZReader is in "
                "stub mode with no ground truth to decode against (no trained CRNN "
                "weights, §1.3) -- reporting MRZ_MODEL_UNAVAILABLE and continuing "
                "with real VIZ-only extraction.",
                doc.document_id,
            )
            viz_boxes = reader.viz_reader.read_boxes(
                doc.rectified, mrz_band=None, document_id=doc.document_id
            )
            from app.ocr.viz import classify_fields

            from app.contracts import ExtractedField

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
                detail=(
                    "The machine-readable zone could not be decoded: no trained CRNN "
                    "checkpoint exists yet, so the reader cannot recognise unlabelled "
                    "MRZ text. This is a coverage gap, not a clean result."
                ),
            )
            return OCRResult(
                fields=fields, signals=[signal], mrz=None,
                confidence=0.5 if viz_fields else 0.0,
                mrz_present=False, viz_field_count=len(viz_fields),
            )

    return await asyncio.to_thread(_read)
