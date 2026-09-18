"""viz-read stage: adapter over RapidOCR (app/ocr/viz.py), via the shared
VizReader instance in app/pipeline/branches/ocr.py. No OCR logic is
reimplemented here.
"""
from __future__ import annotations

import asyncio
import datetime

from app.api.schemas import Modality, Signal, StageState
from app.ocr.viz import VizField, classify_fields
from app.pipeline.branches.ocr import _shared_viz_reader
from app.pipeline.pins import resolve_pins
from app.pipeline.registry import StageId
from app.pipeline.runner import StageContext, StageResult

LOW_CONFIDENCE_THRESHOLD = 0.6
# The fields Gate 1 cross-checks against MRZ (app/ocr/reader.py's
# _CROSSCHECK_FIELDS); a document missing all of these has nothing readable
# to check.
CROSSCHECK_VIZ_FIELDS = {"birth_date", "expiry_date", "doc_number", "name", "nationality", "sex"}


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _signal(code: str, detail: str, pin: str) -> Signal:
    return Signal(
        signal_id=code, modality=Modality.VIZ, label=code.replace("_", " ").title(),
        detail=detail, emitted_at=_now_iso(), model_pin=pin,
    )


class VizReadStage:
    id = StageId.VIZ_READ

    async def run(self, ctx: StageContext) -> StageResult:
        pin = resolve_pins().get("rapidocr_rec", "rapidocr unknown")
        reader = _shared_viz_reader()
        image = ctx.inputs["document_image"]
        document_id = ctx.inputs.get("document_id", "document")
        mrz_band = ctx.inputs.get("mrz_band")  # set by the runner from mrz-read's artefact, if available

        def _read() -> list[VizField]:
            boxes = reader.read_boxes(image, mrz_band=mrz_band, document_id=document_id)
            return classify_fields(boxes)

        viz_fields = await asyncio.to_thread(_read)

        signals: list[Signal] = []
        found_names = {f.name for f in viz_fields}
        for missing in sorted(CROSSCHECK_VIZ_FIELDS - found_names):
            signals.append(_signal(
                "VIZ_FIELD_NOT_READABLE", f"The {missing.replace('_', ' ')} field could not be read from the VIZ.", pin,
            ))

        low_conf = [f for f in viz_fields if f.confidence < LOW_CONFIDENCE_THRESHOLD]
        for f in low_conf:
            signals.append(_signal(
                "VIZ_LOW_CONFIDENCE", f"The {f.name.replace('_', ' ')} field was read with confidence {f.confidence:.2f}.", pin,
            ))

        state = StageState.FAILED if not viz_fields else StageState.PASSED
        return StageResult(
            state=state,
            detail=None if viz_fields else "No VIZ text could be extracted from this document.",
            signals=signals,
            artefacts={"viz_fields": viz_fields},
        )
