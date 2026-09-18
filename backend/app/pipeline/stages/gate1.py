"""gate-1: MRZ <-> VIZ cross-check. Reuses app/ocr/reader.py's
crosscheck_mrz_viz (extracted from OCRReader for this purpose, same
comparison logic OCRReader has always used, not reimplemented) against the
mrz-read and viz-read stages' already-computed artefacts -- neither reader
runs a second time here.

A Gate 1 failure is RECAPTURE_SUGGESTED, never a fraud verdict (glare, blur
and folds cause this far more often than forgery); no detail string or
signal emitted here may call the document or traveller fraudulent.
"""
from __future__ import annotations

import datetime

from app.api.schemas import CrossCheckRow, CrossCheckStatus, MismatchRow, Modality, Signal, StageState
from app.ocr.reader import crosscheck_mrz_viz
from app.pipeline.pins import resolve_pins
from app.pipeline.registry import StageId
from app.pipeline.runner import StageContext, StageResult
from app.pipeline.thresholds import DEFAULT_THRESHOLDS


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _signal(code: str, detail: str, pin: str) -> Signal:
    return Signal(
        signal_id=code, modality=Modality.CROSS_CHECK, label=code.replace("_", " ").title(),
        detail=detail, emitted_at=_now_iso(), model_pin=pin,
    )


class Gate1Stage:
    id = StageId.GATE_1

    async def run(self, ctx: StageContext) -> StageResult:
        pin = resolve_pins().get("mrz_crnn", "mrz_crnn unknown")
        mrz_artefacts = ctx.artefacts.get(StageId.MRZ_READ, {})
        viz_artefacts = ctx.artefacts.get(StageId.VIZ_READ, {})
        mrz_result = mrz_artefacts.get("mrz_read_result")
        viz_fields = viz_artefacts.get("viz_fields", [])

        if mrz_result is None or mrz_result.parsed is None:
            # mrz-read didn't produce a parsed MRZ (band not found, or
            # decode failure) -- Gate 1 cannot compare, so every field is
            # unreadable on the MRZ side. Not a fraud signal.
            return StageResult(
                state=StageState.FAILED,
                detail="Gate 1 could not run: no machine-readable zone was decoded to compare against the VIZ.",
                signals=[_signal(
                    "MRZ_VIZ_FIELD_UNREADABLE_ALL",
                    "The machine-readable zone was not decoded, so no field could be cross-checked "
                    "against the visual inspection zone. Recapture suggested.",
                    pin,
                )],
            )

        outcome = crosscheck_mrz_viz(mrz_result, viz_fields)
        checks_valid = all(c.valid for c in mrz_result.parsed.checks)

        signals: list[Signal] = []
        table: list[CrossCheckRow] = []
        mismatch_rows: list[MismatchRow] = []
        for row in outcome.rows:
            table.append(CrossCheckRow(
                field=row.field, mrz_value=row.mrz_value or "", viz_value=row.viz_value or "",
                status=CrossCheckStatus(row.status),
            ))
            field_code = row.field.upper()
            if row.status == "mismatch":
                signals.append(_signal(
                    f"MRZ_VIZ_MISMATCH_{field_code}",
                    f"The {row.field.replace('_', ' ')} printed in the visual inspection zone does not "
                    f"match the value read from the machine-readable zone. Recapture suggested.",
                    pin,
                ))
                mismatch_rows.append(MismatchRow(field=row.field, mrz=row.mrz_value or "", viz=row.viz_value or ""))
            elif row.status == "unreadable":
                signals.append(_signal(
                    f"MRZ_VIZ_FIELD_UNREADABLE_{field_code}",
                    f"The {row.field.replace('_', ' ')} field could not be read on both sides to cross-check.",
                    pin,
                ))

        passed = checks_valid and not any(r.status == "mismatch" for r in outcome.rows)
        return StageResult(
            state=StageState.PASSED if passed else StageState.FAILED,
            detail=(
                None if passed else
                "RECAPTURE_SUGGESTED: the printed and machine-readable data disagree, or a check digit "
                "failed to validate. Glare, blur and folds cause this far more often than a genuine "
                "document defect; recapture and re-run before treating this as anything else."
            ),
            signals=signals,
            artefacts={"cross_check_table": table, "mrz_viz_tolerance": DEFAULT_THRESHOLDS.mrz_viz_field_tolerance},
            mismatch_rows=mismatch_rows or None,
        )
