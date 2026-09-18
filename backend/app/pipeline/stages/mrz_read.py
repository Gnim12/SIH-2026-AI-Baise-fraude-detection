"""mrz-read stage: adapter over app/ocr/mrz (detect -> CRNN -> checksum-
constrained decode), via the already-shared MRZReader instance in
app/pipeline/branches/ocr.py. No MRZ logic is reimplemented here.

Emits: the parsed field set, per-field check-digit validity, the composite
check digit, the character-level ribbon data (MrzResult.lines) with
recovered characters flagged, and the standard MRZ signal set.
"""
from __future__ import annotations

import asyncio
import datetime

from app.api.schemas import (
    CheckDigitState, Modality, MrzCharacter, MrzCharRole, MrzFieldRow, MrzResult, Signal, StageState,
)
from app.ocr.mrz.infer import MrzReadResult
from app.pipeline.branches.ocr import _shared_mrz_reader  # reuse: one ONNX session per process
from app.pipeline.pins import resolve_pins
from app.pipeline.registry import StageId
from app.pipeline.runner import StageContext, StageResult

LOW_CONFIDENCE_THRESHOLD = 0.6


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _signal(code: str, detail: str, pin: str) -> Signal:
    return Signal(
        signal_id=code, modality=Modality.MRZ, label=code.replace("_", " ").title(),
        detail=detail, emitted_at=_now_iso(), model_pin=pin,
    )


def _line_signal_code(line_index: int) -> str:
    return "MRZ_CHECKSUM_FAIL_LINE1" if line_index == 0 else "MRZ_CHECKSUM_FAIL_LINE2"


def _build_ribbon(result: MrzReadResult, pin: str) -> MrzResult:
    parsed = result.parsed
    assert parsed is not None  # caller only reaches here once a band was decoded
    checks_by_name = {g.name: g for g in parsed.checks}
    decode_groups = result.groups  # name -> decode.GroupDecodeResult, has .beam

    lines: list[list[MrzCharacter]] = []
    for line_idx, text in enumerate(parsed.lines):
        chars = [
            MrzCharacter(char=c, index=i, role=MrzCharRole.NORMAL)
            for i, c in enumerate(text)
        ]
        for check in parsed.checks:
            if check.line != line_idx:
                continue
            chars[check.check_digit_index] = MrzCharacter(
                char=text[check.check_digit_index], index=check.check_digit_index,
                role=MrzCharRole.CHECK_DIGIT,
            )
            if not check.valid:
                for i in range(check.start, check.end):
                    chars[i] = MrzCharacter(
                        char=text[i], index=i, role=MrzCharRole.FAILED_SIGNAL,
                        recovery_detail=None,
                    )
                continue
            # Valid group: flag positions the checksum-constrained beam
            # changed from the network's raw top-1 hypothesis.
            group = decode_groups.get(check.name)
            if group is None or not group.beam:
                continue
            raw_top = group.beam[0][0]
            selected = group.text + group.check_char
            for offset in range(min(len(raw_top), len(selected))):
                if raw_top[offset] == selected[offset]:
                    continue
                pos = check.start + offset
                if pos >= check.end:
                    continue  # check-digit slot handled above
                chars[pos] = MrzCharacter(
                    char=selected[offset], index=pos, role=MrzCharRole.RECOVERED,
                    recovery_detail=(
                        f"Recovered by ICAO check digit · network read "
                        f"{raw_top[offset]!r}, checksum-constrained decode selected {selected[offset]!r}"
                    ),
                )
        lines.append(chars)

    fields: list[MrzFieldRow] = []
    for name in ("doc_number", "birth_date", "expiry_date"):
        field_check = checks_by_name.get(name)
        mf = result.fields.get(name)
        if mf is None:
            continue
        state = CheckDigitState.NONE if field_check is None else (
            CheckDigitState.VALID if field_check.valid else CheckDigitState.INVALID
        )
        fields.append(MrzFieldRow(field=name, value=mf.value, check_digit_state=state))

    composite = checks_by_name.get("composite")
    composite_state: dict[str, str] = {
        "value": (composite.read or "") if composite else "",
        "state": (
            (CheckDigitState.VALID if composite.valid else CheckDigitState.INVALID) if composite else CheckDigitState.NONE
        ).value,
    }

    return MrzResult(
        format=parsed.format.value, line_length=len(parsed.lines[0]) if parsed.lines else 0,
        lines=lines, fields=fields, composite_check_digit=composite_state, model_pin=pin,
    )


class MrzReadStage:
    id = StageId.MRZ_READ

    async def run(self, ctx: StageContext) -> StageResult:
        pin = resolve_pins().get("mrz_crnn", "mrz_crnn unknown")
        reader = _shared_mrz_reader()
        image = ctx.inputs["document_image"]
        ground_truth = ctx.inputs.get("mrz_ground_truth")

        def _read() -> MrzReadResult:
            return reader.read(image, stub_ground_truth=ground_truth)

        try:
            result = await asyncio.to_thread(_read)
        except ValueError:
            # Stub mode (no trained CRNN weights) and no ground truth was
            # supplied: the band exists but cannot be decoded. Real coverage
            # gap, not a crash -- see app/ocr/mrz/infer.py's module docstring.
            return StageResult(
                state=StageState.FAILED,
                detail="MRZ band located but no trained model and no ground truth were available to decode it.",
                signals=[_signal(
                    "MRZ_LOW_CONFIDENCE",
                    "No trained CRNN checkpoint exists yet and no ground truth was supplied "
                    "for this capture; the MRZ could not be decoded.",
                    pin,
                )],
            )

        if result.band is None or result.parsed is None:
            return StageResult(
                state=StageState.FAILED,
                detail="No machine-readable zone could be located on this document.",
                signals=[_signal(
                    "MRZ_BAND_NOT_FOUND", "No machine-readable zone could be located on this document.", pin,
                )],
            )

        signals: list[Signal] = []
        for check in result.parsed.checks:
            if check.valid:
                continue
            code = "MRZ_COMPOSITE_CHECKSUM_FAIL" if check.name == "composite" else _line_signal_code(check.line)
            signals.append(_signal(
                code,
                f"MRZ {check.name.replace('_', ' ')} check digit expected {check.expected!r}, read {check.read!r}.",
                pin,
            ))

        confidences = [f.confidence for f in result.fields.values()]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        if avg_conf < LOW_CONFIDENCE_THRESHOLD:
            signals.append(_signal(
                "MRZ_LOW_CONFIDENCE", f"Average MRZ field confidence {avg_conf:.2f} is below threshold.", pin,
            ))

        ribbon = _build_ribbon(result, pin)
        state = StageState.FAILED if any(not c.valid for c in result.parsed.checks) else StageState.PASSED

        return StageResult(
            state=state,
            detail=None if state == StageState.PASSED else "One or more MRZ check digits failed to validate.",
            signals=signals,
            artefacts={"mrz_read_result": result, "mrz_schema": ribbon},
        )
