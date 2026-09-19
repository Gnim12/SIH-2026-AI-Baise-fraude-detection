"""mrz-read stage: adapter over app/ocr/mrz (detect -> normalize_line -> ONNX
-> checksum-constrained decode), via the process-wide reader resolved in
app/ocr/mrz/runtime.py. No MRZ logic is reimplemented here.

Emits: the parsed field set, per-field check-digit validity, the composite
check digit, the character-level ribbon data (MrzResult.lines) with the
recovered characters, and the MRZ signal set. A field guarded by a check digit
that did not validate is absent, with a signal saying why -- never a guess.

When the model is unavailable (missing, placeholder, hash mismatch) the stage
settles `unavailable`: not failed, not passed.
"""
from __future__ import annotations

import asyncio
import datetime

from app.api.schemas import (
    CheckDigitState, Modality, MrzCharacter, MrzCharRole, MrzFieldRow, MrzResult, RecoveredCharacter,
    RecoveryStats, Signal, StageState,
)
from app.ocr.mrz import decode
from app.ocr.mrz.infer import MrzReadResult
from app.ocr.mrz.runtime import get_mrz_runtime
from app.pipeline.registry import StageId
from app.pipeline.runner import StageContext, StageResult
from app.pipeline.thresholds import DEFAULT_THRESHOLDS


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _signal(code: str, detail: str, pin: str) -> Signal:
    return Signal(
        signal_id=code, modality=Modality.MRZ, label=code.replace("_", " ").title(),
        detail=detail, emitted_at=_now_iso(), model_pin=pin,
    )


def _line_signal_code(line_index: int) -> str:
    return "MRZ_CHECKSUM_FAIL_LINE1" if line_index == 0 else "MRZ_CHECKSUM_FAIL_LINE2"


def _recovered_payload(result: MrzReadResult) -> list[RecoveredCharacter]:
    margin = DEFAULT_THRESHOLDS.mrz_recovery_confidence_margin
    return [
        RecoveredCharacter(
            line_index=rc.line, position=rc.position, raw=rc.raw, recovered=rc.recovered,
            raw_confidence=rc.raw_confidence, recovered_confidence=rc.recovered_confidence,
            suspect=(rc.raw_confidence - rc.recovered_confidence) > margin,
        )
        for rc in result.recovered
    ]


def _recovery_stats(result: MrzReadResult, recovered: list[RecoveredCharacter]) -> RecoveryStats:
    n = len(recovered)
    return RecoveryStats(
        total_characters=sum(len(line) for line in (result.parsed.lines if result.parsed else [])),
        recovered=n,
        suspect=sum(1 for r in recovered if r.suspect),
        mean_raw_confidence=sum(r.raw_confidence for r in recovered) / n if n else None,
        mean_recovered_confidence=sum(r.recovered_confidence for r in recovered) / n if n else None,
    )


def _build_ribbon(result: MrzReadResult, pin: str) -> MrzResult:
    parsed = result.parsed
    assert parsed is not None  # caller only reaches here once a band was decoded
    recovered_payload = _recovered_payload(result)
    checks_by_name = {g.name: g for g in parsed.checks}

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
        # Recovered characters: positions where the checksum-verified reading
        # differs from the network's greedy path. Nothing recovered, nothing
        # tinted.
        for rc in result.recovered:
            if rc.line != line_idx:
                continue
            chars[rc.position] = MrzCharacter(
                char=rc.recovered, index=rc.position, role=MrzCharRole.RECOVERED,
                recovery_detail=(
                    f"Recovered by ICAO check digit · network favoured {rc.raw!r} ({rc.raw_confidence:.0%}), "
                    f"verified reading {rc.recovered!r} ({rc.recovered_confidence:.0%})"
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
        recovered_characters=recovered_payload,
        recovery_stats=_recovery_stats(result, recovered_payload),
    )


class MrzReadStage:
    id = StageId.MRZ_READ

    async def run(self, ctx: StageContext) -> StageResult:
        runtime = get_mrz_runtime()
        reader = runtime.reader
        if reader is None:
            # Missing / placeholder / hash-mismatched model. `unavailable`,
            # not failed and not passed: a check that cannot run is not a
            # check that ran. No signal is emitted -- there is no model to
            # attribute one to.
            return StageResult(
                state=StageState.UNAVAILABLE,
                detail=f"MRZ reader unavailable: {runtime.status.reason}",
                artefacts={"unavailable": True},
            )

        pin = reader.pin
        image = ctx.inputs["document_image"]
        # Ground truth is honoured only by a stub-mode reader, which exists
        # only in tests. A real reader is never handed it.
        ground_truth = ctx.inputs.get("mrz_ground_truth") if reader.stub_mode else None

        def _read() -> MrzReadResult:
            return reader.read(image, stub_ground_truth=ground_truth)

        try:
            result = await asyncio.to_thread(_read)
        except ValueError:
            if not reader.stub_mode:
                raise
            # Stub reader (tests only) with no ground truth to decode against.
            return StageResult(
                state=StageState.FAILED,
                detail="Stub MRZ reader was given no ground truth to decode against.",
                signals=[_signal("MRZ_LOW_CONFIDENCE", "Stub MRZ reader had no ground truth.", pin)],
            )

        if result.band is None or result.parsed is None:
            detail = result.band_failure_detail or "No machine-readable zone could be located on this document."
            return StageResult(
                state=StageState.FAILED, detail=detail,
                signals=[_signal("MRZ_BAND_NOT_FOUND", detail, pin)],
            )

        signals: list[Signal] = []
        failed_by_code: dict[str, list[str]] = {}
        for check in result.parsed.checks:
            if check.valid:
                continue
            code = "MRZ_CHECKSUM_FAIL_COMPOSITE" if check.name == "composite" else _line_signal_code(check.line)
            failed_by_code.setdefault(code, []).append(
                f"{check.name.replace('_', ' ')} check digit expected {check.expected!r}, read {check.read!r}"
            )
        for code, details in failed_by_code.items():
            signals.append(_signal(
                code, "MRZ " + "; ".join(details) + ". Fields guarded by a failed check digit are not reported.", pin,
            ))

        if result.status is decode.DecodeStatus.UNRECOVERABLE:
            unrecoverable = [
                n.replace("_", " ") for n, g in result.groups.items()
                if g.status is decode.DecodeStatus.UNRECOVERABLE
            ]
            signals.append(_signal(
                "MRZ_DECODE_UNRECOVERABLE",
                "The checksum-constrained decoder could not produce a valid reading"
                + (f" for: {', '.join(unrecoverable)}." if unrecoverable else "."),
                pin,
            ))

        mean_conf = result.mean_confidence
        if mean_conf is not None and mean_conf < DEFAULT_THRESHOLDS.mrz_low_confidence_mean:
            signals.append(_signal(
                "MRZ_LOW_CONFIDENCE",
                f"Mean per-character confidence {mean_conf:.2f} is below the "
                f"{DEFAULT_THRESHOLDS.mrz_low_confidence_mean:.2f} threshold.",
                pin,
            ))

        ribbon = _build_ribbon(result, pin)
        suspects = [r for r in ribbon.recovered_characters if r.suspect]
        if suspects:
            where = "; ".join(
                f"line {r.line_index + 1} position {r.position} (network read {r.raw!r} at {r.raw_confidence:.0%}, "
                f"resolved to {r.recovered!r} at {r.recovered_confidence:.0%})"
                for r in suspects
            )
            signals.append(_signal(
                "MRZ_RECOVERY_SUSPECT",
                f"A confident network reading was overridden by the check digit at: {where}. Either the network "
                "is wrong or the band was misdetected and the line is unreliable; the recovery is reported, not trusted.",
                pin,
            ))
        state = StageState.FAILED if any(not c.valid for c in result.parsed.checks) else StageState.PASSED

        return StageResult(
            state=state,
            detail=None if state == StageState.PASSED else "One or more MRZ check digits failed to validate.",
            signals=signals,
            artefacts={"mrz_read_result": result, "mrz_schema": ribbon},
        )
