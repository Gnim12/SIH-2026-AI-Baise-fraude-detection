"""Checksum-constrained beam search decoder for MRZ lines.

The project's differentiator (BACKEND_BRIEF.md §1.3): a hypothesis in the
top-K beam whose check digit validates is `VERIFIED`; if no hypothesis in the
beam validates, the field is `UNRECOVERABLE` -- a high-severity tamper signal,
not a shrugged-off OCR error.

`model.py`'s CRNN emits per-timestep class log-probabilities and is trained
with CTC loss so the network doesn't need to be told the exact image-to-
character alignment. By the time a sequence reaches this module, CTC greedy
collapsing (removing blanks and repeated symbols) has already produced one
log-probability vector per *character slot* in the fixed-width MRZ line --
that per-slot distribution, not the raw per-timestep CTC output, is this
module's input. Beam search here explores *alternative characters per slot*,
not alternative alignments.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Optional

import numpy as np

from . import spec


class DecodeStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNRECOVERABLE = "UNRECOVERABLE"


@dataclass(frozen=True)
class GroupLayout:
    """Where one checksum-bearing field lives inside a multi-line MRZ."""

    name: str
    line: int
    field_start: int
    field_end: int  # exclusive, excludes the check digit
    check_digit_index: int


# The fields worth a dedicated constrained search: they carry the values the
# rest of the pipeline (MRZ<->VIZ cross-check) actually consumes. `composite`
# is left to spec.parse's own validation of the fully-assembled line, since
# its check-digit input is a discontinuous concatenation of several fields
# and isn't a useful unit to beam-search on its own.
LAYOUTS: dict[spec.MrzFormat, list[GroupLayout]] = {
    spec.MrzFormat.TD3: [
        GroupLayout("doc_number", 1, 0, 9, 9),
        GroupLayout("birth_date", 1, 13, 19, 19),
        GroupLayout("expiry_date", 1, 21, 27, 27),
    ],
    spec.MrzFormat.TD2: [
        GroupLayout("doc_number", 1, 0, 9, 9),
        GroupLayout("birth_date", 1, 13, 19, 19),
        GroupLayout("expiry_date", 1, 21, 27, 27),
    ],
    spec.MrzFormat.TD1: [
        GroupLayout("doc_number", 0, 5, 14, 14),
        GroupLayout("birth_date", 1, 0, 6, 6),
        GroupLayout("expiry_date", 1, 8, 14, 14),
    ],
}


@dataclass
class GroupDecodeResult:
    name: str
    text: str  # the field value, excluding the check digit
    check_char: str
    status: DecodeStatus
    logprob: float
    beam: list[tuple[str, float]] = dc_field(default_factory=list)


@dataclass
class LineDecodeResult:
    line: str
    line_logprob: float


@dataclass(frozen=True)
class RecoveredCharacter:
    """One position where the checksum-verified hypothesis differs from the
    network's greedy best path. Confidences are the network's own
    probabilities for each character at that slot -- nothing derived."""

    line: int
    position: int
    raw: str
    recovered: str
    raw_confidence: float
    recovered_confidence: float


@dataclass
class MrzDecodeResult:
    format: spec.MrzFormat
    lines: list[str]
    groups: dict[str, GroupDecodeResult]
    parsed: spec.MrzResult
    greedy_lines: list[str] = dc_field(default_factory=list)
    recovered: list[RecoveredCharacter] = dc_field(default_factory=list)
    # Per line, the network's probability for the character in the final
    # decoded line at each slot.
    char_confidences: list[list[float]] = dc_field(default_factory=list)

    @property
    def status(self) -> DecodeStatus:
        if any(g.status is DecodeStatus.UNRECOVERABLE for g in self.groups.values()):
            return DecodeStatus.UNRECOVERABLE
        if not self.parsed.all_valid:
            return DecodeStatus.UNRECOVERABLE
        return DecodeStatus.VERIFIED


def beam_search_positions(
    logprobs: np.ndarray, charset: str = spec.MRZ_CHARSET, beam_width: int = 8
) -> list[tuple[str, float]]:
    """Beam search over independent per-slot categorical distributions.

    `logprobs`: (L, C) array of log-probabilities over `charset` (len C),
    one row per character slot. Returns up to `beam_width` (string, total
    log-probability) pairs, sorted by descending score.
    """
    if logprobs.shape[1] != len(charset):
        raise ValueError(
            f"logprobs has {logprobs.shape[1]} classes, charset has {len(charset)}"
        )
    beams: list[tuple[str, float]] = [("", 0.0)]
    local_k = min(beam_width, len(charset))
    for t in range(logprobs.shape[0]):
        order = np.argsort(-logprobs[t])[:local_k]
        candidates = [
            (prefix + charset[idx], score + float(logprobs[t, idx]))
            for prefix, score in beams
            for idx in order
        ]
        candidates.sort(key=lambda c: -c[1])
        beams = candidates[:beam_width]
    return beams


def decode_group(
    logprobs_slice: np.ndarray,
    field_len: int,
    *,
    charset: str = spec.MRZ_CHARSET,
    beam_width: int = 16,
) -> GroupDecodeResult:
    """Constrained decode for one checksum-bearing field.

    `logprobs_slice` covers `field_len + 1` character slots: the field itself
    followed by its check digit. Among the beam, the highest-scoring
    hypothesis whose check digit is internally self-consistent wins
    (`VERIFIED`); if none is self-consistent, the single highest-scoring raw
    hypothesis is returned, flagged `UNRECOVERABLE`.
    """
    if logprobs_slice.shape[0] != field_len + 1:
        raise ValueError(
            f"expected {field_len + 1} slots (field + check digit), got {logprobs_slice.shape[0]}"
        )
    beams = beam_search_positions(logprobs_slice, charset, beam_width)
    for text, score in beams:  # beams sorted desc -> first match is best
        candidate_field, candidate_check = text[:field_len], text[field_len]
        if spec.verify_check_digit(candidate_field, candidate_check):
            return GroupDecodeResult(
                name="", text=candidate_field, check_char=candidate_check,
                status=DecodeStatus.VERIFIED, logprob=score, beam=beams,
            )
    best_text, best_score = beams[0]
    return GroupDecodeResult(
        name="", text=best_text[:field_len], check_char=best_text[field_len],
        status=DecodeStatus.UNRECOVERABLE, logprob=best_score, beam=beams,
    )


def decode_mrz(logprobs_lines: list[np.ndarray], fmt: spec.MrzFormat) -> MrzDecodeResult:
    """Decode a full MRZ (all lines) with checksum-constrained search on the
    fields that carry check digits, and plain best-path decoding elsewhere.

    `logprobs_lines[i]`: (line_length, len(charset)) log-probabilities for line i.
    """
    n_lines, line_len = spec.LINE_SHAPE[fmt]
    if len(logprobs_lines) != n_lines:
        raise ValueError(f"{fmt} needs {n_lines} lines, got {len(logprobs_lines)}")
    for lp in logprobs_lines:
        if lp.shape[0] != line_len:
            raise ValueError(f"{fmt} lines must be {line_len} slots, got {lp.shape[0]}")

    # Plain best-path per line as the baseline, then patch in the constrained groups.
    baseline_lines = [
        beam_search_positions(lp, beam_width=1)[0][0] for lp in logprobs_lines
    ]
    line_chars = [list(l) for l in baseline_lines]

    groups: dict[str, GroupDecodeResult] = {}
    recovered: list[RecoveredCharacter] = []
    index_of = {c: i for i, c in enumerate(spec.MRZ_CHARSET)}
    for layout in LAYOUTS[fmt]:
        span_len = layout.check_digit_index - layout.field_start + 1
        lp_slice = logprobs_lines[layout.line][layout.field_start : layout.field_start + span_len]
        result = decode_group(lp_slice, layout.field_end - layout.field_start)
        result.name = layout.name
        groups[layout.name] = result
        patched = result.text + result.check_char
        if result.status is DecodeStatus.VERIFIED:
            for offset, ch in enumerate(patched):
                pos = layout.field_start + offset
                raw_ch = baseline_lines[layout.line][pos]
                if raw_ch == ch:
                    continue
                row = logprobs_lines[layout.line][pos]
                recovered.append(RecoveredCharacter(
                    line=layout.line, position=pos, raw=raw_ch, recovered=ch,
                    raw_confidence=float(np.exp(row[index_of[raw_ch]])),
                    recovered_confidence=float(np.exp(row[index_of[ch]])),
                ))
        line_chars[layout.line][layout.field_start : layout.check_digit_index + 1] = list(patched)

    final_lines = ["".join(chars) for chars in line_chars]
    parsed = spec.parse(final_lines)
    char_confidences = [
        [float(np.exp(lp[i, index_of[ch]])) for i, ch in enumerate(line)]
        for lp, line in zip(logprobs_lines, final_lines)
    ]
    recovered.sort(key=lambda r: (r.line, r.position))
    return MrzDecodeResult(
        format=fmt, lines=final_lines, groups=groups, parsed=parsed,
        greedy_lines=baseline_lines, recovered=recovered, char_confidences=char_confidences,
    )


def fake_logprobs(
    ground_truth: str,
    *,
    charset: str = spec.MRZ_CHARSET,
    confidence: float = 0.9,
    confusable_pairs: Optional[list[tuple[str, str]]] = None,
    error_positions: Optional[dict[int, str]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Synthesize a (len(ground_truth), len(charset)) log-probability matrix
    peaked at `ground_truth`, for exercising the decoder without a trained
    model (stub mode -- see infer.py). Not a substitute for real CRNN output.

    `error_positions`: optional {index: wrong_char} to force specific
    misreads (e.g. simulate a 0/O confusion at a known position) so tests can
    construct exact CHECKSUM_UNRECOVERABLE or checksum-recoverable scenarios.
    `confusable_pairs`: extra probability mass shared between visually
    similar characters (defaults to the classic OCR-B confusions).
    """
    rng = random.Random(seed)
    confusable_pairs = confusable_pairs or [
        ("0", "O"), ("1", "I"), ("5", "S"), ("8", "B"), ("2", "Z"),
    ]
    confusion_of: dict[str, str] = {}
    for a, b in confusable_pairs:
        confusion_of[a] = b
        confusion_of[b] = a

    error_positions = error_positions or {}
    n_classes = len(charset)
    index_of = {c: i for i, c in enumerate(charset)}

    probs = np.full((len(ground_truth), n_classes), (1 - confidence) / max(n_classes - 1, 1), dtype=np.float64)
    for t, true_char in enumerate(ground_truth):
        shown_char = error_positions.get(t, true_char)
        true_idx = index_of[shown_char]
        probs[t, :] = (1 - confidence) / max(n_classes - 1, 1)
        probs[t, true_idx] = confidence
        confusable = confusion_of.get(shown_char)
        if confusable is not None and confusable in index_of:
            # steal a slice of probability mass for the visually-similar character
            steal = min(confidence * 0.5, probs[t, true_idx] * 0.4)
            probs[t, true_idx] -= steal
            probs[t, index_of[confusable]] += steal
        jitter = rng.uniform(-0.02, 0.02) * confidence
        probs[t, true_idx] = max(1e-6, probs[t, true_idx] + jitter)
        probs[t] = probs[t] / probs[t].sum()

    return np.log(np.clip(probs, 1e-12, 1.0))


if __name__ == "__main__":
    icao_l1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    icao_l2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"

    # Clean, high-confidence input decodes to VERIFIED everywhere.
    lp1 = fake_logprobs(icao_l1, confidence=0.97, seed=1)
    lp2 = fake_logprobs(icao_l2, confidence=0.97, seed=2)
    result = decode_mrz([lp1, lp2], spec.MrzFormat.TD3)
    assert result.status is DecodeStatus.VERIFIED, result.status
    assert result.parsed.doc_number == "L898902C3"
    print("decode.py: clean input -> VERIFIED,", result.parsed.doc_number, result.parsed.surname)

    # A classic OCR-B confusion (0 misread as O) inside the expiry-date span:
    # the checksum-constrained beam has residual probability mass on the true
    # '0' via the confusable-pair boost, so it should recover it even though
    # the raw top-1 read at that position is the wrong character.
    lp2_confusable = fake_logprobs(
        icao_l2, confidence=0.6, seed=3, error_positions={23: "O"}  # expiry "120415" pos 2: '0'->'O'
    )
    recovered = decode_mrz([lp1, lp2_confusable], spec.MrzFormat.TD3)
    expiry_group = recovered.groups["expiry_date"]
    assert expiry_group.status is DecodeStatus.VERIFIED, expiry_group
    assert expiry_group.text == "120415", expiry_group.text
    print("decode.py: 0/O confusion in expiry field -> recovered via checksum,",
          "status", expiry_group.status)

    # An arbitrary (non-confusable) high-confidence misread with no
    # plausible alternative in the beam cannot be checksum-recovered --
    # this is the intended failure mode, not a bug.
    lp2_unrelated = fake_logprobs(
        icao_l2, confidence=0.9, seed=4, error_positions={14: "5"}  # birth field pos 1: '4'->'5', unrelated char
    )
    unrecoverable = decode_mrz([lp1, lp2_unrelated], spec.MrzFormat.TD3)
    assert unrecoverable.groups["birth_date"].status is DecodeStatus.UNRECOVERABLE
    assert unrecoverable.status is DecodeStatus.UNRECOVERABLE
    print("decode.py: unrelated high-confidence misread -> UNRECOVERABLE, as intended")

    print("decode.py self-test OK")
