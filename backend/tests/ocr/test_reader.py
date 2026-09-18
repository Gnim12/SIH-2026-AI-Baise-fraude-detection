"""OCRReader integration tests -- stub mode, no trained MRZ weights required.

Covers BACKEND_BRIEF.md §6.2 step 5 (MRZ<->VIZ cross-check) and §1.3
(CHECKSUM_UNRECOVERABLE as a surfaced fraud signal, not a buried decode
status), using real RapidOCR (locally cached, hash-verified ONNX weights,
see app/ocr/viz.py) for the VIZ side and MRZReader's stub mode for the MRZ
side.
"""
from __future__ import annotations

import random

import pytest

from app.ocr.mrz import decode, spec, synth
from app.ocr.mrz.infer import MissingWeightsError, MRZReader
from app.ocr.reader import OCRReader
from app.ocr.viz import VizReader

from .conftest import build_document, render_viz_zone

BIRTH_LAYOUT = next(g for g in decode.LAYOUTS[spec.MrzFormat.TD3] if g.name == "birth_date")


def _make_record(seed: int) -> synth.SyntheticRecord:
    rng = random.Random(seed)
    return synth.build_td3_record(
        surname="MARTIN", given_names="ISABELLE",
        doc_number=f"X{seed:06d}Y"[:9], nationality="UTO", issuing_country="UTO",
        birth_raw="850315", expiry_raw="301231", sex="F",
    )


def _tamper_check_digit(lines: list[str], layout: decode.GroupLayout) -> list[str]:
    """Flip one check digit to an incorrect value, leaving the underlying
    field digits untouched -- simulates a checksum that cannot be
    reconciled with any plausible reading, not an ordinary misread."""
    chars = list(lines[layout.line])
    original = chars[layout.check_digit_index]
    chars[layout.check_digit_index] = "0" if original != "0" else "1"
    new_lines = list(lines)
    new_lines[layout.line] = "".join(chars)
    return new_lines


def test_genuine_document_no_signals(ocr_reader: OCRReader):
    record = _make_record(seed=1)
    viz_zone = render_viz_zone(record)
    document = build_document(viz_zone, seed=1)

    result = ocr_reader.read(document, stub_mrz_ground_truth=record.lines)

    print("=== genuine document ===")
    for name, f in sorted(result.fields.items()):
        print(f"  field {name}: {f.value!r} (confidence={f.confidence:.2f})")
    print(f"  signals: {[s.code for s in result.signals]}")
    print(f"  overall confidence: {result.confidence:.2f}")

    assert result.mrz_present
    assert result.viz_field_count > 0
    assert result.fields["mrz.doc_number"].value == record.doc_number
    assert result.signals == [], [s.code for s in result.signals]


def test_altered_viz_birth_date_triggers_mismatch(ocr_reader: OCRReader):
    record = _make_record(seed=2)
    # MRZ ground truth is untouched (checksum-valid); only the *printed* VIZ
    # date of birth is wrong.
    viz_zone = render_viz_zone(record, birth_display_override="01/01/1999")
    document = build_document(viz_zone, seed=2)

    result = ocr_reader.read(document, stub_mrz_ground_truth=record.lines)

    print("=== altered VIZ birth date ===")
    for name, f in sorted(result.fields.items()):
        print(f"  field {name}: {f.value!r} (confidence={f.confidence:.2f})")
    for s in result.signals:
        print(f"  signal {s.code} [{s.severity}]: {s.detail}")

    mismatches = [s for s in result.signals if s.code == "MRZ_VIZ_MISMATCH"]
    assert len(mismatches) == 1, result.signals
    assert mismatches[0].severity == "high"
    assert "birth date" in mismatches[0].detail
    assert not any(s.code == "CHECKSUM_UNRECOVERABLE" for s in result.signals)


def test_unrecoverable_mrz_checksum_coexists_with_viz_fields(ocr_reader: OCRReader):
    record = _make_record(seed=3)
    tampered_lines = _tamper_check_digit(record.lines, BIRTH_LAYOUT)
    viz_zone = render_viz_zone(record)  # VIZ is genuine and fully extractable
    document = build_document(viz_zone, seed=3)

    result = ocr_reader.read(document, stub_mrz_ground_truth=tampered_lines)

    print("=== unrecoverable MRZ checksum, genuine VIZ ===")
    for name, f in sorted(result.fields.items()):
        print(f"  field {name}: {f.value!r} (confidence={f.confidence:.2f})")
    for s in result.signals:
        print(f"  signal {s.code} [{s.severity}]: {s.detail}")

    assert any(s.code == "CHECKSUM_UNRECOVERABLE" for s in result.signals), result.signals
    # The pipeline must not crash, and VIZ extraction must still have run.
    assert result.viz_field_count > 0
    assert result.mrz_present


def test_require_trained_weights_default_raises():
    with pytest.raises(MissingWeightsError):
        MRZReader()


def test_require_trained_weights_false_does_not_raise():
    reader = MRZReader(require_trained_weights=False)
    assert reader.stub_mode is True
