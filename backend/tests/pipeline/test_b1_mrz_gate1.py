"""Real stage-level tests for mrz-read, viz-read and gate-1 (Milestone B1
spec section 10, 'MRZ' and 'Gate 1'), using the same synthetic-document
fixtures as tests/ocr/test_reader.py.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from app.api.schemas import MrzCharRole, StageState
from app.ocr.mrz import decode, detect, spec, synth
from app.ocr.mrz.infer import MRZReader
from app.pipeline.registry import StageId
from app.pipeline.runner import StageContext
from app.pipeline.stages.gate1 import Gate1Stage
from app.pipeline.stages.mrz_read import MrzReadStage, _build_ribbon
from app.pipeline.stages.viz_read import VizReadStage

from tests.ocr.conftest import build_document, render_viz_zone


def _record(seed: int) -> synth.SyntheticRecord:
    return synth.build_td3_record(
        surname="MARTIN", given_names="ISABELLE",
        doc_number=f"X{seed:06d}Y"[:9], nationality="UTO", issuing_country="UTO",
        birth_raw="850315", expiry_raw="301231", sex="F",
    )


@pytest.mark.asyncio
async def test_clean_td3_decodes_all_fields_with_valid_check_digits():
    record = _record(1)
    viz_zone = render_viz_zone(record)
    document = build_document(viz_zone, seed=1)

    ctx = StageContext(run_id="mrz1", inputs={"document_image": document, "mrz_ground_truth": record.lines})
    result = await MrzReadStage().run(ctx)

    assert result.state == StageState.PASSED, result.detail
    assert not any(s.signal_id.startswith("MRZ_CHECKSUM") for s in result.signals)
    assert not any(s.signal_id == "MRZ_COMPOSITE_CHECKSUM_FAIL" for s in result.signals)
    ribbon = result.artefacts["mrz_schema"]
    assert ribbon.fields
    assert all(f.check_digit_state.value != "invalid" for f in ribbon.fields)


@pytest.mark.asyncio
async def test_missing_band_reports_signal_without_crashing():
    blank = np.full((200, 400, 3), 255, dtype=np.uint8)
    ctx = StageContext(run_id="mrz2", inputs={"document_image": blank, "mrz_ground_truth": ["A" * 44, "A" * 44]})

    result = await MrzReadStage().run(ctx)

    assert result.state == StageState.FAILED
    assert any(s.signal_id == "MRZ_BAND_NOT_FOUND" for s in result.signals)


def test_recovered_character_reported_in_ribbon():
    """Mirrors decode.py's own self-test: a 0/O confusion inside a
    checksum-bearing field that the checksum-constrained beam recovers must
    show up as a 'recovered' character in the ribbon, with the raw network
    reading and the recovered value in its detail."""
    record = _record(9)
    viz_zone = render_viz_zone(record)
    document = build_document(viz_zone, seed=9)
    band = detect.find_mrz(document, n_lines=2)
    assert band is not None

    lp1 = decode.fake_logprobs(record.lines[0], confidence=0.95, seed=1)
    lp2 = decode.fake_logprobs(record.lines[1], confidence=0.6, seed=3, error_positions={23: "O"})
    decoded = decode.decode_mrz([lp1, lp2], spec.MrzFormat.TD3)
    assert decoded.groups["expiry_date"].status is decode.DecodeStatus.VERIFIED

    reader = MRZReader(require_trained_weights=False)
    result = reader._to_result(decoded, band)

    ribbon = _build_ribbon(result, pin="mrz_crnn test")
    recovered = [c for line in ribbon.lines for c in line if c.role == MrzCharRole.RECOVERED]
    assert recovered, "expected at least one recovered character in the ribbon"
    assert all(c.recovery_detail for c in recovered)


async def _run_ocr_pair(document, ground_truth, document_id="doc"):
    mrz_ctx = StageContext(run_id="g1", inputs={"document_image": document, "mrz_ground_truth": ground_truth})
    mrz_result = await MrzReadStage().run(mrz_ctx)
    viz_ctx = StageContext(run_id="g1", inputs={"document_image": document, "document_id": document_id})
    viz_result = await VizReadStage().run(viz_ctx)
    return mrz_result, viz_result


@pytest.mark.asyncio
async def test_gate1_passes_when_mrz_and_viz_match():
    record = _record(2)
    viz_zone = render_viz_zone(record)
    document = build_document(viz_zone, seed=2)
    mrz_result, viz_result = await _run_ocr_pair(document, record.lines)

    gate_ctx = StageContext(
        run_id="g1",
        artefacts={StageId.MRZ_READ: mrz_result.artefacts, StageId.VIZ_READ: viz_result.artefacts},
    )
    result = await Gate1Stage().run(gate_ctx)

    assert result.state == StageState.PASSED, result.detail
    assert not any(s.signal_id.startswith("MRZ_VIZ_MISMATCH") for s in result.signals)
    assert result.artefacts["cross_check_table"]  # field comparison table present


@pytest.mark.asyncio
async def test_gate1_fails_on_altered_viz_surname():
    record = _record(3)
    # Printed surname diverges from the MRZ; MRZ itself is untouched.
    viz_zone = render_viz_zone(record, given_names_display="ISABELLE")
    document = build_document(viz_zone, seed=3)
    # Force the surname mismatch by rendering a different name line directly.
    from PIL import Image, ImageDraw

    img = Image.fromarray(document[:, :, 0])
    draw = ImageDraw.Draw(img)
    draw.rectangle((30, 20, 400, 60), fill=255)
    draw.text((40, 30), "WRONGNAME ISABELLE", fill=0)
    document = np.stack([np.array(img)] * 3, axis=-1)

    mrz_result, viz_result = await _run_ocr_pair(document, record.lines)
    gate_ctx = StageContext(
        run_id="g2",
        artefacts={StageId.MRZ_READ: mrz_result.artefacts, StageId.VIZ_READ: viz_result.artefacts},
    )
    result = await Gate1Stage().run(gate_ctx)

    assert result.state == StageState.FAILED
    assert any(s.signal_id == "MRZ_VIZ_MISMATCH_SURNAME" for s in result.signals)

    # No signal or detail string may call the document or traveller fraudulent.
    banned = ("fraud", "forg", "fake")
    assert result.detail is not None
    assert not any(word in result.detail.lower() for word in banned)
    for s in result.signals:
        assert not any(word in s.detail.lower() for word in banned)
