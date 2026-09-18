"""Full B1 vertical slice: capture -> Wave 1 -> Gate 1 -> findings, via
app.pipeline.analyse.run_analysis, plus the "no scalar score anywhere"
acceptance check on the wire schema.
"""
from __future__ import annotations

import inspect

import pytest

from app.api import schemas
from app.api.schemas import StageState
from app.ocr.mrz import synth
from app.pipeline.analyse import run_analysis
from app.pipeline.registry import StageId

from tests.ocr.conftest import build_document, render_viz_zone


def _record(seed: int) -> synth.SyntheticRecord:
    return synth.build_td3_record(
        surname="MARTIN", given_names="ISABELLE",
        doc_number=f"X{seed:06d}Y"[:9], nationality="UTO", issuing_country="UTO",
        birth_raw="850315", expiry_raw="301231", sex="F",
    )


@pytest.mark.asyncio
async def test_genuine_document_clears_through_gate1():
    record = _record(5)
    viz_zone = render_viz_zone(record)
    document = build_document(viz_zone, seed=5)

    result = await run_analysis("run-5", document, mrz_ground_truth=record.lines)

    assert result.mrz is not None
    assert result.cross_check
    # Every registered stage has a coverage entry, including the four
    # unavailable Wave 2 stages and gate-2/decision.
    assert len(result.coverage) == len(list(StageId))
    unavailable_modalities = {c.modality for c in result.coverage if c.unavailable}
    assert {"tamper", "ovd", "face", "identity-graph"} <= unavailable_modalities


@pytest.mark.asyncio
async def test_gate1_failure_still_produces_findings_and_recapture_reason():
    record = _record(6)
    from PIL import Image, ImageDraw
    import numpy as np

    viz_zone = render_viz_zone(record)
    document = build_document(viz_zone, seed=6)
    img = Image.fromarray(document[:, :, 0])
    draw = ImageDraw.Draw(img)
    draw.rectangle((30, 20, 400, 60), fill=255)
    draw.text((40, 30), "WRONGNAME ISABELLE", fill=0)
    document = np.stack([np.array(img)] * 3, axis=-1)

    result = await run_analysis("run-6", document, mrz_ground_truth=record.lines)

    assert result.findings  # Gate 1 failure signals still get grouped into findings
    assert "fraud" not in result.reason.lower()
    assert "forg" not in result.reason.lower()


def test_no_scalar_score_field_anywhere_in_the_wire_schema():
    banned_names = {"risk", "score", "confidence_score", "riskscore"}
    for name, model in vars(schemas).items():
        if not (inspect.isclass(model) and issubclass(model, schemas.BaseModel)):
            continue
        for field_name in model.model_fields:
            assert field_name.lower() not in banned_names, f"{name}.{field_name} looks like a scalar score"
