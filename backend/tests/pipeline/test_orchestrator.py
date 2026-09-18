"""Orchestrator DAG tests (BACKEND_BRIEF.md §5.3/§5.4): real event ordering,
the real OCR branch producing real fields, and real Degraded/timeout handling
-- not stubbed out for the test."""
from __future__ import annotations

import random

import cv2
import pytest

from app.contracts import DecisionEvent, OcrEvent, ReceivedEvent
from app.ocr.mrz import synth
from app.pipeline import orchestrator
from app.pipeline.context import DocumentInput, PipelineContext
from tests.ocr.conftest import build_document, render_viz_zone


def _genuine_document_input(document_id: str = "doc-1") -> DocumentInput:
    rng = random.Random(7)
    record = synth.build_td3_record(
        surname="MARTIN", given_names="ISABELLE", doc_number="X0000007Y",
        nationality="UTO", issuing_country="UTO", birth_raw="850315",
        expiry_raw="301231", sex="F",
    )
    viz_zone = render_viz_zone(record)
    image = build_document(viz_zone, seed=7)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return DocumentInput(
        document_id=document_id, doc_type="PASSPORT", original_bytes=buf.tobytes(),
        rectified=image, mrz_ground_truth=record.lines,
    )


@pytest.fixture()
def ctx() -> PipelineContext:
    return PipelineContext(
        session_id="test-session", lane_id="LANE-1", officer_id="OFF-1",
        model_versions={"rapidocr_det": "v5"}, mrz_require_trained_weights=False,
    )


@pytest.fixture(autouse=True)
def _generous_ocr_budget(monkeypatch):
    # BACKEND_BRIEF.md §5.3's BUDGET_MS values are production targets on
    # tuned hardware. This dev sandbox's cold ONNX Runtime session load (the
    # RapidOCR det/rec/cls models) plus CPU inference genuinely exceeds
    # 700ms, which would make every test here exercise the timeout path
    # instead of the real branch -- defeating the point of these tests.
    # Warm the shared singletons once, outside any timed `run()` call, and
    # widen the OCR budget for the timing-sensitive tests below; the branch's
    # own logic and Degraded/timeout wiring stay exactly as shipped.
    from app.pipeline.branches.ocr import _shared_mrz_reader, _shared_viz_reader

    _shared_viz_reader()
    _shared_mrz_reader()
    monkeypatch.setitem(orchestrator.BUDGET_MS, "ocr", 8000)


@pytest.mark.asyncio
async def test_run_publishes_events_in_order_ending_in_decision(ctx: PipelineContext):
    inputs = orchestrator.PipelineInputs(
        session_id=ctx.session_id, lane_id=ctx.lane_id, officer_id=ctx.officer_id,
        documents=[_genuine_document_input()],
    )
    session = await orchestrator.run(inputs, ctx)

    bus = orchestrator.event_bus.get(ctx.session_id)
    stages = [e.stage for e in bus.replay]
    assert stages[0] == "received"
    assert "quality" in stages
    assert "classified" in stages
    assert "ocr" in stages
    assert "forensics" in stages
    assert "database" in stages
    assert "crossdoc" in stages
    assert stages[-1] == "decision"

    assert isinstance(bus.replay[0], ReceivedEvent)
    ocr_events = [e for e in bus.replay if isinstance(e, OcrEvent)]
    assert len(ocr_events) == 1
    # Real extracted fields, not hardcoded case-01 fixture data.
    ocr_fields_by_key = {f.key: f.value for f in ocr_events[0].fields}
    assert ocr_fields_by_key.get("doc_number") == "X0000007Y"
    assert ocr_events[0].mrz is not None
    assert ocr_events[0].mrz.status == "VERIFIED"

    decision = [e for e in bus.replay if isinstance(e, DecisionEvent)][0]
    assert decision.band == "CLEAR"
    assert session.band == "CLEAR"
    assert session.documents[0].fields  # populated from the real OCR branch


@pytest.mark.asyncio
async def test_degraded_branch_on_timeout_produces_coverage_gap(ctx: PipelineContext, monkeypatch):
    # `ocr` is the first branch wave_1 awaits, and real inference genuinely
    # takes non-trivial wall-clock time, so this reliably exercises the
    # timeout path (a branch later in wave_1's dict order can finish "for
    # free" while an earlier branch is still being awaited, since all tasks
    # are created concurrently up front per §5.3 -- ocr going first avoids
    # that ordering artefact making the timeout flaky).
    monkeypatch.setitem(orchestrator.BUDGET_MS, "ocr", 1)

    inputs = orchestrator.PipelineInputs(
        session_id="timeout-session", lane_id=ctx.lane_id, officer_id=ctx.officer_id,
        documents=[_genuine_document_input(document_id="doc-2")],
    )
    ctx2 = PipelineContext(
        session_id="timeout-session", lane_id=ctx.lane_id, officer_id=ctx.officer_id,
        model_versions=ctx.model_versions, mrz_require_trained_weights=False,
    )
    session = await orchestrator.run(inputs, ctx2)

    assert "no_ocr" in session.coverage_flags
    ocr_signals = [s for s in session.signals if s.code == "OCR_UNAVAILABLE"]
    assert len(ocr_signals) == 1
    assert ocr_signals[0].severity == "info"
    # A degraded branch must never silently look like a clean pass.
    assert session.band is not None
    assert session.documents[0].fields == []
