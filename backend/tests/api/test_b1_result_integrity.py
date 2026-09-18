"""B1b acceptance checks on GET /result's wire body: coverage completeness,
no scalar risk score, no full document number, and Gate-1-failure language."""
from __future__ import annotations

import time

import cv2
from fastapi.testclient import TestClient

from app.main import app
from app.ocr.mrz import synth
from app.pipeline.registry import StageId
from tests.api._b1_helpers import analyse_in_process
from tests.ocr.conftest import build_document, render_viz_zone


def _document_jpeg(seed: int) -> bytes:
    record = synth.build_td3_record(
        surname="OKONKWO", given_names="AMARA", doc_number=f"Y{seed:07d}"[:9],
        nationality="UTO", issuing_country="UTO", birth_raw="900101", expiry_raw="311231", sex="F",
    )
    viz_zone = render_viz_zone(record)
    image = build_document(viz_zone, seed=seed)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def _document_and_record(seed: int):
    record = synth.build_td3_record(
        surname="OKONKWO", given_names="AMARA", doc_number=f"Y{seed:07d}"[:9],
        nationality="UTO", issuing_country="UTO", birth_raw="900101", expiry_raw="311231", sex="F",
    )
    viz_zone = render_viz_zone(record)
    image = build_document(viz_zone, seed=seed)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes(), record


def _run_to_settled(client: TestClient, seed: int, *, with_ground_truth: bool = False) -> dict:
    resp = client.post("/api/sessions")
    session_id = resp.json()["sessionId"]
    jpeg_bytes, record = _document_and_record(seed)
    client.post(
        f"/api/sessions/{session_id}/artefacts",
        data={"kind": "document-still"},
        files={"file": ("doc.jpg", jpeg_bytes, "image/jpeg")},
    )
    if with_ground_truth:
        # B1d closed the mrzGroundTruth HTTP door -- inject in-process.
        analyse_in_process(session_id, jpeg_bytes, record.lines)
    else:
        client.post(f"/api/sessions/{session_id}/analyse")

    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        resp = client.get(f"/api/sessions/{session_id}/result")
        if resp.status_code == 200:
            return resp.json()
        time.sleep(0.1)
    raise AssertionError("run never settled")


def _walk(obj):
    """Yield every (key, value) pair anywhere in a nested dict/list body."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item)


def test_coverage_contains_one_entry_per_registered_stage():
    with TestClient(app) as client:
        result = _run_to_settled(client, seed=201)
        assert len(result["coverage"]) == len(list(StageId))


def test_wave2_entries_report_unavailable_never_ran_true():
    with TestClient(app) as client:
        # Needs a genuine Gate-1 pass to see Wave 2 actually dispatch (and
        # report unavailable) rather than cascade to not-evaluated -- see
        # tests/api/_b1_helpers.py's in-process ground-truth injection.
        result = _run_to_settled(client, seed=202, with_ground_truth=True)
        wave2_modalities = {"tamper", "ovd", "face", "identity-graph"}
        for entry in result["coverage"]:
            if entry["modality"] in wave2_modalities:
                assert entry["unavailable"] is True
                assert entry["ran"] is False


def test_gate1_failure_yields_wave2_not_evaluated_distinct_from_unavailable():
    # No trained MRZ weights and no ground truth over this HTTP API -> MRZ
    # read fails -> Gate 1 fails -> Wave 2 cascades to not-evaluated, which
    # is a *different* signal than "this build cannot run this check at all"
    # (StageCoverage has no direct not-evaluated flag; a cascaded stage
    # reports blockedBy set and unavailable false, distinct from the
    # Wave-2-stub stages' unavailable=true/blockedBy=None).
    with TestClient(app) as client:
        result = _run_to_settled(client, seed=203)
        assert result["verdict"] == "review-required"  # Gate 1 failed -> recapture suggested

    convergence_or_gate2 = [e for e in result["coverage"] if e.get("blockedBy") == "gate-1"]
    assert convergence_or_gate2, "expected at least one stage cascaded to not-evaluated by gate-1"
    for entry in convergence_or_gate2:
        assert entry["unavailable"] is False


def test_no_scalar_risk_score_field_in_response_body():
    with TestClient(app) as client:
        result = _run_to_settled(client, seed=204)
    banned = {"risk", "score", "confidencescore", "riskscore"}
    for key, _value in _walk(result):
        assert key.lower() not in banned, f"found banned scalar-score-looking field: {key}"


def test_no_full_document_number_in_session_or_health_response_bodies():
    # ScreeningResult.mrz.fields intentionally includes the read MRZ
    # characters (the officer-verification ribbon, an existing tested B1a
    # contract -- see app/api/schemas.py's module docstring) and is out of
    # scope here. This test covers the NEW B1b resources this milestone
    # introduces: session/artefact metadata and health/thresholds, none of
    # which should ever carry a raw document number.
    with TestClient(app) as client:
        resp = client.post("/api/sessions")
        session_id = resp.json()["sessionId"]
        resp = client.get(f"/api/sessions/{session_id}")
        body = resp.json()
        assert "documentNumber" not in body
        for key, value in _walk(body):
            if isinstance(value, str):
                assert not value.startswith("X1234567"), "looks like a raw document number leaked"

        resp = client.get("/api/health")
        assert "documentNumber" not in resp.text
        resp = client.get("/api/config/thresholds")
        assert "documentNumber" not in resp.text


def test_gate1_failure_response_has_no_fraud_or_forgery_language():
    with TestClient(app) as client:
        result = _run_to_settled(client, seed=205)
    assert result["verdict"] == "review-required"
    reason = result["reason"].lower()
    assert "fraud" not in reason
    assert "forg" not in reason
    for finding in result["findings"]:
        assert "fraud" not in finding["hypothesis"].lower()
        assert "forg" not in finding["hypothesis"].lower()
        assert "fraud" not in finding["title"].lower()
        assert "forg" not in finding["title"].lower()
