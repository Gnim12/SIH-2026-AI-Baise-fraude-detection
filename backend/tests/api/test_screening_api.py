"""End-to-end API test (BACKEND_BRIEF.md §7): real multipart upload -> real
orchestrator run (real OCR branch) -> events streamed over the real WS route
-> a real Postgres-shaped row (sqlite in this test run, see tests/conftest.py)
readable back via GET /api/v1/history.

This exercises the exact multipart field names the already-built frontend's
src/api/capture.ts `submitCapture` sends, and the exact WS wire shape its
src/api/socket.ts consumes (camelCase, via app/contracts/wire.py)."""
from __future__ import annotations

import json
import random
import time

import cv2
from fastapi.testclient import TestClient

from app.main import app
from app.ocr.mrz import synth
from tests.api.conftest import login
from tests.ocr.conftest import build_document, render_viz_zone


def _genuine_jpeg_and_lines():
    rng = random.Random(11)
    record = synth.build_td3_record(
        surname="OKONKWO", given_names="AMARA", doc_number="Y1122334Z",
        nationality="UTO", issuing_country="UTO", birth_raw="900101",
        expiry_raw="311231", sex="F",
    )
    viz_zone = render_viz_zone(record)
    image = build_document(viz_zone, seed=11)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes(), record


def _wait_until_persisted(client: TestClient, session_id: str, *, timeout_s: float = 5.0) -> None:
    # The DecisionEvent is published on the WS a moment before the pipeline's
    # background task finishes writing the sealed... no, the *unsealed*
    # session row to the DB (upsert happens after orchestrator.run returns).
    # A test reading GET/decision immediately after seeing the decision event
    # can legitimately race that write; poll rather than assume ordering.
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = client.get(f"/api/v1/screening/{session_id}")
        if resp.status_code == 200:
            return
        time.sleep(0.05)
    raise AssertionError(f"session {session_id} was never persisted within {timeout_s}s")


def test_multipart_upload_streams_real_ocr_and_ends_in_decision():
    jpeg_bytes, record = _genuine_jpeg_and_lines()

    with TestClient(app) as client:
        login(client)  # auth is now in front of every endpoint below (task brief item 4)
        resp = client.post(
            "/api/v1/screening",
            data={
                "documentCount": "1",
                "document_0_type": "PASSPORT",
                "document_0_mrz_ground_truth": json.dumps(record.lines),
                "checkpointId": "IGI-T3-LANE-07",
                "officerId": "OFF-2291",
            },
            files={
                "document_0": ("passport.jpg", jpeg_bytes, "image/jpeg"),
                "liveFrame": ("live.jpg", jpeg_bytes, "image/jpeg"),
            },
        )
        assert resp.status_code == 201, resp.text
        session_id = resp.json()["sessionId"]

        events = []
        with client.websocket_connect(f"/ws/screening/{session_id}") as ws:
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["stage"] == "decision":
                    break

        stages = [e["stage"] for e in events]
        assert stages[0] == "received"
        assert "ocr" in stages
        assert stages[-1] == "decision"

        ocr_event = next(e for e in events if e["stage"] == "ocr")
        # camelCase wire shape matching frontend/src/types/screening.ts, and
        # real extracted values (not hardcoded case-01 fixture data).
        fields_by_key = {f["key"]: f["value"] for f in ocr_event["fields"]}
        assert fields_by_key.get("doc_number") == "Y1122334Z"
        assert ocr_event["mrz"]["status"] == "VERIFIED"
        assert "documentId" in ocr_event  # camelCase, not document_id

        history_resp = client.get("/api/v1/history")
        assert history_resp.status_code == 200
        # Not sealed yet (no officer decision) -- must not appear in history.
        assert all(s["sessionId"] != session_id for s in history_resp.json())

        _wait_until_persisted(client, session_id)
        decision_resp = client.post(
            f"/api/v1/screening/{session_id}/decision",
            json={"decision": "CLEAR", "note": ""},
        )
        assert decision_resp.status_code == 200, decision_resp.text
        sealed = decision_resp.json()
        assert sealed["sealed"] is True
        assert sealed["officerDecision"]["decision"] == "CLEAR"

        history_resp = client.get("/api/v1/history")
        assert any(s["sessionId"] == session_id for s in history_resp.json())

        entry_resp = client.get(f"/api/v1/history/{session_id}")
        assert entry_resp.status_code == 200
        assert entry_resp.json()["sessionId"] == session_id


def test_decision_requires_note_on_override():
    jpeg_bytes, record = _genuine_jpeg_and_lines()
    with TestClient(app) as client:
        login(client)
        resp = client.post(
            "/api/v1/screening",
            data={
                "documentCount": "1", "document_0_type": "PASSPORT",
                "document_0_mrz_ground_truth": json.dumps(record.lines),
                "checkpointId": "LANE-1", "officerId": "OFF-1",
            },
            files={"document_0": ("passport.jpg", jpeg_bytes, "image/jpeg")},
        )
        session_id = resp.json()["sessionId"]

        with client.websocket_connect(f"/ws/screening/{session_id}") as ws:
            while ws.receive_json()["stage"] != "decision":
                pass
        _wait_until_persisted(client, session_id)

        # System band is CLEAR; overriding to HOLD with no note must be rejected.
        bad = client.post(
            f"/api/v1/screening/{session_id}/decision",
            json={"decision": "HOLD", "note": ""},
        )
        assert bad.status_code == 422

        good = client.post(
            f"/api/v1/screening/{session_id}/decision",
            json={"decision": "HOLD", "note": "escalating on officer judgement"},
        )
        assert good.status_code == 200
        assert good.json()["officerDecision"]["override"] is True

        # A sealed session refuses a second decision.
        again = client.post(
            f"/api/v1/screening/{session_id}/decision",
            json={"decision": "CLEAR", "note": "x"},
        )
        assert again.status_code == 409
