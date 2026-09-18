"""B1b: POST /analyse, WS /events, GET /result -- a real multipart upload
into a real run_analysis() call (no trained MRZ weights and no ground truth
is ever supplied over this HTTP API, so these runs reach and fail Gate 1;
that is still a genuine screening reaching Gate 1, which is this milestone's
acceptance bar -- MRZ weight training is B1c, a separate, later milestone).
"""
from __future__ import annotations

import random
import time

import cv2
from fastapi.testclient import TestClient

from app.main import app
from app.ocr.mrz import synth
from tests.ocr.conftest import build_document, render_viz_zone


def _document_jpeg(seed: int) -> bytes:
    rng = random.Random(seed)
    record = synth.build_td3_record(
        surname="OKONKWO", given_names="AMARA", doc_number=f"Y{seed:07d}"[:9],
        nationality="UTO", issuing_country="UTO", birth_raw="900101", expiry_raw="311231", sex="F",
    )
    viz_zone = render_viz_zone(record)
    image = build_document(viz_zone, seed=seed)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def _create_session_with_document(client: TestClient, seed: int) -> str:
    resp = client.post("/api/sessions")
    session_id = resp.json()["sessionId"]
    resp = client.post(
        f"/api/sessions/{session_id}/artefacts",
        data={"kind": "document-still"},
        files={"file": ("doc.jpg", _document_jpeg(seed), "image/jpeg")},
    )
    assert resp.status_code == 200, resp.text
    return session_id


def _wait_for_settled(client: TestClient, session_id: str, *, timeout_s: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = client.get(f"/api/sessions/{session_id}/result")
        if resp.status_code == 200:
            return resp.json()
        assert resp.status_code == 409, resp.text
        time.sleep(0.1)
    raise AssertionError(f"run for session {session_id} never settled within {timeout_s}s")


def test_analyse_with_missing_artefacts_is_rejected_naming_the_gap():
    with TestClient(app) as client:
        resp = client.post("/api/sessions")
        session_id = resp.json()["sessionId"]
        resp = client.post(f"/api/sessions/{session_id}/analyse")
        assert resp.status_code == 422, resp.text
        body = resp.json()
        assert "document-still" in body["error"]["message"]
        assert body["error"]["detail"]["missingKind"] == "document-still"


def test_mrz_ground_truth_is_not_reachable_over_http():
    """B1d: POST /analyse used to accept `mrzGroundTruth` in its JSON body as
    a dev/test-only affordance. That field is now removed from the route's
    signature entirely -- FastAPI/Pydantic silently ignores an unknown extra
    body key by default here (the route takes no body model at all), so this
    asserts the field has zero effect on the run rather than being consumed:
    supplying it produces the exact same outcome (Gate-1 failure, no
    ground-truth-driven pass) as supplying nothing."""
    with TestClient(app) as client:
        session_id = _create_session_with_document(client, seed=109)
        resp = client.post(
            f"/api/sessions/{session_id}/analyse",
            json={"mrzGroundTruth": ["P<UTOOKONKWO<<AMARA<<<<<<<<<<<<<<<<<<<<<<<<<"]},
        )
        assert resp.status_code == 200, resp.text
        result = _wait_for_settled(client, session_id)
        # No trained MRZ weights and the field above was never consumed ->
        # Gate 1 still fails, proving the ground truth had no effect.
        assert result["verdict"] == "review-required"


def test_analyse_returns_run_id_without_blocking():
    with TestClient(app) as client:
        session_id = _create_session_with_document(client, seed=101)
        t0 = time.monotonic()
        resp = client.post(f"/api/sessions/{session_id}/analyse")
        elapsed = time.monotonic() - t0
        assert resp.status_code == 200, resp.text
        assert "runId" in resp.json()
        # A real Gate-1 pipeline run takes well over a second; returning in
        # under that proves the endpoint didn't await run_analysis inline.
        assert elapsed < 1.0, f"POST /analyse took {elapsed:.2f}s -- looks like it blocked on the run"


def test_second_analyse_during_in_flight_run_returns_same_run_id():
    with TestClient(app) as client:
        session_id = _create_session_with_document(client, seed=102)
        resp1 = client.post(f"/api/sessions/{session_id}/analyse")
        run_id_1 = resp1.json()["runId"]
        resp2 = client.post(f"/api/sessions/{session_id}/analyse")
        run_id_2 = resp2.json()["runId"]
        assert run_id_1 == run_id_2
        _wait_for_settled(client, session_id)  # drain the background task cleanly


def test_result_returns_409_while_in_flight_and_200_once_settled():
    with TestClient(app) as client:
        session_id = _create_session_with_document(client, seed=103)
        client.post(f"/api/sessions/{session_id}/analyse")
        resp = client.get(f"/api/sessions/{session_id}/result")
        assert resp.status_code == 409, resp.text
        assert "status" in resp.json()["error"]["detail"]

        result = _wait_for_settled(client, session_id)
        assert result["verdict"]


def test_client_connecting_mid_run_receives_prior_events_in_order_then_live():
    with TestClient(app) as client:
        session_id = _create_session_with_document(client, seed=104)
        client.post(f"/api/sessions/{session_id}/analyse")
        time.sleep(0.3)  # let a few events land before this client connects

        with client.websocket_connect(f"/api/sessions/{session_id}/events") as ws:
            seen_types = []
            terminal = False
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline and not terminal:
                msg = ws.receive_json()
                seen_types.append(msg.get("type"))
                if msg.get("type") == "run.settled":
                    terminal = True
            assert terminal, f"never saw a terminal message; got {seen_types}"
            assert "stage-started" in seen_types


def test_two_clients_receive_the_same_event_stream():
    with TestClient(app) as client:
        session_id = _create_session_with_document(client, seed=105)
        client.post(f"/api/sessions/{session_id}/analyse")

        def _collect(ws) -> list[str]:
            types = []
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                msg = ws.receive_json()
                types.append(msg.get("type"))
                if msg.get("type") == "run.settled":
                    break
            return types

        with client.websocket_connect(f"/api/sessions/{session_id}/events") as ws1, \
                client.websocket_connect(f"/api/sessions/{session_id}/events") as ws2:
            types1 = _collect(ws1)
            types2 = _collect(ws2)

        stage_events_1 = [t for t in types1 if t and t.startswith("stage-")]
        stage_events_2 = [t for t in types2 if t and t.startswith("stage-")]
        assert stage_events_1 == stage_events_2
        assert stage_events_1


def test_disconnecting_a_client_does_not_affect_the_run():
    with TestClient(app) as client:
        session_id = _create_session_with_document(client, seed=106)
        client.post(f"/api/sessions/{session_id}/analyse")

        with client.websocket_connect(f"/api/sessions/{session_id}/events") as ws:
            ws.receive_json()  # read exactly one event, then disconnect early

        result = _wait_for_settled(client, session_id)
        assert result["verdict"]


def test_a_terminal_message_is_sent_when_the_run_settles():
    with TestClient(app) as client:
        session_id = _create_session_with_document(client, seed=107)
        client.post(f"/api/sessions/{session_id}/analyse")

        with client.websocket_connect(f"/api/sessions/{session_id}/events") as ws:
            terminal_msg = None
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                msg = ws.receive_json()
                if msg.get("type") == "run.settled":
                    terminal_msg = msg
                    break
            assert terminal_msg is not None
            assert "runId" in terminal_msg
            assert "verdict" in terminal_msg


def test_heartbeats_are_emitted(monkeypatch):
    import app.api.b1_analyse as b1_analyse

    monkeypatch.setattr(b1_analyse, "HEARTBEAT_INTERVAL_S", 0.05)
    with TestClient(app) as client:
        session_id = _create_session_with_document(client, seed=108)
        client.post(f"/api/sessions/{session_id}/analyse")

        with client.websocket_connect(f"/api/sessions/{session_id}/events") as ws:
            saw_heartbeat = False
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                msg = ws.receive_json()
                if msg.get("type") == "heartbeat":
                    saw_heartbeat = True
                    break
                if msg.get("type") == "run.settled":
                    break
            assert saw_heartbeat, "expected at least one heartbeat before the run settled"
