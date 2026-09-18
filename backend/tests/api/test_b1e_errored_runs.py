"""B1e: a run that crashes during settle must surface as a distinct
`errored` status -- never as `settled` (which would make GET /result 409
forever, indistinguishable from a slow run) and never silently.

B1d already added the `errored` status and the corresponding `run.error` WS
terminal message / `run_failed` HTTP error while fixing a real Postgres
FK-violation crash discovered in that milestone -- this file is the test
coverage B1e adds for that existing contract (it had none before).
"""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.main import app
from app.storage import b1_repositories as repo

from .test_b1_analyse_and_ws_api import _create_session_with_document


def _force_settle_to_raise(monkeypatch) -> None:
    import app.api.b1_analyse as b1_analyse

    async def _boom(*args, **kwargs):
        raise RuntimeError("simulated crash during settle_run (B1e test fault injection)")

    monkeypatch.setattr(b1_analyse.repo, "settle_run", _boom)


def _wait_for_errored(client: TestClient, session_id: str, *, timeout_s: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        resp = client.get(f"/api/sessions/{session_id}/result")
        if resp.status_code == 500:
            return resp.json()
        assert resp.status_code == 409, resp.text
        time.sleep(0.1)
    raise AssertionError(f"run for session {session_id} never reached errored within {timeout_s}s")


def test_a_run_that_raises_during_settle_is_recorded_errored(monkeypatch):
    _force_settle_to_raise(monkeypatch)
    with TestClient(app) as client:
        session_id = _create_session_with_document(client, seed=201)
        resp = client.post(f"/api/sessions/{session_id}/analyse")
        run_id = resp.json()["runId"]

        body = _wait_for_errored(client, session_id)
        assert body["error"]["code"] == "run_failed"
        assert body["error"]["detail"]["runId"] == run_id
        # No follow-up polling loop needed: the 500 is itself the exhaustion
        # signal, unlike a 409 which means "ask again".


def test_errored_run_result_never_reports_settled_or_hangs_at_409(monkeypatch):
    """A crashed run must not be indistinguishable from a slow one -- this is
    the exact regression B1d fixed (an earlier handler marked crashed runs
    "settled" with no result_json, so GET /result returned 409 forever)."""
    _force_settle_to_raise(monkeypatch)
    with TestClient(app) as client:
        session_id = _create_session_with_document(client, seed=202)
        client.post(f"/api/sessions/{session_id}/analyse")

        _wait_for_errored(client, session_id)  # succeeds only if 500 is ever reached
        # A second read is still the terminal error, not a flip back to 409.
        resp = client.get(f"/api/sessions/{session_id}/result")
        assert resp.status_code == 500, resp.text


def test_ws_terminal_message_reports_run_error_when_connected_live(monkeypatch):
    _force_settle_to_raise(monkeypatch)
    with TestClient(app) as client:
        session_id = _create_session_with_document(client, seed=203)
        resp = client.post(f"/api/sessions/{session_id}/analyse")
        run_id = resp.json()["runId"]

        with client.websocket_connect(f"/api/sessions/{session_id}/events") as ws:
            terminal = None
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline and terminal is None:
                msg = ws.receive_json()
                if msg.get("type") in ("run.error", "run.settled"):
                    terminal = msg
            assert terminal is not None, "never saw a terminal message"
            assert terminal["type"] == "run.error", terminal
            assert terminal["runId"] == run_id
            assert "message" in terminal


def test_ws_terminal_message_reports_run_error_on_replay_after_run_already_errored(monkeypatch):
    """A client that connects *after* the crash (bus already closed) must
    still learn about the error via the durable-log replay path, not just
    the live-bus path exercised above."""
    _force_settle_to_raise(monkeypatch)
    with TestClient(app) as client:
        session_id = _create_session_with_document(client, seed=204)
        resp = client.post(f"/api/sessions/{session_id}/analyse")
        run_id = resp.json()["runId"]

        _wait_for_errored(client, session_id)

        with client.websocket_connect(f"/api/sessions/{session_id}/events") as ws:
            terminal = None
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and terminal is None:
                msg = ws.receive_json()
                if msg.get("type") in ("run.error", "run.settled"):
                    terminal = msg
            assert terminal is not None
            assert terminal["type"] == "run.error", terminal
            assert terminal["runId"] == run_id


def test_no_banner_copy_implies_the_document_or_traveller_is_at_fault(monkeypatch):
    _force_settle_to_raise(monkeypatch)
    with TestClient(app) as client:
        session_id = _create_session_with_document(client, seed=205)
        client.post(f"/api/sessions/{session_id}/analyse")
        body = _wait_for_errored(client, session_id)

        message = body["error"]["message"].lower()
        for term in ("fraud", "fake", "forged", "invalid document", "suspicious", "traveller is"):
            assert term not in message, f"{term!r} found in error copy: {body['error']['message']}"


def test_re_run_after_errored_is_not_blocked_by_the_in_flight_check(monkeypatch):
    """The B1d per-session lock only treats status=='running' as in-flight --
    an errored run must not permanently block re-analysis of the session."""
    _force_settle_to_raise(monkeypatch)
    with TestClient(app) as client:
        session_id = _create_session_with_document(client, seed=206)
        resp1 = client.post(f"/api/sessions/{session_id}/analyse")
        run_id_1 = resp1.json()["runId"]
        _wait_for_errored(client, session_id)

        monkeypatch.undo()  # let the re-run actually settle normally
        resp2 = client.post(f"/api/sessions/{session_id}/analyse")
        run_id_2 = resp2.json()["runId"]
        assert run_id_2 != run_id_1

        deadline = time.monotonic() + 20.0
        settled = False
        while time.monotonic() < deadline:
            r = client.get(f"/api/sessions/{session_id}/result")
            if r.status_code == 200:
                settled = True
                break
            time.sleep(0.1)
        assert settled, "re-run after an errored run never settled"
