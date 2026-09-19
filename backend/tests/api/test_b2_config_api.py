"""B2: PUT /api/config/thresholds -- bounds validation, reason length,
batch-seals-as-one-entry, and model pins being read-only."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_get_thresholds_includes_officer_tunable_list():
    with TestClient(app) as client:
        resp = client.get("/api/config/thresholds")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "officerThresholds" in body
        ids = {t["id"] for t in body["officerThresholds"]}
        assert "mrz-viz-field-tolerance" in ids
        assert "graph-match-min" in ids


def test_out_of_range_threshold_rejected_with_bounds_named():
    with TestClient(app) as client:
        resp = client.put(
            "/api/config/thresholds",
            json={
                "officerId": "OFF-CFG", "terminalId": "TERM-CFG",
                "changes": [{"id": "face-match-min", "value": 5.0}],
                "reason": "Testing an out-of-range threshold change end to end.",
            },
        )
        assert resp.status_code == 422, resp.text
        message = resp.json()["error"]["message"]
        assert "0.4" in message and "0.95" in message


def test_change_without_sufficient_reason_rejected():
    with TestClient(app) as client:
        resp = client.put(
            "/api/config/thresholds",
            json={
                "officerId": "OFF-CFG", "terminalId": "TERM-CFG",
                "changes": [{"id": "face-match-min", "value": 0.7}],
                "reason": "too short",
            },
        )
        assert resp.status_code == 422, resp.text
        assert "reason" in resp.json()["error"]["message"].lower()


def test_a_batch_seals_as_one_entry_containing_all_changes():
    with TestClient(app) as client:
        resp = client.put(
            "/api/config/thresholds",
            json={
                "officerId": "OFF-CFG", "terminalId": "TERM-BATCH",
                "changes": [
                    {"id": "face-match-min", "value": 0.7},
                    {"id": "tamper-confidence-min", "value": 0.5},
                ],
                "reason": "Recalibrating after quarterly review of false-positive rate.",
            },
        )
        assert resp.status_code == 200, resp.text
        entry_id = resp.json()["entryId"]

        audit = client.get("/api/config/audit").json()
        matching = [c for c in audit if c["auditEntryHash"] == resp.json()["entryHash"]]
        assert len(matching) == 1
        assert len(matching[0]["changes"]) == 2

        verify = client.get("/api/audit/verify", params={"terminalId": "TERM-BATCH"}).json()
        assert verify["intact"] is True
        assert entry_id


def test_no_endpoint_can_modify_a_model_pin():
    with TestClient(app) as client:
        resp = client.put(
            "/api/config/thresholds",
            json={
                "officerId": "OFF-CFG", "terminalId": "TERM-CFG",
                "changes": [{"id": "mrz-crnn-slot", "value": 9.9}],
                "reason": "Attempting to sneak a model pin through the threshold endpoint.",
            },
        )
        assert resp.status_code == 422, resp.text  # not a known threshold id

        # No route in the app accepts a model-pin write at all.
        for route in app.routes:
            path = getattr(route, "path", "")
            methods = getattr(route, "methods", set()) or set()
            if "model" in path and "pin" in path:
                assert methods.isdisjoint({"PUT", "POST", "PATCH", "DELETE"})
