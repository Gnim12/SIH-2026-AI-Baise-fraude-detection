"""B2: POST /api/sessions/{id}/decision -- server-side validation, computed
divergence, transactional seal, and the 409-on-second-decision rule."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from tests.api._b2_helpers import all_accepted_dispositions, create_settled_session, decision_payload


def test_missing_disposition_rejected_naming_the_finding():
    with TestClient(app) as client:
        sid, result = create_settled_session(client, seed=601)
        missing_finding_id = result["findings"][0]["findingId"]
        payload = decision_payload(result, finding_dispositions=[])
        resp = client.post(f"/api/sessions/{sid}/decision", json=payload)
        assert resp.status_code == 422, resp.text
        body = resp.json()
        assert missing_finding_id in body["error"]["message"]


def test_dismissal_without_sufficient_reason_rejected():
    with TestClient(app) as client:
        sid, result = create_settled_session(client, seed=602)
        dispositions = [{"findingId": f["findingId"], "disposition": "dismissed", "reason": "too short"}
                        for f in result["findings"]]
        payload = decision_payload(result, finding_dispositions=dispositions)
        resp = client.post(f"/api/sessions/{sid}/decision", json=payload)
        assert resp.status_code == 422, resp.text
        assert "reason" in resp.json()["error"]["message"].lower()


def test_dismissal_with_sufficient_reason_accepted():
    with TestClient(app) as client:
        sid, result = create_settled_session(client, seed=603)
        dispositions = [
            {"findingId": f["findingId"], "disposition": "dismissed", "reason": "Reviewed, artefact of print noise."}
            for f in result["findings"]
        ]
        payload = decision_payload(result, finding_dispositions=dispositions)
        resp = client.post(f"/api/sessions/{sid}/decision", json=payload)
        assert resp.status_code == 200, resp.text


def test_client_supplied_divergence_is_ignored_server_computes_its_own():
    with TestClient(app) as client:
        sid, result = create_settled_session(client, seed=604)
        assert result["verdict"] == "cleared"
        payload = decision_payload(
            result, action="reject", notes="Escalating on manual review.",
            divergence_reason="Suspicious watermark spotted by officer during physical inspection.",
        )
        payload["divergence"] = "none"  # a client trying to declare no divergence on an override
        resp = client.post(f"/api/sessions/{sid}/decision", json=payload)
        assert resp.status_code == 200, resp.text

        audit = client.get(f"/api/cases/{resp.json()['caseId']}/audit").json()
        assert audit[-1]["divergence"] == "officer-rejected-cleared"  # computed, not the client's "none"


def test_override_without_sufficient_reason_rejected():
    with TestClient(app) as client:
        sid, result = create_settled_session(client, seed=605)
        payload = decision_payload(result, action="reject", notes="Escalating.", divergence_reason="too short")
        resp = client.post(f"/api/sessions/{sid}/decision", json=payload)
        assert resp.status_code == 422, resp.text
        assert "divergen" in resp.json()["error"]["message"].lower() or "reason" in resp.json()["error"]["message"].lower()


def test_reject_without_notes_rejected():
    with TestClient(app) as client:
        sid, result = create_settled_session(client, seed=606)
        payload = decision_payload(
            result, action="reject", notes="",
            divergence_reason="Suspicious watermark spotted by officer during physical inspection.",
        )
        resp = client.post(f"/api/sessions/{sid}/decision", json=payload)
        assert resp.status_code == 422, resp.text
        assert "notes" in resp.json()["error"]["message"].lower()


def test_second_decision_on_same_session_returns_409():
    with TestClient(app) as client:
        sid, result = create_settled_session(client, seed=607)
        payload = decision_payload(result)
        first = client.post(f"/api/sessions/{sid}/decision", json=payload)
        assert first.status_code == 200, first.text
        second = client.post(f"/api/sessions/{sid}/decision", json=payload)
        assert second.status_code == 409, second.text


def test_missing_attestation_rejected():
    with TestClient(app) as client:
        sid, result = create_settled_session(client, seed=608)
        payload = decision_payload(result, attestation=False)
        resp = client.post(f"/api/sessions/{sid}/decision", json=payload)
        assert resp.status_code == 422, resp.text


def test_sealed_entry_contains_every_signal_disposition_pin_and_threshold():
    with TestClient(app) as client:
        sid, result = create_settled_session(client, seed=609)
        all_signal_ids = {s["signalId"] for f in result["findings"] for s in f["signals"]}
        all_finding_ids = {f["findingId"] for f in result["findings"]}
        payload = decision_payload(result)
        resp = client.post(f"/api/sessions/{sid}/decision", json=payload)
        assert resp.status_code == 200, resp.text
        case_id = resp.json()["caseId"]

        audit = client.get(f"/api/cases/{case_id}/audit").json()
        entry = audit[-1]
        assert set(entry["signalIds"]) == all_signal_ids
        assert {d["findingId"] for d in entry["findingDispositions"]} == all_finding_ids
        assert entry["modelPins"]  # non-empty
        assert entry["thresholds"]  # non-empty, the thresholds in force at seal time


def test_a_decision_requires_a_settled_run():
    with TestClient(app) as client:
        resp = client.post("/api/sessions")
        sid = resp.json()["sessionId"]
        payload = decision_payload({"findings": []})
        resp = client.post(f"/api/sessions/{sid}/decision", json=payload)
        assert resp.status_code == 409, resp.text
