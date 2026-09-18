"""B2: GET /api/cases -- server-side filtering, pagination, and the
first-class coverage/divergence/linked filters."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from tests.api._b2_helpers import create_settled_session, decision_payload


def _walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item)


def test_pagination_returns_the_stated_range():
    with TestClient(app) as client:
        for seed in range(701, 703):
            sid, result = create_settled_session(client, seed=seed)
            client.post(f"/api/sessions/{sid}/decision", json=decision_payload(result))

        resp = client.get("/api/cases", params={"page": 1})
        body = resp.json()
        assert resp.status_code == 200, resp.text
        assert body["page"] == 1
        assert body["pageSize"] == 20
        assert body["total"] >= 2
        assert len(body["cases"]) <= 20


def test_divergence_only_returns_exactly_the_diverged_cases():
    with TestClient(app) as client:
        sid_clear, result_clear = create_settled_session(client, seed=704)
        client.post(f"/api/sessions/{sid_clear}/decision", json=decision_payload(result_clear, action="clear"))

        sid_diverge, result_diverge = create_settled_session(client, seed=705)
        assert result_diverge["verdict"] == "cleared"
        payload = decision_payload(
            result_diverge, action="reject", notes="Escalating on manual inspection.",
            divergence_reason="Officer identified a discrepancy during physical document handling.",
        )
        client.post(f"/api/sessions/{sid_diverge}/decision", json=payload)

        resp = client.get("/api/cases", params={"divergence": "only"})
        body = resp.json()
        case_ids = {c["sessionId"] for c in body["cases"]}
        assert sid_diverge in case_ids
        assert sid_clear not in case_ids
        for case in body["cases"]:
            assert case["divergence"] not in (None, "none")


def test_coverage_partial_returns_exactly_the_partial_coverage_cases():
    with TestClient(app) as client:
        # with_ground_truth=False -> Gate 1 never genuinely passes (no
        # trained MRZ weights), so Wave 2 cascades to not-evaluated and the
        # run's coverage is incomplete.
        sid_partial, result_partial = create_settled_session(client, seed=706, with_ground_truth=False)
        client.post(
            f"/api/sessions/{sid_partial}/decision",
            json=decision_payload(result_partial, action="request-recapture"),
        )

        sid_complete, result_complete = create_settled_session(client, seed=707, with_ground_truth=True)
        client.post(f"/api/sessions/{sid_complete}/decision", json=decision_payload(result_complete))

        resp = client.get("/api/cases", params={"coverage": "partial"})
        body = resp.json()
        session_ids = {c["sessionId"] for c in body["cases"]}
        assert sid_partial in session_ids
        assert sid_complete not in session_ids
        for case in body["cases"]:
            assert case["coverageComplete"] is False


def test_linked_only_returns_empty_with_unavailable_indicator():
    with TestClient(app) as client:
        sid, result = create_settled_session(client, seed=708)
        client.post(f"/api/sessions/{sid}/decision", json=decision_payload(result))

        resp = client.get("/api/cases", params={"linked": "only"})
        body = resp.json()
        assert body["cases"] == []
        assert body["linkedUnavailable"] is True


def test_no_full_document_number_in_any_case_response():
    with TestClient(app) as client:
        sid, result = create_settled_session(client, seed=709)
        resp = client.post(f"/api/sessions/{sid}/decision", json=decision_payload(result))
        case_id = resp.json()["caseId"]

        list_resp = client.get("/api/cases")
        for _, v in _walk(list_resp.json()):
            if isinstance(v, str):
                assert not v.startswith("Y070"), "looks like a raw document number leaked"

        record_resp = client.get(f"/api/cases/{case_id}")
        for _, v in _walk(record_resp.json()):
            if isinstance(v, str):
                assert not v.startswith("Y070"), "looks like a raw document number leaked"


def test_case_record_returns_full_result_and_decision_with_dispositions():
    with TestClient(app) as client:
        sid, result = create_settled_session(client, seed=710)
        dispositions = [
            {"findingId": f["findingId"], "disposition": "dismissed", "reason": "Reviewed and confirmed benign."}
            for f in result["findings"]
        ]
        resp = client.post(
            f"/api/sessions/{sid}/decision",
            json=decision_payload(result, finding_dispositions=dispositions, notes="", action="clear"),
        )
        case_id = resp.json()["caseId"]

        record = client.get(f"/api/cases/{case_id}").json()
        assert record["case"]["caseId"] == case_id
        assert record["result"]["verdict"] == result["verdict"]
        assert record["decision"]["findingDispositions"][0]["disposition"] == "dismissed"
        assert record["decision"]["findingDispositions"][0]["reason"] == "Reviewed and confirmed benign."
