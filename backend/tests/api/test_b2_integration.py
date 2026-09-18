"""B2 spec §6 "Integration -- run for real": complete a screening, submit a
decision, verify the chain, read the case back; then manually corrupt one
stored entry and confirm verification names it."""
from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from app.main import app
from app.storage.b2_models import B2AuditEntry
from app.storage.db import async_session_factory
from tests.api._b2_helpers import create_settled_session, decision_payload


def _unique_terminal(label: str) -> str:
    """tests/_test_screening.db persists across separate `pytest` runs, so a
    fixed terminal-id literal would accumulate leftover (and, in the second
    test below, deliberately corrupted) entries on a rerun."""
    return f"TERM-{label}-{uuid.uuid4().hex[:8]}"


def test_full_real_flow_screening_decision_verify_case_readback():
    with TestClient(app) as client:
        sid, result = create_settled_session(client, seed=801)
        assert result["verdict"] == "cleared"

        # A unique terminal id keeps this test's chain-verification check
        # scoped to entries this test itself sealed -- an unscoped
        # GET /api/audit/verify walks *every* terminal's chain in the shared
        # test sqlite file, including any other test's deliberately
        # corrupted rows (see tests/audit/test_chain.py's cleanup note).
        terminal_id = _unique_terminal("INTEGRATION-801")
        payload = decision_payload(result, action="clear", terminal_id=terminal_id)
        resp = client.post(f"/api/sessions/{sid}/decision", json=payload)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        case_id, entry_hash = body["caseId"], body["entryHash"]

        verify = client.get("/api/audit/verify", params={"terminalId": terminal_id}).json()
        assert verify["intact"] is True

        record = client.get(f"/api/cases/{case_id}").json()
        assert record["case"]["caseId"] == case_id
        assert record["decision"]["entryHash"] == entry_hash
        assert record["result"]["verdict"] == "cleared"


def test_manually_corrupted_entry_is_named_by_verify():
    with TestClient(app) as client:
        sid, result = create_settled_session(client, seed=802)
        terminal_id = _unique_terminal("INTEGRATION-802")
        resp = client.post(
            f"/api/sessions/{sid}/decision",
            json=decision_payload(result, action="clear", notes="", finding_dispositions=None, terminal_id=terminal_id),
        )
        assert resp.status_code == 200, resp.text
        entry_id = resp.json()["entryId"]

        pre_corruption = client.get("/api/audit/verify", params={"terminalId": terminal_id}).json()
        assert pre_corruption["intact"] is True

        import asyncio

        async def _corrupt():
            async with async_session_factory() as db:
                row = await db.get(B2AuditEntry, entry_id)
                row.payload_json = {**row.payload_json, "notes": "an officer did not actually write this"}
                await db.commit()

        asyncio.run(_corrupt())

        post_corruption = client.get("/api/audit/verify", params={"terminalId": terminal_id}).json()
        assert post_corruption["intact"] is False
        assert post_corruption["brokenEntryId"] == entry_id
        assert entry_id in post_corruption["reason"]
