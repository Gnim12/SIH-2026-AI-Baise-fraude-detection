"""B2: shared helper to get a session through to a genuinely settled run
(Gate 1 pass, via in-process ground-truth injection -- see
tests/api/_b1_helpers.py) so decision/case tests have real findings/signals
to submit a decision against."""
from __future__ import annotations

import time
from typing import Any

import cv2
from fastapi.testclient import TestClient

from app.ocr.mrz import synth
from tests.api._b1_helpers import analyse_in_process
from tests.ocr.conftest import build_document, render_viz_zone


def document_and_record(seed: int) -> tuple[bytes, Any]:
    record = synth.build_td3_record(
        surname="OKONKWO", given_names="AMARA", doc_number=f"Y{seed:07d}"[:9],
        nationality="UTO", issuing_country="UTO", birth_raw="900101", expiry_raw="311231", sex="F",
    )
    viz_zone = render_viz_zone(record)
    image = build_document(viz_zone, seed=seed)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes(), record


def create_settled_session(client: TestClient, seed: int, *, with_ground_truth: bool = True) -> tuple[str, dict]:
    """Returns (session_id, ScreeningResult-as-dict)."""
    resp = client.post("/api/sessions")
    session_id = resp.json()["sessionId"]
    jpeg_bytes, record = document_and_record(seed)
    client.post(
        f"/api/sessions/{session_id}/artefacts",
        data={"kind": "document-still"},
        files={"file": ("doc.jpg", jpeg_bytes, "image/jpeg")},
    )
    if with_ground_truth:
        analyse_in_process(session_id, jpeg_bytes, record.lines)
    else:
        client.post(f"/api/sessions/{session_id}/analyse")

    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        resp = client.get(f"/api/sessions/{session_id}/result")
        if resp.status_code == 200:
            return session_id, resp.json()
        time.sleep(0.1)
    raise AssertionError("run never settled")


def all_accepted_dispositions(result: dict) -> list[dict]:
    return [{"findingId": f["findingId"], "disposition": "accepted"} for f in result["findings"]]


def decision_payload(
    result: dict, *, action: str = "clear", officer_id: str = "OFF-B2-TEST", terminal_id: str = "TERM-B2-TEST",
    notes: str = "", divergence_reason: str | None = None, attestation: bool = True,
    finding_dispositions: list[dict] | None = None,
) -> dict:
    return {
        "officerId": officer_id,
        "terminalId": terminal_id,
        "action": action,
        "findingDispositions": finding_dispositions if finding_dispositions is not None else all_accepted_dispositions(result),
        "notes": notes,
        "divergenceReason": divergence_reason,
        "attestation": attestation,
    }
