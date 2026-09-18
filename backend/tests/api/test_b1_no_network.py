"""B1b: no outbound network call during a full run -- a border checkpoint
loses connectivity; every model must run fully offline (BACKEND_BRIEF.md
§1.5). Guards `socket.socket.connect`/`connect_ex`, the calls actually used
to reach a remote host -- not socket construction itself, which asyncio's
own event loop (self-pipe, in-process transport) legitimately uses on
Windows even for purely local work."""
from __future__ import annotations

import socket
import time

import cv2
from fastapi.testclient import TestClient

from app.main import app
from app.ocr.mrz import synth
from tests.ocr.conftest import build_document, render_viz_zone

_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


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


def test_no_outbound_network_call_during_a_full_run(monkeypatch):
    violations: list[tuple] = []
    real_connect = socket.socket.connect

    def _guarded_connect(self, address, *a, **kw):
        host = address[0] if isinstance(address, tuple) else address
        if host not in _LOCAL_HOSTS:
            violations.append(address)
            raise AssertionError(f"unexpected outbound connection attempt to {address!r}")
        return real_connect(self, address, *a, **kw)

    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)

    with TestClient(app) as client:
        resp = client.post("/api/sessions")
        session_id = resp.json()["sessionId"]
        client.post(
            f"/api/sessions/{session_id}/artefacts",
            data={"kind": "document-still"},
            files={"file": ("doc.jpg", _document_jpeg(401), "image/jpeg")},
        )
        client.post(f"/api/sessions/{session_id}/analyse")

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if client.get(f"/api/sessions/{session_id}/result").status_code == 200:
                break
            time.sleep(0.1)

    assert violations == []
