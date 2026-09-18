import logging
logging.basicConfig(level=logging.DEBUG)

from starlette.testclient import TestClient
from app.main import app

with TestClient(app) as client:
    session_id = "81e6c5d9-a2c5-4a01-bc4d-8a4a78bbf510"
    payload = {
        "officerId": "BCF-0001-0001",
        "terminalId": "TERM-07",
        "action": "request-recapture",
        "findingDispositions": [
            {"findingId": "e73026b5-2faf-4a26-8662-13e8747e1ede", "disposition": "accepted"},
            {"findingId": "04239e57-d063-4811-a3e0-bc1d8613af20", "disposition": "accepted"},
            {"findingId": "1555e9e7-fb30-4eab-8e2c-5b05597212c4", "disposition": "accepted"},
            {"findingId": "299a3d0c-c06b-4b2d-82ce-be8b4b540d90", "disposition": "accepted"},
            {"findingId": "740b499f-10bd-456f-b78d-983d0e313556", "disposition": "accepted"},
            {"findingId": "544fc5af-5cab-4cab-bd6f-586e7d472695", "disposition": "accepted"},
        ],
        "notes": "",
        "divergenceReason": None,
        "attestation": True,
    }
    resp = client.post(f"/api/sessions/{session_id}/decision", json=payload)
    print("STATUS", resp.status_code)
    print(resp.text)
