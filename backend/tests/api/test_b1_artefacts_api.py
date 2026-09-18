"""B1b: POST/GET /api/sessions, POST/DELETE /api/sessions/{id}/artefacts."""
from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient

from app.main import app


def _create_session(client: TestClient) -> str:
    resp = client.post("/api/sessions")
    assert resp.status_code == 201, resp.text
    return resp.json()["sessionId"]


def _jpeg_bytes() -> bytes:
    # Minimal valid JPEG: SOI + APP0 + EOI is enough to satisfy magic-byte
    # sniffing (FF D8 FF ...), which is all the upload route inspects.
    return bytes.fromhex("ffd8ffe000104a46494600010100000100010000") + b"\x00" * 32 + bytes.fromhex("ffd9")


def _png_bytes() -> bytes:
    return bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 32


def _pdf_bytes() -> bytes:
    return b"%PDF-1.4\n" + b"\x00" * 32


def _mp4_bytes() -> bytes:
    # box size (4) + 'ftyp' + brand, at the offsets the sniffer checks.
    return b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32


def _gif_bytes() -> bytes:
    return b"GIF89a" + b"\x00" * 32


def test_upload_accepts_each_allowed_type_by_magic_bytes():
    with TestClient(app) as client:
        session_id = _create_session(client)
        for kind, data, content_type in (
            ("document-still", _jpeg_bytes(), "image/jpeg"),
            ("secondary-still", _png_bytes(), "image/png"),
        ):
            resp = client.post(
                f"/api/sessions/{session_id}/artefacts",
                data={"kind": kind},
                files={"file": (f"{kind}.bin", data, "application/octet-stream")},
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["artefact"]["contentType"] == content_type
            assert body["artefact"]["sha256"] == hashlib.sha256(data).hexdigest()

        # A PDF upload under document-still, still sniffed correctly.
        session_id2 = _create_session(client)
        resp = client.post(
            f"/api/sessions/{session_id2}/artefacts",
            data={"kind": "document-still"},
            files={"file": ("doc.bin", _pdf_bytes(), "application/octet-stream")},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["artefact"]["contentType"] == "application/pdf"


def test_extension_is_not_trusted_a_jpg_containing_a_gif_is_rejected():
    with TestClient(app) as client:
        session_id = _create_session(client)
        resp = client.post(
            f"/api/sessions/{session_id}/artefacts",
            data={"kind": "document-still"},
            files={"file": ("passport.jpg", _gif_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 415, resp.text
        body = resp.json()
        assert "error" in body
        assert "jpeg" in body["error"]["message"].lower() or "accepted" in body["error"]["message"].lower()


def test_oversized_upload_rejected_with_size_in_message(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "b1_max_image_pdf_upload_mb", 0)  # any real file now exceeds 0MB
    with TestClient(app) as client:
        session_id = _create_session(client)
        data = _jpeg_bytes()
        resp = client.post(
            f"/api/sessions/{session_id}/artefacts",
            data={"kind": "document-still"},
            files={"file": ("passport.jpg", data, "image/jpeg")},
        )
        assert resp.status_code == 413, resp.text
        message = resp.json()["error"]["message"]
        assert str(len(data)) in message


def test_attaching_new_document_still_deletes_existing_sweep_and_its_bytes(monkeypatch):
    import app.api.b1_sessions as b1_sessions

    # Sweep coverage extraction isn't implemented (real cv2.VideoCapture on
    # a fake mp4 would just fail to open and yield (None, None, None), which
    # is fine for this test) -- no need to build a real video container.

    with TestClient(app) as client:
        session_id = _create_session(client)
        resp = client.post(
            f"/api/sessions/{session_id}/artefacts",
            data={"kind": "ovd-sweep"},
            files={"file": ("sweep.mp4", _mp4_bytes(), "video/mp4")},
        )
        assert resp.status_code == 200, resp.text
        sweep_path = resp.json()["artefact"]  # noqa: F841 -- path isn't in the wire shape; checked via DB below

        from app.storage import b1_repositories as repo
        from app.storage.db import async_session_factory
        import asyncio

        async def _sweep_path() -> str:
            async with async_session_factory() as db:
                sweep = await repo.get_artefact(db, session_id, "ovd-sweep")
                assert sweep is not None
                return sweep.stored_path

        stored_path = asyncio.run(_sweep_path())
        from pathlib import Path
        assert Path(stored_path).exists()

        resp = client.post(
            f"/api/sessions/{session_id}/artefacts",
            data={"kind": "document-still"},
            files={"file": ("doc.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["notice"] == (
            "Document replaced. Re-capture the tilt sweep so both refer to the same document."
        )
        assert not Path(stored_path).exists()

        resp = client.get(f"/api/sessions/{session_id}")
        kinds = {a["kind"] for a in resp.json()["artefacts"]}
        assert "ovd-sweep" not in kinds
        assert "document-still" in kinds


def test_deleting_document_still_deletes_the_sweep():
    with TestClient(app) as client:
        session_id = _create_session(client)
        client.post(
            f"/api/sessions/{session_id}/artefacts",
            data={"kind": "document-still"},
            files={"file": ("doc.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        client.post(
            f"/api/sessions/{session_id}/artefacts",
            data={"kind": "ovd-sweep"},
            files={"file": ("sweep.mp4", _mp4_bytes(), "video/mp4")},
        )

        resp = client.delete(f"/api/sessions/{session_id}/artefacts/document-still")
        assert resp.status_code == 200, resp.text
        assert resp.json()["notice"] == (
            "Document replaced. Re-capture the tilt sweep so both refer to the same document."
        )

        resp = client.get(f"/api/sessions/{session_id}")
        kinds = {a["kind"] for a in resp.json()["artefacts"]}
        assert kinds == set()


def test_sweep_below_coverage_threshold_rejected_with_wider_angle_message(monkeypatch):
    import app.api.b1_sessions as b1_sessions

    # Real angular-coverage estimation isn't implemented (always None, per
    # the B1b spec: "do not invent a value") -- to prove the enforcement
    # code path itself is real and wired correctly, simulate a future
    # estimator reporting insufficient coverage.
    monkeypatch.setattr(
        b1_sessions, "_extract_sweep_metadata", lambda path: (1200.0, 40, 10.0),
    )

    with TestClient(app) as client:
        session_id = _create_session(client)
        resp = client.post(
            f"/api/sessions/{session_id}/artefacts",
            data={"kind": "ovd-sweep"},
            files={"file": ("sweep.mp4", _mp4_bytes(), "video/mp4")},
        )
        assert resp.status_code == 422, resp.text
        assert resp.json()["error"]["message"] == "Insufficient tilt range. Sweep again through a wider angle."


def test_sha256_is_recorded_and_matches_stored_bytes():
    with TestClient(app) as client:
        session_id = _create_session(client)
        data = _jpeg_bytes()
        resp = client.post(
            f"/api/sessions/{session_id}/artefacts",
            data={"kind": "document-still"},
            files={"file": ("doc.jpg", data, "image/jpeg")},
        )
        assert resp.status_code == 200, resp.text
        artefact = resp.json()["artefact"]
        assert artefact["sha256"] == hashlib.sha256(data).hexdigest()

        from app.storage import b1_repositories as repo
        from app.storage.db import async_session_factory
        import asyncio

        async def _check() -> None:
            async with async_session_factory() as db:
                row = await repo.get_artefact(db, session_id, "document-still")
                assert row is not None
                from pathlib import Path
                on_disk = Path(row.stored_path).read_bytes()
                assert hashlib.sha256(on_disk).hexdigest() == artefact["sha256"]

        asyncio.run(_check())
