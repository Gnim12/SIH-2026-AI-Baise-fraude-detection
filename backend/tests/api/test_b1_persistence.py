"""B1b: stage-event append-only-ness, monotonic sequencing, masked document
numbers, artefact bytes on disk."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import cv2
from fastapi.testclient import TestClient

from app.main import app
from app.ocr.mrz import synth
from tests.api._b1_helpers import analyse_in_process
from tests.ocr.conftest import build_document, render_viz_zone


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


def test_every_stage_transition_is_stored_with_monotonic_sequence():
    with TestClient(app) as client:
        resp = client.post("/api/sessions")
        session_id = resp.json()["sessionId"]
        client.post(
            f"/api/sessions/{session_id}/artefacts",
            data={"kind": "document-still"},
            files={"file": ("doc.jpg", _document_jpeg(301), "image/jpeg")},
        )
        resp = client.post(f"/api/sessions/{session_id}/analyse")
        run_id = resp.json()["runId"]

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if client.get(f"/api/sessions/{session_id}/result").status_code == 200:
                break
            time.sleep(0.1)

        from app.storage import b1_repositories as repo
        from app.storage.db import async_session_factory

        async def _events():
            async with async_session_factory() as db:
                return await repo.list_stage_events(db, run_id)

        events = asyncio.run(_events())
        assert events, "expected at least one stage event to have been recorded"
        sequences = [e.sequence for e in events]
        assert sequences == sorted(sequences)
        assert sequences == list(range(1, len(sequences) + 1))


def test_stage_events_are_append_only_no_update_or_delete_path():
    """The only functions in the repository module that reference
    B1StageEvent at all are `append_stage_event` (an INSERT) and
    `list_stage_events` (a SELECT) -- there is no update or delete path for
    this table anywhere in the codebase."""
    import inspect

    import app.storage.b1_repositories as repo_module

    touching = [
        name for name, obj in vars(repo_module).items()
        if inspect.isfunction(obj) and obj.__module__ == repo_module.__name__
        and "B1StageEvent" in inspect.getsource(obj)
    ]
    assert set(touching) == {"append_stage_event", "list_stage_events"}

    append_src = inspect.getsource(repo_module.append_stage_event)
    list_src = inspect.getsource(repo_module.list_stage_events)
    assert "db.delete(" not in append_src and "db.delete(" not in list_src
    assert ".update(" not in append_src and ".update(" not in list_src


def test_stored_document_numbers_are_masked():
    from app.storage.b1_models import mask_document_number

    assert mask_document_number("X1234567Y") == "•••• 567Y"
    assert mask_document_number(None) is None
    assert mask_document_number("") is None

    with TestClient(app) as client:
        resp = client.post("/api/sessions")
        session_id = resp.json()["sessionId"]
        record = synth.build_td3_record(
            surname="OKONKWO", given_names="AMARA", doc_number="Y1234567Z",
            nationality="UTO", issuing_country="UTO", birth_raw="900101", expiry_raw="311231", sex="F",
        )
        viz_zone = render_viz_zone(record)
        image = build_document(viz_zone, seed=302)
        ok, buf = cv2.imencode(".jpg", image)
        assert ok
        client.post(
            f"/api/sessions/{session_id}/artefacts",
            data={"kind": "document-still"},
            files={"file": ("doc.jpg", buf.tobytes(), "image/jpeg")},
        )
        # B1d closed the mrzGroundTruth HTTP door -- inject in-process.
        analyse_in_process(session_id, buf.tobytes(), record.lines)

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if client.get(f"/api/sessions/{session_id}/result").status_code == 200:
                break
            time.sleep(0.1)

        from app.storage import b1_repositories as repo
        from app.storage.db import async_session_factory
        from sqlalchemy import select
        from app.storage.b1_models import B1Case

        async def _case():
            async with async_session_factory() as db:
                result = await db.execute(select(B1Case).where(B1Case.session_id == session_id))
                return result.scalar_one_or_none()

        case = asyncio.run(_case())
        assert case is not None
        assert case.document_number_masked is not None
        assert "Y1234567Z" not in case.document_number_masked
        assert case.document_number_masked.endswith("567Z")


def test_settle_run_persists_findings_sharing_a_signal_code():
    """Regression: multiple findings can carry the same signal *code* (e.g.
    VIZ_LOW_CONFIDENCE on several distinct VIZ fields) -- signal_id is a
    category, not a per-instance identifier. settle_run() dedupes these into
    one b1_signal row and points every finding's finding_signal join row at
    it. Without an explicit flush between the signal/finding inserts and the
    join-row inserts, this previously threw a ForeignKeyViolationError
    against Postgres (the real dev/prod DB) on every run whose findings
    shared a signal code -- silently invisible in the sqlite-backed test
    suite until PRAGMA foreign_keys=ON was enabled (see app/storage/db.py)."""
    with TestClient(app) as client:
        resp = client.post("/api/sessions")
        session_id = resp.json()["sessionId"]
        data = _document_jpeg(304)
        client.post(
            f"/api/sessions/{session_id}/artefacts",
            data={"kind": "document-still"},
            files={"file": ("doc.jpg", data, "image/jpeg")},
        )
        resp = client.post(f"/api/sessions/{session_id}/analyse")
        assert resp.status_code == 200, resp.text

        deadline = time.monotonic() + 20.0
        result_resp = None
        while time.monotonic() < deadline:
            result_resp = client.get(f"/api/sessions/{session_id}/result")
            if result_resp.status_code == 200:
                break
            time.sleep(0.1)

        assert result_resp is not None and result_resp.status_code == 200, (
            result_resp.text if result_resp is not None else "no response"
        )
        body = result_resp.json()
        signal_ids = [s["signalId"] for f in body["findings"] for s in f["signals"]]
        assert signal_ids, "expected this untrained-MRZ run to produce findings"
        # The bug only reproduces when a signal code repeats across findings;
        # assert that's actually exercised here, not vacuously passing.
        assert len(signal_ids) != len(set(signal_ids)), (
            "fixture no longer produces a repeated signal code across findings; "
            "this test needs a fixture that does to guard the settle_run bug"
        )


def test_second_analyse_during_in_flight_run_does_not_create_a_parallel_run():
    """Regression: two near-simultaneous POST /analyse calls for the same
    session (a double-click, or a client retry) must not each create their
    own run row and run the pipeline twice. The dedup check-then-create in
    start_analysis_route is guarded by a per-session asyncio.Lock."""
    with TestClient(app) as client:
        resp = client.post("/api/sessions")
        session_id = resp.json()["sessionId"]
        client.post(
            f"/api/sessions/{session_id}/artefacts",
            data={"kind": "document-still"},
            files={"file": ("doc.jpg", _document_jpeg(305), "image/jpeg")},
        )

        import concurrent.futures

        def _post():
            return client.post(f"/api/sessions/{session_id}/analyse")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(lambda _: _post(), range(2)))

        run_ids = {r.json()["runId"] for r in responses}
        assert len(run_ids) == 1, f"expected one run id, got {run_ids}"

        from sqlalchemy import select

        from app.storage.b1_models import B1Run
        from app.storage.db import async_session_factory

        async def _run_count() -> int:
            async with async_session_factory() as db:
                rows = (
                    await db.execute(select(B1Run).where(B1Run.session_id == session_id))
                ).scalars().all()
                return len(rows)

        assert asyncio.run(_run_count()) == 1


def test_artefact_bytes_land_at_expected_path():
    with TestClient(app) as client:
        resp = client.post("/api/sessions")
        session_id = resp.json()["sessionId"]
        data = _document_jpeg(303)
        resp = client.post(
            f"/api/sessions/{session_id}/artefacts",
            data={"kind": "document-still"},
            files={"file": ("doc.jpg", data, "image/jpeg")},
        )
        assert resp.status_code == 200, resp.text

        from app.config import settings
        expected_dir = settings.b1_artefact_root / session_id
        matches = list(expected_dir.glob("document-still.*"))
        assert len(matches) == 1
        assert matches[0].read_bytes() == data
