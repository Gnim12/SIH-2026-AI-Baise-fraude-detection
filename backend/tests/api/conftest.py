import asyncio
import datetime

import pytest


@pytest.fixture(autouse=True)
def _b1_artefact_root_tmp(tmp_path, monkeypatch):
    # Without this, B1b artefact-upload tests would write real bytes under
    # the dev artefact root (app/config.py's default data/artefacts/),
    # permanently polluting the working tree every test run.
    from app.config import settings

    monkeypatch.setattr(settings, "b1_artefact_root", tmp_path / "b1_artefacts")


@pytest.fixture(autouse=True)
def _generous_ocr_budget(monkeypatch):
    # See tests/pipeline/test_orchestrator.py's identical fixture docstring:
    # cold ONNX Runtime session load + CPU inference exceeds the production
    # 700ms OCR budget in this dev sandbox.
    from app.pipeline import orchestrator
    from app.pipeline.branches.ocr import _shared_mrz_reader, _shared_viz_reader

    _shared_viz_reader()
    _shared_mrz_reader()
    monkeypatch.setitem(orchestrator.BUDGET_MS, "ocr", 8000)


# Shared test officer for API tests that need a logged-in session (auth in
# front of the existing screening/history endpoints -- see
# tests/api/test_auth_api.py and test_screening_api.py's login() calls).
TEST_OFFICER_ID = "OFF-2291"
TEST_OFFICER_PASSWORD = "dev-test-password-2291"  # test-only, never a real credential


async def _ensure_test_officer() -> None:
    from app.auth.models import Officer
    from app.auth.repositories import get_officer_by_officer_id
    from app.auth.security import hash_password
    from app.storage.db import async_session_factory

    async with async_session_factory() as db:
        existing = await get_officer_by_officer_id(db, TEST_OFFICER_ID)
        if existing is None:
            db.add(Officer(
                officer_id=TEST_OFFICER_ID, name="Test Officer", role="officer", active=True,
                password_hash=hash_password(TEST_OFFICER_PASSWORD),
                created_at=datetime.datetime.now(datetime.timezone.utc),
            ))
            await db.commit()


def login(client, *, officer_id: str = TEST_OFFICER_ID, password: str = TEST_OFFICER_PASSWORD):
    """Seed the shared test officer (idempotent) and log `client` in.

    Call inside `with TestClient(app) as client:` so app startup has already
    created the tables. `client` keeps the session cookie for subsequent
    requests, same as a browser would."""
    asyncio.run(_ensure_test_officer())
    resp = client.post("/api/v1/auth/login", json={"officer_id": officer_id, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["officer"]


# admin-role test officer -- app/auth/dependencies.py's require_admin gate
# for GET /api/v1/dashboard/summary?scope=all (tests/api/test_dashboard_api.py).
TEST_ADMIN_ID = "OFF-9000"
TEST_ADMIN_PASSWORD = "dev-test-admin-password-9000"


async def _ensure_test_role_officer(officer_id: str, password: str, role: str) -> None:
    from app.auth.models import Officer
    from app.auth.repositories import get_officer_by_officer_id
    from app.auth.security import hash_password
    from app.storage.db import async_session_factory

    async with async_session_factory() as db:
        existing = await get_officer_by_officer_id(db, officer_id)
        if existing is None:
            db.add(Officer(
                officer_id=officer_id, name=f"Test {role.title()}", role=role, active=True,
                password_hash=hash_password(password),
                created_at=datetime.datetime.now(datetime.timezone.utc),
            ))
            await db.commit()


def login_admin(client):
    asyncio.run(_ensure_test_role_officer(TEST_ADMIN_ID, TEST_ADMIN_PASSWORD, "admin"))
    resp = client.post("/api/v1/auth/login", json={"officer_id": TEST_ADMIN_ID, "password": TEST_ADMIN_PASSWORD})
    assert resp.status_code == 200, resp.text
    assert resp.json()["officer"]["role"] == "admin"
    return resp.json()["officer"]
