"""Password reset request flow (app/auth/routes.py's request_password_reset,
list_pending_reset_requests, resolve_reset_request_route):

- reset request for a real officer_id creates a row; for a fake one, the
  HTTP response is identical but no row is created (enumeration safety).
- rate limiting triggers on repeated requests for the same officer_id.
- non-admin cannot list or resolve requests (403).
- resolving a request changes the password AND invalidates existing
  sessions.
"""
from __future__ import annotations

import asyncio
import datetime

from fastapi.testclient import TestClient

from app.main import app

from .conftest import login, login_admin

RESET_URL = "/api/v1/auth/reset-requests"
ADMIN_LIST_URL = "/api/v1/admin/reset-requests"


def _unique_officer_id(label: str) -> str:
    return f"OFF-{label}-{datetime.datetime.now().strftime('%H%M%S%f')}"


async def _seed_officer(officer_id: str, password: str = "some-password-123") -> None:
    from app.auth.models import Officer
    from app.auth.security import hash_password
    from app.storage.db import async_session_factory

    async with async_session_factory() as db:
        db.add(Officer(
            officer_id=officer_id, name="Test Target", role="officer", active=True,
            password_hash=hash_password(password),
            created_at=datetime.datetime.now(datetime.timezone.utc),
        ))
        await db.commit()


async def _pending_requests_for(officer_id: str):
    from app.auth.repositories import get_pending_reset_requests
    from app.storage.db import async_session_factory

    async with async_session_factory() as db:
        rows = await get_pending_reset_requests(db)
        return [r for r in rows if r.officer_id == officer_id]


def test_reset_request_for_real_officer_creates_row_and_returns_reference_code():
    real_officer_id = _unique_officer_id("REAL")
    asyncio.run(_seed_officer(real_officer_id))

    with TestClient(app) as client:
        resp = client.post(RESET_URL, json={"officer_id": real_officer_id, "reason": "forgot password"})
        assert resp.status_code == 202
        body = resp.json()
        assert body["referenceCode"].startswith("PWR-")

        rows = asyncio.run(_pending_requests_for(real_officer_id))
        assert len(rows) == 1
        assert rows[0].reason == "forgot password"
        assert rows[0].status == "pending"


def test_reset_request_for_fake_officer_returns_same_response_but_creates_no_row():
    real_officer_id = _unique_officer_id("REAL2")
    fake_officer_id = _unique_officer_id("FAKE")
    asyncio.run(_seed_officer(real_officer_id))

    with TestClient(app) as client:
        real_resp = client.post(RESET_URL, json={"officer_id": real_officer_id, "reason": "forgot"})
        fake_resp = client.post(RESET_URL, json={"officer_id": fake_officer_id, "reason": "forgot"})

        # Same shape and status regardless of whether officer_id is real --
        # only the reference code value itself differs (each is freshly
        # generated), which is not observable as a validity signal.
        assert real_resp.status_code == fake_resp.status_code == 202
        assert set(real_resp.json().keys()) == set(fake_resp.json().keys())
        assert real_resp.json()["message"] == fake_resp.json()["message"]

        # Directly against the DB, not just the HTTP response: no row for
        # the fake officer_id.
        rows = asyncio.run(_pending_requests_for(fake_officer_id))
        assert rows == []


def test_rate_limiting_triggers_on_repeated_requests():
    officer_id = f"OFF-RATE-LIMIT-{datetime.datetime.now().microsecond}"
    with TestClient(app) as client:
        from app.config import settings

        statuses = []
        for _ in range(settings.reset_request_rate_limit + 2):
            resp = client.post(RESET_URL, json={"officer_id": officer_id})
            statuses.append(resp.status_code)

        assert statuses[: settings.reset_request_rate_limit] == [202] * settings.reset_request_rate_limit
        assert statuses[settings.reset_request_rate_limit] == 429
        assert statuses[settings.reset_request_rate_limit + 1] == 429


def test_non_admin_cannot_list_or_resolve_reset_requests():
    with TestClient(app) as client:
        login(client)
        assert client.get(ADMIN_LIST_URL).status_code == 403
        assert client.post(f"{ADMIN_LIST_URL}/1/resolve").status_code == 403


def test_resolving_a_request_changes_password_and_invalidates_existing_sessions():
    officer_id = _unique_officer_id("RESOLVE")
    asyncio.run(_seed_officer(officer_id, password="original-password-123"))

    with TestClient(app) as officer_client:
        # Log in as the target officer to get a live session to invalidate.
        login_resp = officer_client.post(
            "/api/v1/auth/login", json={"officer_id": officer_id, "password": "original-password-123"},
        )
        assert login_resp.status_code == 200
        assert officer_client.get("/api/v1/auth/me").status_code == 200

        with TestClient(app) as public_client:
            reset_resp = public_client.post(RESET_URL, json={"officer_id": officer_id, "reason": "locked out"})
            assert reset_resp.status_code == 202

        rows = asyncio.run(_pending_requests_for(officer_id))
        assert len(rows) == 1
        request_id = rows[0].id

        with TestClient(app) as admin_client:
            login_admin(admin_client)
            resolve_resp = admin_client.post(f"{ADMIN_LIST_URL}/{request_id}/resolve")
            assert resolve_resp.status_code == 200
            temp_password = resolve_resp.json()["temporaryPassword"]
            assert temp_password

        # Old session cookie no longer works -- reset invalidated it.
        assert officer_client.get("/api/v1/auth/me").status_code == 401

        # New password works; old one no longer does.
        with TestClient(app) as new_client:
            fresh_login = new_client.post(
                "/api/v1/auth/login", json={"officer_id": officer_id, "password": temp_password},
            )
            assert fresh_login.status_code == 200

        with TestClient(app) as old_pw_client:
            old_login = old_pw_client.post(
                "/api/v1/auth/login", json={"officer_id": officer_id, "password": "original-password-123"},
            )
            assert old_login.status_code == 401

        # The request is no longer pending.
        remaining = asyncio.run(_pending_requests_for(officer_id))
        assert remaining == []
