"""Auth API tests (this task's brief item "Tests"):

- login w/ correct credentials sets a cookie, /auth/me returns the officer
- login w/ wrong password -> 401, no cookie
- protected endpoint w/ no cookie -> 401
- protected endpoint w/ a valid cookie -> succeeds
- logout clears the session -> subsequent /auth/me -> 401
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

from .conftest import TEST_OFFICER_ID, TEST_OFFICER_PASSWORD, login


def test_login_success_sets_cookie_and_me_returns_officer():
    with TestClient(app) as client:
        officer = login(client)
        assert officer["officerId"] == TEST_OFFICER_ID
        assert officer["role"] == "officer"
        assert "password" not in officer and "passwordHash" not in officer

        # httpOnly cookie is invisible to JS but the test client (like a
        # browser) still holds it and sends it on the next request.
        assert any(c.name == "screening_session" for c in client.cookies.jar)

        me = client.get("/api/v1/auth/me")
        assert me.status_code == 200
        assert me.json()["officer"]["officerId"] == TEST_OFFICER_ID


def test_login_wrong_password_returns_401_and_sets_no_cookie():
    with TestClient(app) as client:
        import asyncio

        from .conftest import _ensure_test_officer
        asyncio.run(_ensure_test_officer())

        resp = client.post(
            "/api/v1/auth/login",
            json={"officer_id": TEST_OFFICER_ID, "password": "definitely-wrong"},
        )
        assert resp.status_code == 401
        assert "screening_session" not in resp.cookies
        assert not any(c.name == "screening_session" for c in client.cookies.jar)


def test_protected_endpoint_without_cookie_returns_401():
    with TestClient(app) as client:
        resp = client.post("/api/v1/screening", data={"documentCount": "0"})
        assert resp.status_code == 401

        resp = client.get("/api/v1/history")
        assert resp.status_code == 401


def test_protected_endpoint_with_valid_cookie_succeeds():
    with TestClient(app) as client:
        login(client)
        # documentCount=0 clears auth (401 would mean the dependency didn't
        # run) and instead reaches the route's own 422 validation --
        # proving the request passed require_officer.
        resp = client.post("/api/v1/screening", data={"documentCount": "0"})
        assert resp.status_code == 422

        resp = client.get("/api/v1/history")
        assert resp.status_code == 200


def test_logout_clears_session():
    with TestClient(app) as client:
        login(client)
        assert client.get("/api/v1/auth/me").status_code == 200

        logout_resp = client.post("/api/v1/auth/logout")
        assert logout_resp.status_code == 200

        assert client.get("/api/v1/auth/me").status_code == 401
        # Protected app endpoints are locked out too, not just /auth/me.
        assert client.get("/api/v1/history").status_code == 401
