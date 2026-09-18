"""FastAPI dependency that validates the session cookie and injects the
current Officer into route handlers -- applied to every endpoint that
touches screening data (see app/api/screening.py, app/api/history.py) and,
via `get_officer_from_ws_cookie`, the WS stream.

WS auth note: a WebSocket handshake is still a plain HTTP request (the
upgrade happens after it succeeds), so the browser attaches the session
cookie to it exactly as it would any other same-site request -- Starlette
exposes it as `websocket.cookies`. There is no separate "WS auth" mechanism
here; it's the same cookie, read the same way, just off `WebSocket` instead
of `Request`.
"""
from __future__ import annotations

from fastapi import Cookie, Depends, HTTPException, WebSocket
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.storage.db import get_session

from .models import Officer
from .repositories import get_officer_for_token


async def require_officer(
    db: AsyncSession = Depends(get_session),
    session_token: str | None = Cookie(default=None, alias=settings.session_cookie_name),
) -> Officer:
    if session_token is None:
        raise HTTPException(401, "not authenticated")
    officer = await get_officer_for_token(db, session_token)
    if officer is None or not officer.active:
        raise HTTPException(401, "not authenticated")
    return officer


async def require_admin(officer: Officer = Depends(require_officer)) -> Officer:
    """Layered on top of require_officer: authenticates first (401 if not
    logged in), then checks role (403 if logged in but not admin).

    Not every route that needs an admin check wants it unconditionally as a
    route-level `Depends` -- app/api/dashboard.py's scope='all' branch is
    only admin-gated when that specific query param is passed, so it calls
    `await require_admin(officer)` directly instead. That works because this
    is a plain async function; `Depends(require_officer)` is only resolved
    by FastAPI's own injection machinery, not by a bare Python call, so
    passing an already-authenticated Officer in explicitly bypasses it
    (and the redundant DB lookup it would otherwise trigger) while still
    reusing this exact role check.
    """
    if officer.role != "admin":
        raise HTTPException(403, "admin role required")
    return officer


async def get_officer_from_ws_cookie(websocket: WebSocket, db: AsyncSession) -> Officer | None:
    token = websocket.cookies.get(settings.session_cookie_name)
    if token is None:
        return None
    officer = await get_officer_for_token(db, token)
    if officer is None or not officer.active:
        return None
    return officer
