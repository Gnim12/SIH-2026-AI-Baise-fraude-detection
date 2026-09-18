"""POST /api/v1/auth/login, /logout, GET /api/v1/auth/me.

**Session strategy: server-side session table, not a JWT.** The cookie
carries an opaque, high-entropy token (app/auth/security.py's
new_session_token); app/auth/models.py's OfficerSession maps it to an
officer_id server-side. Chosen over a JWT because:

- `POST /auth/logout` needs to be a real revocation, not a documented
  no-op. A stateless JWT can only be invalidated client-side (the token
  stays valid, replayable, until it expires on its own) unless you also
  build a server-side denylist -- at which point you're maintaining
  server-side session state anyway, just as an exception list instead of
  the primary record. Doing that as the primary record is simpler.
- This is a small, single-service system (BACKEND_BRIEF.md §2: "No Celery
  in v1", one Postgres). A DB lookup per authenticated request is not the
  distributed-systems problem JWTs exist to solve here.

Real security implication of this choice: logout is immediate and complete
-- the row is deleted, the token is dead everywhere, on the next request.
The tradeoff is a DB read on every authenticated request (require_officer),
which is fine at this system's scale.
"""
from __future__ import annotations

import datetime
import logging

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.storage.db import get_session

from .dependencies import require_admin, require_officer
from .models import Officer
from .rate_limit import check_and_record
from .repositories import (
    create_reset_request,
    create_session,
    delete_session,
    delete_sessions_for_officer,
    get_officer_by_officer_id,
    get_pending_reset_requests,
    get_reset_request_by_id,
    resolve_reset_request,
)
from .security import hash_password, new_reference_code, new_session_token, new_temporary_password, verify_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
admin_router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


class LoginRequest(BaseModel):
    officer_id: str
    password: str


def _officer_out(officer: Officer) -> dict:
    return {
        "id": officer.id,
        "officerId": officer.officer_id,
        "name": officer.name,
        "role": officer.role,
    }


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.session_cookie_name,
        value=token,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="lax",
        max_age=settings.session_ttl_hours * 3600,
        path="/",
    )


@router.post("/login")
async def login(body: LoginRequest, response: Response, db: AsyncSession = Depends(get_session)):
    officer = await get_officer_by_officer_id(db, body.officer_id)
    # Same 401 + message whether the officer_id doesn't exist or the
    # password is wrong -- distinguishing them lets a caller enumerate
    # valid officer_ids.
    if officer is None or not officer.active or not verify_password(body.password, officer.password_hash):
        raise HTTPException(401, "invalid officer id or password")

    token = new_session_token()
    await create_session(db, token, officer.officer_id)
    _set_session_cookie(response, token)
    return {"officer": _officer_out(officer)}


@router.post("/logout")
async def logout(
    response: Response,
    db: AsyncSession = Depends(get_session),
    session_token: str | None = Cookie(default=None, alias=settings.session_cookie_name),
):
    # Real revocation, not client-side-only: the session row is deleted, so
    # the token is dead on the server immediately, not just forgotten by
    # this one browser (see module docstring). No-op (still 200) if there
    # was no cookie or it was already invalid -- logout is idempotent.
    if session_token is not None:
        await delete_session(db, session_token)
    response.delete_cookie(key=settings.session_cookie_name, path="/")
    return {"ok": True}


@router.get("/me")
async def me(officer: Officer = Depends(require_officer)):
    return {"officer": _officer_out(officer)}


class ResetRequestIn(BaseModel):
    officer_id: str
    reason: str | None = None


@router.post("/reset-requests", status_code=202)
async def request_password_reset(body: ResetRequestIn, db: AsyncSession = Depends(get_session)):
    """Public (no auth) by necessity -- an officer filing this is often
    exactly the one who can't log in. That unauthenticated surface is why
    every response here is deliberately identical whether officer_id is
    real or not: a caller who could tell the two apart could enumerate
    valid officer_ids for free. A row in password_reset_requests is only
    ever created for a real officer_id, but the reference code, status
    code, and body are the same regardless -- see security.new_reference_code
    and the module docstring on app/auth/rate_limit.py for the other half
    of hardening this (throttling).
    """
    allowed = check_and_record(
        body.officer_id,
        limit=settings.reset_request_rate_limit,
        window=datetime.timedelta(hours=settings.reset_request_rate_window_hours),
    )
    # Logged either way (real officer_id or not) -- this is the audit trail
    # for abuse investigation; it is never surfaced to the caller.
    logger.info(
        "password reset request attempt: officer_id=%s allowed=%s", body.officer_id, allowed,
    )
    if not allowed:
        raise HTTPException(429, "too many reset requests for this officer id, try again later")

    reference_code = new_reference_code()
    officer = await get_officer_by_officer_id(db, body.officer_id)
    if officer is not None:
        await create_reset_request(db, officer_id=officer.officer_id, reason=body.reason, reference_code=reference_code)

    return {
        "referenceCode": reference_code,
        "message": "If that officer ID exists, a password reset request has been received and will be reviewed by an admin.",
    }


def _reset_request_out(request) -> dict:
    return {
        "id": request.id,
        "officerId": request.officer_id,
        "reason": request.reason,
        "createdAt": request.created_at.isoformat(),
    }


@admin_router.get("/reset-requests")
async def list_pending_reset_requests(
    admin: Officer = Depends(require_admin), db: AsyncSession = Depends(get_session),
):
    pending = await get_pending_reset_requests(db)
    return {"requests": [_reset_request_out(r) for r in pending]}


@admin_router.post("/reset-requests/{request_id}/resolve")
async def resolve_reset_request_route(
    request_id: int, admin: Officer = Depends(require_admin), db: AsyncSession = Depends(get_session),
):
    """Generates the new password server-side rather than accepting an
    admin-supplied one: an admin-chosen password is one more place a weak
    or reused password can sneak in, and an arbitrary string in the request
    body is one more thing that ends up in logs/proxies along the way. A
    random 128-bit token has neither problem -- it's returned once, here,
    for the admin to relay to the officer out of band."""
    request = await get_reset_request_by_id(db, request_id)
    if request is None or request.status != "pending":
        raise HTTPException(404, "reset request not found")

    temporary_password = new_temporary_password()
    await resolve_reset_request(
        db, request, password_hash=hash_password(temporary_password), resolved_by=admin.officer_id,
    )
    # The whole point of a password reset is that a compromised-password
    # scenario is actually resolved -- an old session surviving the reset
    # would mean it isn't.
    await delete_sessions_for_officer(db, request.officer_id)

    return {"officerId": request.officer_id, "temporaryPassword": temporary_password}
