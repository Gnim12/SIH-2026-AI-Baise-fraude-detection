"""Repository functions over the `officers` / `officer_sessions` tables."""
from __future__ import annotations

import datetime
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

from .models import Officer, OfficerSession, PasswordResetRequest


async def get_officer_by_officer_id(db: AsyncSession, officer_id: str) -> Optional[Officer]:
    result = await db.execute(select(Officer).where(Officer.officer_id == officer_id))
    return result.scalar_one_or_none()


async def create_session(db: AsyncSession, token: str, officer_id: str) -> OfficerSession:
    now = datetime.datetime.now(datetime.timezone.utc)
    row = OfficerSession(
        token=token, officer_id=officer_id, created_at=now,
        expires_at=now + datetime.timedelta(hours=settings.session_ttl_hours),
    )
    db.add(row)
    await db.commit()
    return row


async def get_officer_for_token(db: AsyncSession, token: str) -> Optional[Officer]:
    result = await db.execute(select(OfficerSession).where(OfficerSession.token == token))
    session_row = result.scalar_one_or_none()
    if session_row is None:
        return None
    now = datetime.datetime.now(datetime.timezone.utc)
    expires_at = session_row.expires_at
    if expires_at.tzinfo is None:  # sqlite drops tzinfo on round-trip
        expires_at = expires_at.replace(tzinfo=datetime.timezone.utc)
    if expires_at < now:
        await db.execute(delete(OfficerSession).where(OfficerSession.token == token))
        await db.commit()
        return None
    return await get_officer_by_officer_id(db, session_row.officer_id)


async def delete_session(db: AsyncSession, token: str) -> None:
    await db.execute(delete(OfficerSession).where(OfficerSession.token == token))
    await db.commit()


async def delete_sessions_for_officer(db: AsyncSession, officer_id: str) -> None:
    """Revoke every existing session for `officer_id`. Used by password
    reset resolution (routes.py's resolve_reset_request_route) so a
    password change actually ends any session started with the old
    password, not just prevents new logins with it."""
    await db.execute(delete(OfficerSession).where(OfficerSession.officer_id == officer_id))
    await db.commit()


async def create_reset_request(
    db: AsyncSession, *, officer_id: str, reason: str | None, reference_code: str,
) -> PasswordResetRequest:
    now = datetime.datetime.now(datetime.timezone.utc)
    row = PasswordResetRequest(
        officer_id=officer_id, reason=reason, status="pending",
        reference_code=reference_code, created_at=now,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def get_pending_reset_requests(db: AsyncSession) -> list[PasswordResetRequest]:
    result = await db.execute(
        select(PasswordResetRequest)
        .where(PasswordResetRequest.status == "pending")
        .order_by(PasswordResetRequest.created_at)
    )
    return list(result.scalars().all())


async def get_reset_request_by_id(db: AsyncSession, request_id: int) -> Optional[PasswordResetRequest]:
    result = await db.execute(select(PasswordResetRequest).where(PasswordResetRequest.id == request_id))
    return result.scalar_one_or_none()


async def resolve_reset_request(
    db: AsyncSession, request: PasswordResetRequest, *, password_hash: str, resolved_by: str,
) -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    officer = await get_officer_by_officer_id(db, request.officer_id)
    officer.password_hash = password_hash
    request.status = "resolved"
    request.resolved_at = now
    request.resolved_by = resolved_by
    await db.commit()
