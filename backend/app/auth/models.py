"""Officer accounts and server-side sessions.

See app/auth/routes.py's module docstring for why sessions are a server-side
table (opaque token) rather than a JWT -- the short version is that it's
what makes `POST /auth/logout` a real revocation instead of a documented
lie.
"""
from __future__ import annotations

import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.storage.db import Base


class Officer(Base):
    __tablename__ = "officers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # Human-readable badge/login id, e.g. "OFF-2291" -- what the officer
    # types into the login form and what shows up in audit records
    # (BACKEND_BRIEF.md §8.2's officer_id), not the surrogate `id`.
    officer_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    name: Mapped[str] = mapped_column(String)
    # argon2id hash (app/auth/security.py). Never the plaintext password,
    # never logged.
    password_hash: Mapped[str] = mapped_column(String)
    # 'officer' | 'supervisor' | 'admin'. Plain string column, no DB-level
    # enum/CHECK constraint (consistent with the rest of this M1-sized
    # schema, see storage/models.py's SessionRow docstring) -- validity is
    # enforced where it matters, in app/auth/dependencies.py's require_admin.
    role: Mapped[str] = mapped_column(String, default="officer")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))


class OfficerSession(Base):
    """One row per logged-in session. The cookie carries only `token`
    (a high-entropy opaque string) -- everything else is looked up here on
    every request, which is what makes logout and 'revoke this officer's
    sessions' both real operations instead of promises."""

    __tablename__ = "officer_sessions"

    token: Mapped[str] = mapped_column(String, primary_key=True)
    officer_id: Mapped[str] = mapped_column(
        String, ForeignKey("officers.officer_id", ondelete="CASCADE"), index=True,
    )
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), index=True)


class PasswordResetRequest(Base):
    """An officer-initiated 'I can't log in, please reset my password'
    request, filed unauthenticated (app/auth/routes.py's
    request_password_reset) and actioned by an admin
    (resolve_reset_request_route). `reference_code` is what the requesting
    officer is given back to quote when following up -- it exists on every
    row (real or would-be) generated the same way regardless of whether the
    officer_id turned out to be real, so its shape never leaks that."""

    __tablename__ = "password_reset_requests"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    officer_id: Mapped[str] = mapped_column(
        String, ForeignKey("officers.officer_id", ondelete="CASCADE"), index=True,
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 'pending' | 'resolved' -- plain string column, no DB-level enum/CHECK,
    # consistent with Officer.role above.
    status: Mapped[str] = mapped_column(String, default="pending")
    reference_code: Mapped[str] = mapped_column(String, unique=True, index=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # The admin Officer who resolved it. SET NULL (not CASCADE) on delete --
    # an admin account going away shouldn't delete the historical record
    # that they were the one who handled this request.
    resolved_by: Mapped[str | None] = mapped_column(
        String, ForeignKey("officers.officer_id", ondelete="SET NULL"), nullable=True,
    )
