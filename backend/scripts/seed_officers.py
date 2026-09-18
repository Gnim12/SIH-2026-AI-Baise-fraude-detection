#!/usr/bin/env python
"""DEV-ONLY seed data. Creates a handful of officer accounts, with known
passwords, so the login screen has something to log in with on a local/demo
box.

**Never run this against a production database and never reuse these
credentials there.** These are throwaway dev passwords committed to source
control in plaintext right here in this file -- that is only acceptable
because it is understood to be dev/demo seed data, not a real account
provisioning path. A real deployment needs a real officer-provisioning flow
(admin-created accounts, forced password reset on first login, etc.), which
is out of scope for this v1 auth slice.

Run: python scripts/seed_officers.py
"""
from __future__ import annotations

import asyncio
import datetime

from app.auth.models import Officer
from app.auth.repositories import get_officer_by_officer_id
from app.auth.security import hash_password
from app.storage.db import Base, async_session_factory, engine

# officer_id, name, password, role
DEV_OFFICERS = [
    ("OFF-2291", "A. Okonkwo", "dev-password-2291", "officer"),
    ("OFF-3310", "R. Singh", "dev-password-3310", "officer"),
    ("OFF-0001", "M. Alvarez", "dev-password-0001", "supervisor"),
    # dev-only admin account: the only role that can pass require_admin
    # (app/auth/dependencies.py) and so the only one that can request
    # GET /api/v1/dashboard/summary?scope=all (app/api/dashboard.py).
    ("OFF-9000", "S. Kowalski", "dev-password-9000", "admin"),
]


async def main() -> None:
    import app.auth.models  # noqa: F401
    import app.storage.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session_factory() as db:
        for officer_id, name, password, role in DEV_OFFICERS:
            existing = await get_officer_by_officer_id(db, officer_id)
            if existing is not None:
                print(f"skip {officer_id}: already exists")
                continue
            db.add(Officer(
                officer_id=officer_id, name=name, role=role, active=True,
                password_hash=hash_password(password),
                created_at=datetime.datetime.now(datetime.timezone.utc),
            ))
            print(f"created {officer_id} ({name}, role={role}) -- dev password: {password}")
        await db.commit()


if __name__ == "__main__":
    asyncio.run(main())
