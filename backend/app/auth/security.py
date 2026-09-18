"""Password hashing (argon2id, via argon2-cffi -- already a transitive dep
of this project's ML stack, so no new dependency, and argon2id is the
current OWASP-recommended default over bcrypt) and session token
generation. Passwords are never stored or logged in plaintext."""
from __future__ import annotations

import datetime
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

_hasher = PasswordHasher()


def hash_password(plaintext: str) -> str:
    return _hasher.hash(plaintext)


def verify_password(plaintext: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, plaintext)
    except VerifyMismatchError:
        return False


def new_session_token() -> str:
    # 256 bits of entropy, URL-safe -- fine to put directly in a cookie value.
    return secrets.token_urlsafe(32)


def new_reference_code() -> str:
    """A reference code for a password reset request (e.g. "PWR-20260901-
    A1B2C3"). Generated the same way regardless of whether the officer_id
    the request named turned out to exist -- see
    app/auth/routes.py's request_password_reset for why that matters:
    the code's shape must never be a signal of validity."""
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    return f"PWR-{today}-{secrets.token_hex(3).upper()}"


def new_temporary_password() -> str:
    # 128 bits of entropy. This becomes the officer's actual password until
    # they log in and it's rotated by a future reset -- there's no
    # self-service "change password" flow yet -- so it's sized like a real
    # credential, not a short one-time PIN.
    return secrets.token_urlsafe(16)
