"""In-process rate limiting for POST /api/v1/auth/reset-requests
(app/auth/routes.py) -- the one auth endpoint that's unauthenticated by
necessity (an officer who can't log in still needs to be able to ask for
help), which is exactly what makes it a spam/enumeration-probe target.

In-memory rather than DB- or Redis-backed: this is a single-service system
with no Celery/no distributed workers in v1 (app/config.py, BACKEND_BRIEF.md
§2), so there's exactly one process for this state to live in. The tradeoff
-- counts reset on a process restart -- is fine for "a few requests per
hour" abuse throttling; it is not a security boundary on its own (the
enumeration-safety property comes from the endpoint always returning the
same response, see routes.py), just a spam brake.
"""
from __future__ import annotations

import collections
import datetime

_attempts: dict[str, collections.deque[datetime.datetime]] = collections.defaultdict(collections.deque)


def check_and_record(officer_id: str, *, limit: int, window: datetime.timedelta) -> bool:
    """Record an attempt for `officer_id` and report whether it's within
    the rate limit. Prunes attempts older than `window` first, so the
    window slides rather than resetting on a fixed boundary."""
    now = datetime.datetime.now(datetime.timezone.utc)
    attempts = _attempts[officer_id]
    while attempts and now - attempts[0] > window:
        attempts.popleft()
    if len(attempts) >= limit:
        return False
    attempts.append(now)
    return True
