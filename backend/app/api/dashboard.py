"""GET /api/v1/dashboard/summary -- a new, additive feature (analytics over
already-sealed sessions), separate from the in-progress M2 forensics work.
Not part of BACKEND_BRIEF.md §7's original endpoint list; follows the same
auth/response conventions as the endpoints that are (app/api/history.py,
app/api/screening.py).

Access control (this is a real boundary, not a UI convenience -- see
tests/api/test_dashboard_api.py):
- scope='me' is always allowed for any authenticated officer, and always
  returns only THEIR OWN data -- an officer_id query param is accepted for
  API shape symmetry with scope='all' but silently ignored here, never
  honoured, so a non-admin cannot use it to view another officer's data.
- scope='all' requires the admin role (app/auth/dependencies.py's
  require_admin, called directly rather than as a route-level Depends --
  see that function's docstring for why).
"""
from __future__ import annotations

import datetime as dt
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import require_admin, require_officer
from app.auth.models import Officer
from app.contracts.wire import to_wire
from app.storage.dashboard_repository import DashboardFilters, get_dashboard_summary
from app.storage.db import get_session

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])

DEFAULT_RANGE_DAYS = 30


@router.get("/summary")
async def dashboard_summary(
    scope: Literal["me", "all"] = "me",
    from_date: Optional[dt.date] = Query(default=None),
    to_date: Optional[dt.date] = Query(default=None),
    officer_id: Optional[str] = Query(default=None),
    lane_id: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_session),
    officer: Officer = Depends(require_officer),
):
    resolved_to = to_date or dt.date.today()
    resolved_from = from_date or (resolved_to - dt.timedelta(days=DEFAULT_RANGE_DAYS - 1))

    if scope == "all":
        # See app/auth/dependencies.py's require_admin docstring: called
        # directly (not `Depends(require_admin)` on the route) because the
        # admin requirement is conditional on `scope`, not unconditional --
        # scope='me' must stay open to every authenticated officer.
        await require_admin(officer)
        filters = DashboardFilters(
            from_date=resolved_from, to_date=resolved_to, officer_id=officer_id, lane_id=lane_id,
        )
    else:
        # scope='me': officer_id/lane_id filters are accepted on the wire
        # (same query-param shape as scope='all') but never applied -- the
        # only officer_id that can ever narrow a scope='me' query is the
        # caller's own, taken from their session, not from the query
        # string. This is what makes "pass someone else's officer_id under
        # scope=me" a no-op instead of a data leak.
        filters = DashboardFilters(from_date=resolved_from, to_date=resolved_to, officer_id=officer.officer_id)

    summary = await get_dashboard_summary(db, scope=scope, filters=filters)
    return to_wire(summary)
