"""GET /api/v1/dashboard/summary (task brief's "Tests" section, verbatim):

- scope='me' returns only the calling officer's data, even if they pass a
  different officer_id filter (confirm this is actually enforced, not just
  ignored client-side).
- scope='all' as a non-admin returns 403.
- scope='all' as an admin returns aggregated data across officers.
- Date range filtering actually filters (test with a narrow range that
  should exclude some seeded sessions).
- Percentile/aggregation math spot-checked against a known small seeded
  dataset with hand-computable expected values -- don't just trust the SQL,
  verify one case by hand.

Sessions are inserted directly via app/storage/repositories.py's
upsert_session (the same function the real orchestrator/decision endpoint
persists through) rather than run through the full multipart-upload ->
pipeline -> WS -> decision flow tests/api/test_screening_api.py exercises --
that flow is what's actually under test there; here the thing under test is
the aggregation SQL and the access-control boundary, so building
ScreeningSession rows directly is the more direct test, not a shortcut
around anything.

Every test uses its own uuid-based lane_id, but scope='me' deliberately
ignores lane_id (app/api/dashboard.py: only officer_id/lane_id filters are
honoured under scope='all' -- see the task brief's own spec for scope='me').
That means a scope='me' query can't isolate itself from another test's rows
for the same officer_id/date the way a lane_id filter would, so the
`_cleanup_seeded_sessions` autouse fixture below deletes every row this
module inserts at the end of each test -- tests/_test_screening.db is a
persistent sqlite file across local runs (see tests/conftest.py), and
without this, a second run of this file would double-count leftover rows
from the first.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.contracts import OfficerDecision, ScreeningSession, Signal
from app.main import app
from app.storage.db import async_session_factory
from app.storage.models import SessionRow
from app.storage.repositories import upsert_session

from .conftest import TEST_OFFICER_ID, login, login_admin

_seeded_session_ids: list[str] = []


@pytest.fixture(autouse=True)
def _cleanup_seeded_sessions():
    yield
    if not _seeded_session_ids:
        return
    ids = list(_seeded_session_ids)
    _seeded_session_ids.clear()

    async def _delete_all():
        async with async_session_factory() as db:
            await db.execute(delete(SessionRow).where(SessionRow.session_id.in_(ids)))
            await db.commit()

    asyncio.run(_delete_all())


def _iso(date_str: str, hour: int = 10) -> str:
    return dt.datetime.fromisoformat(f"{date_str}T{hour:02d}:00:00+00:00").isoformat()


def _session(
    *, officer_id: str, lane_id: str, date_str: str, band: str, decision: str,
    total_ms: float, signals: list[Signal] | None = None, coverage_flags: list[str] | None = None,
) -> ScreeningSession:
    started_at = _iso(date_str)
    return ScreeningSession(
        session_id=str(uuid.uuid4()), lane_id=lane_id, officer_id=officer_id, started_at=started_at,
        band=band, risk=50.0, confidence=0.9, signals=signals or [], coverage_flags=coverage_flags or [],
        timing_ms={"total": total_ms},
        officer_decision=OfficerDecision(
            decision=decision, note="test" if decision != band else "", decided_at=started_at,
            override=(decision != band),
        ),
        sealed=True,
    )


def _seed(*sessions: ScreeningSession) -> None:
    _seeded_session_ids.extend(s.session_id for s in sessions)

    async def _insert_all():
        async with async_session_factory() as db:
            for s in sessions:
                await upsert_session(db, s)
    asyncio.run(_insert_all())


def test_scope_me_ignores_officer_id_filter_and_returns_only_own_data():
    """The core access-control claim: scope='me' must not leak another
    officer's sessions even when the caller explicitly asks for them via
    the officer_id query param."""
    lane = f"LANE-{uuid.uuid4().hex[:8]}"
    date = "2031-01-10"
    _seed(
        _session(officer_id=TEST_OFFICER_ID, lane_id=lane, date_str=date, band="CLEAR", decision="CLEAR", total_ms=900),
        _session(officer_id=TEST_OFFICER_ID, lane_id=lane, date_str=date, band="HOLD", decision="HOLD", total_ms=1800),
    )
    other_officer_id = "OFF-3310"
    _seed(
        _session(officer_id=other_officer_id, lane_id=lane, date_str=date, band="CLEAR", decision="CLEAR", total_ms=1000),
        _session(officer_id=other_officer_id, lane_id=lane, date_str=date, band="CLEAR", decision="CLEAR", total_ms=1000),
        _session(officer_id=other_officer_id, lane_id=lane, date_str=date, band="CLEAR", decision="CLEAR", total_ms=1000),
    )

    with TestClient(app) as client:
        login(client)  # TEST_OFFICER_ID
        resp = client.get(
            "/api/v1/dashboard/summary",
            params={"scope": "me", "from_date": date, "to_date": date, "officer_id": other_officer_id},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # 2, not 5: the officer_id=other_officer_id filter must be a no-op
        # under scope=me, not a way to view someone else's 3 sessions.
        assert body["totalSessions"] == 2
        # to_wire() (app/contracts/wire.py) dumps with exclude_none=True, so
        # a None filter_options is an absent key on the wire, not a null.
        assert "filterOptions" not in body


def test_scope_all_as_non_admin_returns_403():
    with TestClient(app) as client:
        login(client)  # plain 'officer' role
        resp = client.get("/api/v1/dashboard/summary", params={"scope": "all"})
        assert resp.status_code == 403


def test_scope_all_as_admin_returns_aggregated_data_across_officers():
    lane = f"LANE-{uuid.uuid4().hex[:8]}"
    date = "2031-02-10"
    _seed(
        _session(officer_id=TEST_OFFICER_ID, lane_id=lane, date_str=date, band="CLEAR", decision="CLEAR", total_ms=900),
        _session(officer_id="OFF-3310", lane_id=lane, date_str=date, band="HOLD", decision="HOLD", total_ms=1800),
        _session(officer_id="OFF-3310", lane_id=lane, date_str=date, band="SECONDARY", decision="SECONDARY", total_ms=1300),
    )

    with TestClient(app) as client:
        login_admin(client)
        resp = client.get(
            "/api/v1/dashboard/summary",
            params={"scope": "all", "from_date": date, "to_date": date, "lane_id": lane},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["totalSessions"] == 3  # across BOTH officers, not just the admin's own (0) sessions
        assert set(body["filterOptions"]["officerIds"]) == {TEST_OFFICER_ID, "OFF-3310"}
        assert body["filterOptions"]["laneIds"] == [lane]


def test_date_range_filtering_excludes_out_of_range_sessions():
    lane = f"LANE-{uuid.uuid4().hex[:8]}"
    in_range_date = "2031-03-15"
    out_of_range_date = "2031-04-20"  # well outside the narrow range queried below
    _seed(
        _session(officer_id=TEST_OFFICER_ID, lane_id=lane, date_str=in_range_date, band="CLEAR", decision="CLEAR", total_ms=900),
        _session(officer_id=TEST_OFFICER_ID, lane_id=lane, date_str=in_range_date, band="CLEAR", decision="CLEAR", total_ms=950),
        _session(officer_id=TEST_OFFICER_ID, lane_id=lane, date_str=out_of_range_date, band="HOLD", decision="HOLD", total_ms=2000),
    )

    with TestClient(app) as client:
        login(client)
        resp = client.get(
            "/api/v1/dashboard/summary",
            params={"scope": "me", "from_date": "2031-03-10", "to_date": "2031-03-20"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["totalSessions"] == 2  # the out-of-range HOLD session must not be counted
        bands = {b: c for bucket in body["decisionPatterns"]["systemBandByDay"] for b, c in bucket["counts"].items()}
        assert bands.get("HOLD") is None


def test_percentile_and_band_aggregation_matches_hand_calculation():
    """Four sessions, two days, hand-computable p50/p95 and override rate --
    see the assertions below for the arithmetic. This is deliberately the
    exact same shape verified by hand against real Postgres during
    development (backend/app/storage/dashboard_repository.py); reproducing
    it here as an automated test against sqlite proves the two dialects'
    query paths agree, not just that the SQL runs without error."""
    lane = f"LANE-{uuid.uuid4().hex[:8]}"
    admin_id = None  # filled in below once we know the admin's officer_id

    day1, day2 = "2031-05-10", "2031-05-11"
    sessions = [
        _session(officer_id="OFF-2291", lane_id=lane, date_str=day1, band="CLEAR", decision="CLEAR", total_ms=1000.0),
        _session(officer_id="OFF-2291", lane_id=lane, date_str=day1, band="HOLD", decision="HOLD", total_ms=2000.0,
                  signals=[Signal(id="s1", code="MRZ_CHECKDIGIT_DOB", module="validation", severity="high",
                                   weight=30, detail="x", convergence_group="g1")],
                  coverage_flags=["no_biometric"]),
        _session(officer_id="OFF-2291", lane_id=lane, date_str=day2, band="SECONDARY", decision="HOLD", total_ms=3000.0,
                  signals=[Signal(id="s2", code="FACE_MISMATCH", module="face", severity="high", weight=20, detail="x")]),
        _session(officer_id="OFF-2291", lane_id=lane, date_str=day2, band="CLEAR", decision="CLEAR", total_ms=1500.0,
                  coverage_flags=["no_biometric"]),
    ]
    _seed(*sessions)

    with TestClient(app) as client:
        login_admin(client)
        resp = client.get(
            "/api/v1/dashboard/summary",
            params={"scope": "all", "from_date": day1, "to_date": day2, "lane_id": lane},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

    # --- hand-computed expected values -----------------------------------
    # total_ms sorted: [1000, 1500, 2000, 3000], n=4
    # p50: linear-interpolation index k=(4-1)*0.5=1.5 -> between values[1]=1500
    #      and values[2]=2000, weight 0.5/0.5 -> 1500*0.5 + 2000*0.5 = 1750
    # p95: k=(4-1)*0.95=2.85 -> between values[2]=2000 (weight 0.15) and
    #      values[3]=3000 (weight 0.85) -> 2000*0.15 + 3000*0.85 = 2850
    assert body["totalSessions"] == 4
    assert body["operational"]["latency"]["sampleSize"] == 4
    assert abs(body["operational"]["latency"]["p50Ms"] - 1750.0) < 1e-6
    assert abs(body["operational"]["latency"]["p95Ms"] - 2850.0) < 1e-6

    # override = officer_decision.decision != band: only session 3
    # (band=SECONDARY, decision=HOLD) overrides. 1/4 = 25%.
    assert abs(body["decisionPatterns"]["overrideRatePct"] - 25.0) < 1e-9

    # system band counts: day1 {CLEAR:1, HOLD:1}, day2 {SECONDARY:1, CLEAR:1}
    by_day = {b["date"]: b["counts"] for b in body["decisionPatterns"]["systemBandByDay"]}
    assert by_day[day1] == {"CLEAR": 1, "HOLD": 1}
    assert by_day[day2] == {"SECONDARY": 1, "CLEAR": 1}

    # coverage flag frequency: no_biometric appears on exactly 2 sessions
    coverage = {c["flag"]: c["sessionCount"] for c in body["operational"]["coverageFlagFrequency"]}
    assert coverage == {"no_biometric": 2}

    # exactly one session (the HOLD one) carries a convergence_group
    assert body["fraudSignals"]["sessionsWithConvergenceGroup"] == 1
    assert abs(body["fraudSignals"]["sessionsWithConvergenceGroupPct"] - 25.0) < 1e-9

    # both signal codes appear once each
    codes = {c["code"]: c["count"] for c in body["fraudSignals"]["topSignalCodes"]}
    assert codes == {"MRZ_CHECKDIGIT_DOB": 1, "FACE_MISMATCH": 1}
