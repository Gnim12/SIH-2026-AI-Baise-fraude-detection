"""Real SQL aggregation over `sessions` for GET /api/v1/dashboard/summary
(app/api/dashboard.py). No fetch-everything-then-aggregate-in-Python here --
every facet is a GROUP BY / date_trunc / percentile_cont query, because this
has to scale with real session volume, not the row count of a demo box.

Two execution paths, like app/storage/repositories.py's own `_is_sqlite`
precedent (see that module's docstring): Postgres is the real path --
`percentile_cont`, `date_trunc`, `json_array_elements(_text)` do the actual
work in the database. SQLite (the local test suite only, tests/conftest.py)
has no `percentile_cont` and a different JSON-array-unnesting function
(`json_each` instead of `json_array_elements`), so it gets its own queries;
percentiles specifically fall back to a small Python computation over a
single sorted column pulled back from one still-server-side-aggregated
query (day-bucketing, filtering, and NULL-exclusion all still happen in
SQL) -- not a "fetch everything" bypass, just the one statistic SQLite's
SQL dialect cannot compute for us.

`payload` is stored as plain `json` (see storage/models.py), not `jsonb` --
every operator used here (`->`, `->>`, `json_array_elements`,
`json_array_elements_text`) is defined for both, so nothing here assumes
jsonb.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import case, cast, func, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts.dashboard import (
    BandDayBucket,
    CoverageFlagFrequency,
    DailyCount,
    DashboardFilterOptions,
    DashboardSummary,
    DecisionPatterns,
    FraudSignals,
    LatencyPercentiles,
    OperationalMetrics,
    OverrideDayBucket,
    SignalCodeFrequency,
)

from .models import SessionRow

TOP_SIGNAL_CODES_N = 15


@dataclass(frozen=True)
class DashboardFilters:
    """Fully-resolved query filters -- scope='me' vs 'all' and the
    admin/role check both happen in app/api/dashboard.py before this is
    built. This module only ever sees the final officer_id/lane_id it
    should filter by (or None for "no filter"), never a scope string."""

    from_date: dt.date
    to_date: dt.date  # inclusive
    officer_id: Optional[str] = None
    lane_id: Optional[str] = None


def _is_sqlite(db: AsyncSession) -> bool:
    return db.bind.dialect.name == "sqlite"


def _date_window(f: DashboardFilters) -> tuple[str, str]:
    """[from_ts, to_ts) as ISO strings. Safe to compare lexicographically
    against started_at (always `datetime.now(UTC).isoformat()`, see
    pipeline/orchestrator.py's `_now_iso`): both boundaries are strict
    prefixes of any timestamp that should fall on that side of them, and a
    string that is a strict prefix of another always sorts before it."""
    from_ts = f"{f.from_date.isoformat()}T00:00:00"
    to_ts = f"{(f.to_date + dt.timedelta(days=1)).isoformat()}T00:00:00"
    return from_ts, to_ts


def _core_conditions(f: DashboardFilters):
    from_ts, to_ts = _date_window(f)
    conditions = [
        SessionRow.sealed.is_(True),
        SessionRow.started_at >= from_ts,
        SessionRow.started_at < to_ts,
    ]
    if f.officer_id:
        conditions.append(SessionRow.officer_id == f.officer_id)
    if f.lane_id:
        conditions.append(SessionRow.lane_id == f.lane_id)
    return conditions


def _day_expr(is_sqlite: bool):
    if is_sqlite:
        # started_at always starts 'YYYY-MM-DDT...' -- see _date_window.
        return func.substr(SessionRow.started_at, 1, 10)
    return func.to_char(
        func.date_trunc("day", cast(SessionRow.started_at, postgresql.TIMESTAMP(timezone=True))),
        "YYYY-MM-DD",
    )


def _percentile(sorted_values: list[float], pct: float) -> Optional[float]:
    """Linear-interpolation percentile matching Postgres's percentile_cont
    exactly (same formula), so the SQLite test path and the Postgres
    production path agree on a hand-computable example -- see
    tests/api/test_dashboard_api.py's percentile spot-check."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    f_idx, c_idx = math.floor(k), math.ceil(k)
    if f_idx == c_idx:
        return sorted_values[int(k)]
    d0 = sorted_values[f_idx] * (c_idx - k)
    d1 = sorted_values[c_idx] * (k - f_idx)
    return d0 + d1


async def _band_by_day(db: AsyncSession, f: DashboardFilters) -> list[BandDayBucket]:
    day = _day_expr(_is_sqlite(db)).label("day")
    stmt = (
        select(day, SessionRow.band, func.count().label("n"))
        .where(*_core_conditions(f), SessionRow.band.is_not(None))
        .group_by(day, SessionRow.band)
        .order_by(day)
    )
    rows = (await db.execute(stmt)).all()
    return _bucket_rows(rows)


async def _officer_decision_by_day(db: AsyncSession, f: DashboardFilters) -> list[BandDayBucket]:
    day = _day_expr(_is_sqlite(db)).label("day")
    decision = SessionRow.payload["officer_decision"]["decision"].as_string().label("decision")
    stmt = (
        select(day, decision, func.count().label("n"))
        .where(*_core_conditions(f), decision.is_not(None))
        .group_by(day, decision)
        .order_by(day)
    )
    rows = (await db.execute(stmt)).all()
    return _bucket_rows(rows)


def _bucket_rows(rows) -> list[BandDayBucket]:
    by_day: dict[str, dict[str, int]] = {}
    for day, key, n in rows:
        by_day.setdefault(day, {})[key] = n
    return [BandDayBucket(date=d, counts=counts) for d, counts in sorted(by_day.items())]


async def _overrides_by_day(db: AsyncSession, f: DashboardFilters) -> tuple[list[OverrideDayBucket], float]:
    day = _day_expr(_is_sqlite(db)).label("day")
    override = SessionRow.payload["officer_decision"]["override"].as_boolean()
    # CASE, not CAST(bool AS int): a direct boolean->integer cast is a
    # Postgres extension, not standard SQL, and SQLite's take on it
    # (bool doesn't really exist as a type there) is a different-enough
    # story that CASE is the one form both dialects agree on.
    overrides_sum = func.sum(case((override.is_(True), 1), else_=0)).label("overrides")
    stmt = (
        select(day, func.count().label("total"), overrides_sum)
        .where(*_core_conditions(f))
        .group_by(day)
        .order_by(day)
    )
    rows = (await db.execute(stmt)).all()
    buckets = []
    total_all = 0
    overrides_all = 0
    for day_val, total, overrides in rows:
        overrides = overrides or 0
        total_all += total
        overrides_all += overrides
        rate = round(100.0 * overrides / total, 2) if total else 0.0
        buckets.append(OverrideDayBucket(date=day_val, total=total, overrides=overrides, override_rate_pct=rate))
    overall_rate = round(100.0 * overrides_all / total_all, 2) if total_all else 0.0
    return buckets, overall_rate


async def _sessions_by_day(db: AsyncSession, f: DashboardFilters) -> list[DailyCount]:
    day = _day_expr(_is_sqlite(db)).label("day")
    stmt = select(day, func.count().label("n")).where(*_core_conditions(f)).group_by(day).order_by(day)
    rows = (await db.execute(stmt)).all()
    return [DailyCount(date=d, count=n) for d, n in rows]


async def _latency_percentiles(db: AsyncSession, f: DashboardFilters) -> LatencyPercentiles:
    total_ms = SessionRow.payload["timing_ms"]["total"].as_float()
    if not _is_sqlite(db):
        p50 = func.percentile_cont(0.5).within_group(total_ms.asc())
        p95 = func.percentile_cont(0.95).within_group(total_ms.asc())
        stmt = select(p50, p95, func.count(total_ms)).where(*_core_conditions(f), total_ms.is_not(None))
        row = (await db.execute(stmt)).one()
        p50_val, p95_val, n = row
        return LatencyPercentiles(p50_ms=p50_val, p95_ms=p95_val, sample_size=n)

    stmt = (
        select(total_ms)
        .where(*_core_conditions(f), total_ms.is_not(None))
        .order_by(total_ms.asc())
    )
    values = [v for (v,) in (await db.execute(stmt)).all() if v is not None]
    return LatencyPercentiles(
        p50_ms=_percentile(values, 0.5), p95_ms=_percentile(values, 0.95), sample_size=len(values),
    )


def _raw_where(f: DashboardFilters) -> tuple[str, dict]:
    """WHERE clause + bound params shared by the raw text() queries below
    (array-unnesting has no portable SQLAlchemy Core expression, unlike
    everything above)."""
    from_ts, to_ts = _date_window(f)
    # CAST(:param AS TEXT), not a bare :param IS NULL: asyncpg's prepared
    # statements need a concrete type for every parameter, and a parameter
    # that is ONLY ever compared to NULL (the officer_id=None / lane_id=None
    # "no filter" case) gives it nothing to infer a type from --
    # AmbiguousParameterError. The cast is a no-op for the non-NULL case.
    return (
        "sealed = :sealed AND started_at >= :from_ts AND started_at < :to_ts "
        "AND (CAST(:officer_id AS TEXT) IS NULL OR officer_id = CAST(:officer_id AS TEXT)) "
        "AND (CAST(:lane_id AS TEXT) IS NULL OR lane_id = CAST(:lane_id AS TEXT))",
        {
            "sealed": True,
            "from_ts": from_ts,
            "to_ts": to_ts,
            "officer_id": f.officer_id,
            "lane_id": f.lane_id,
        },
    )


async def _signal_facets(db: AsyncSession, f: DashboardFilters) -> tuple[list[SignalCodeFrequency], int]:
    """Returns (top signal codes, sessions_with_convergence_group)."""
    where_sql, params = _raw_where(f)

    if _is_sqlite(db):
        all_signals_cte = f"""
        WITH filtered AS (
            SELECT session_id, payload FROM sessions WHERE {where_sql}
        ),
        all_signals AS (
            SELECT json_extract(je.value, '$.code') AS code,
                   json_extract(je.value, '$.convergence_group') AS convergence_group,
                   filtered.session_id AS session_id
            FROM filtered, json_each(filtered.payload, '$.signals') AS je
            UNION ALL
            SELECT json_extract(je.value, '$.code'), json_extract(je.value, '$.convergence_group'),
                   filtered.session_id
            FROM filtered, json_each(filtered.payload, '$.cross_document_signals') AS je
        )
        """
    else:
        all_signals_cte = f"""
        WITH filtered AS (
            SELECT session_id, payload FROM sessions WHERE {where_sql}
        ),
        all_signals AS (
            SELECT elem->>'code' AS code, elem->>'convergence_group' AS convergence_group,
                   filtered.session_id AS session_id
            FROM filtered, LATERAL json_array_elements(COALESCE(filtered.payload->'signals', '[]'::json)) AS elem
            UNION ALL
            SELECT elem->>'code', elem->>'convergence_group', filtered.session_id
            FROM filtered,
                 LATERAL json_array_elements(COALESCE(filtered.payload->'cross_document_signals', '[]'::json)) AS elem
        )
        """

    top_codes_sql = all_signals_cte + (
        "SELECT code, count(*) AS n FROM all_signals GROUP BY code ORDER BY n DESC, code LIMIT :top_n"
    )
    top_rows = (await db.execute(text(top_codes_sql), {**params, "top_n": TOP_SIGNAL_CODES_N})).all()
    top_codes = [SignalCodeFrequency(code=code, count=n) for code, n in top_rows]

    convergence_sql = all_signals_cte + (
        "SELECT count(DISTINCT session_id) FROM all_signals WHERE convergence_group IS NOT NULL"
    )
    convergence_count = (await db.execute(text(convergence_sql), params)).scalar_one()

    return top_codes, convergence_count or 0


async def _coverage_flag_frequency(db: AsyncSession, f: DashboardFilters) -> list[CoverageFlagFrequency]:
    where_sql, params = _raw_where(f)
    if _is_sqlite(db):
        sql = f"""
        SELECT je.value AS flag, count(*) AS n
        FROM sessions, json_each(sessions.payload, '$.coverage_flags') AS je
        WHERE {where_sql}
        GROUP BY je.value ORDER BY n DESC, flag
        """
    else:
        sql = f"""
        SELECT flag, count(*) AS n
        FROM sessions,
             LATERAL json_array_elements_text(COALESCE(sessions.payload->'coverage_flags', '[]'::json)) AS flag
        WHERE {where_sql}
        GROUP BY flag ORDER BY n DESC, flag
        """
    rows = (await db.execute(text(sql), params)).all()
    return [CoverageFlagFrequency(flag=flag, session_count=n) for flag, n in rows]


async def _filter_options(db: AsyncSession, f: DashboardFilters) -> DashboardFilterOptions:
    """Distinct officer_ids/lane_ids for the date range -- deliberately
    NOT filtered by f.officer_id/f.lane_id themselves (see
    DashboardFilterOptions's docstring): a dropdown shouldn't shrink to
    just the currently-selected value."""
    from_ts, to_ts = _date_window(f)
    base = [SessionRow.sealed.is_(True), SessionRow.started_at >= from_ts, SessionRow.started_at < to_ts]
    officer_ids = (
        await db.execute(select(SessionRow.officer_id).where(*base).distinct().order_by(SessionRow.officer_id))
    ).scalars().all()
    lane_ids = (
        await db.execute(select(SessionRow.lane_id).where(*base).distinct().order_by(SessionRow.lane_id))
    ).scalars().all()
    return DashboardFilterOptions(officer_ids=list(officer_ids), lane_ids=list(lane_ids))


async def get_dashboard_summary(
    db: AsyncSession, *, scope: str, filters: DashboardFilters,
) -> DashboardSummary:
    total_sessions = (
        await db.execute(select(func.count()).select_from(SessionRow).where(*_core_conditions(filters)))
    ).scalar_one()

    system_band_by_day = await _band_by_day(db, filters)
    officer_decision_by_day = await _officer_decision_by_day(db, filters)
    overrides_by_day, override_rate_pct = await _overrides_by_day(db, filters)

    sessions_by_day = await _sessions_by_day(db, filters)
    latency = await _latency_percentiles(db, filters)
    coverage = await _coverage_flag_frequency(db, filters)

    top_codes, convergence_count = await _signal_facets(db, filters)
    convergence_pct = round(100.0 * convergence_count / total_sessions, 2) if total_sessions else 0.0

    filter_options = await _filter_options(db, filters) if scope == "all" else None

    return DashboardSummary(
        scope=scope,  # type: ignore[arg-type]
        from_date=filters.from_date.isoformat(),
        to_date=filters.to_date.isoformat(),
        total_sessions=total_sessions,
        decision_patterns=DecisionPatterns(
            system_band_by_day=system_band_by_day,
            officer_decision_by_day=officer_decision_by_day,
            override_rate_pct=override_rate_pct,
            overrides_by_day=overrides_by_day,
        ),
        operational=OperationalMetrics(
            sessions_by_day=sessions_by_day, latency=latency, coverage_flag_frequency=coverage,
        ),
        fraud_signals=FraudSignals(
            top_signal_codes=top_codes,
            sessions_with_convergence_group=convergence_count,
            sessions_with_convergence_group_pct=convergence_pct,
        ),
        filter_options=filter_options,
    )
