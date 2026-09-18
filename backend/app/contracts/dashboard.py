"""GET /api/v1/dashboard/summary's response contract (app/api/dashboard.py).

Unlike the rest of app/contracts/ (see __init__.py's docstring), this file is
NOT part of the FRONTEND_BRIEF.md-mirrored contract set -- the dashboard is a
new, additive backend-only feature with no frontend consumer yet. It lives
here anyway because this is where this codebase puts Pydantic response
contracts, and because keeping it a real typed model (not an ad hoc dict) is
what makes app/storage/dashboard_repository.py's output checkable at all.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

Scope = Literal["me", "all"]


class DailyCount(BaseModel):
    date: str  # 'YYYY-MM-DD'
    count: int


class BandDayBucket(BaseModel):
    date: str  # 'YYYY-MM-DD'
    counts: dict[str, int]  # band/decision value -> count; only keys seen that day are present


class OverrideDayBucket(BaseModel):
    date: str  # 'YYYY-MM-DD', same bucketing as BandDayBucket
    total: int
    overrides: int
    override_rate_pct: float


class DecisionPatterns(BaseModel):
    # Two breakdowns, not one, because SystemBand and Decision are disjoint
    # enums that don't unify into a single "band" axis: RECAPTURE/ABSTAIN
    # can never be an officer_decision.decision (app/api/screening.py's
    # decision endpoint only accepts CLEAR/SECONDARY/HOLD/REFER), and REFER
    # can never be a system band. Collapsing them into one bucket would
    # either drop REFER entirely or silently conflate "the system abstained"
    # with "the officer referred" -- exactly the system-vs-officer
    # conflation BACKEND_BRIEF.md §0 rule 1 exists to keep apart.
    system_band_by_day: list[BandDayBucket]
    officer_decision_by_day: list[BandDayBucket]
    override_rate_pct: float  # overall, whole date range
    overrides_by_day: list[OverrideDayBucket]


class LatencyPercentiles(BaseModel):
    p50_ms: Optional[float]
    p95_ms: Optional[float]
    sample_size: int


class CoverageFlagFrequency(BaseModel):
    flag: str
    session_count: int


class OperationalMetrics(BaseModel):
    sessions_by_day: list[DailyCount]
    latency: LatencyPercentiles
    coverage_flag_frequency: list[CoverageFlagFrequency]


class SignalCodeFrequency(BaseModel):
    code: str
    count: int


class FraudSignals(BaseModel):
    top_signal_codes: list[SignalCodeFrequency]  # top N (default 15)
    sessions_with_convergence_group: int
    sessions_with_convergence_group_pct: float


class DashboardFilterOptions(BaseModel):
    """scope='all' only -- the full set of officer/lane ids present in the
    date range, independent of any officer_id/lane_id filter also applied,
    so a filter dropdown doesn't shrink to just the currently-selected
    value."""

    officer_ids: list[str] = []
    lane_ids: list[str] = []


class DashboardSummary(BaseModel):
    scope: Scope
    from_date: str  # 'YYYY-MM-DD'
    to_date: str  # 'YYYY-MM-DD', inclusive
    total_sessions: int
    decision_patterns: DecisionPatterns
    operational: OperationalMetrics
    fraud_signals: FraudSignals
    filter_options: Optional[DashboardFilterOptions] = None  # scope='all' only


__all__ = [
    "Scope",
    "DailyCount",
    "BandDayBucket",
    "OverrideDayBucket",
    "DecisionPatterns",
    "LatencyPercentiles",
    "CoverageFlagFrequency",
    "OperationalMetrics",
    "SignalCodeFrequency",
    "FraudSignals",
    "DashboardFilterOptions",
    "DashboardSummary",
]
