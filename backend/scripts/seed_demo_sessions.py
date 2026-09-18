#!/usr/bin/env python
"""DEV/DEMO-ONLY synthetic data. Generates a realistic spread of a few
hundred SEALED screening sessions across the last 30 days -- enough volume
and variety (band, officer, lane, override, signal codes, coverage flags,
timing) that GET /api/v1/dashboard/summary's charts have real shape instead
of the 3 flat data points a handful of hand-written fixtures would give.

**These are not real screenings.** No document images, no real identity
data, no pipeline run -- each row is a hand-built ScreeningSession inserted
straight through app/storage/repositories.py's upsert_session, the same
function the real orchestrator uses to persist a completed session. Never
run this against a production database.

Run: python scripts/seed_demo_sessions.py [--count 400] [--seed 42]
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import random
import uuid

from app.contracts import OfficerDecision, ScreeningSession, Signal
from app.storage.db import async_session_factory
from app.storage.repositories import upsert_session

OFFICER_IDS = ["OFF-2291", "OFF-3310", "OFF-0001"]  # the non-admin dev officers (seed_officers.py)
LANE_IDS = ["IGI-T3-LANE-01", "IGI-T3-LANE-02", "IGI-T3-LANE-03", "IGI-T3-LANE-04", "IGI-T3-LANE-05"]

# code, module, severity, weight -- a representative slice of
# BACKEND_BRIEF.md's signal catalogue (§6.2-§6.7), not the exhaustive set.
SIGNAL_POOL = [
    ("MRZ_CHECKDIGIT_DOB", "validation", "high", 30.0),
    ("MRZ_VIZ_MISMATCH", "ocr", "high", 25.0),
    ("CHECKSUM_UNRECOVERABLE", "ocr", "high", 28.0),
    ("FACE_MISMATCH", "face", "high", 22.0),
    ("TAMPER_ELA_ANOMALY", "tamper", "medium", 18.0),
    ("COPY_MOVE_DETECTED", "tamper", "critical", 40.0),
    ("FONT_MISMATCH", "tamper", "medium", 15.0),
    ("PORTRAIT_SEAM_ANOMALY", "tamper", "high", 24.0),
    ("GUILLOCHE_ANOMALY", "tamper", "medium", 16.0),
    ("PRINT_ORIGIN_MISMATCH", "tamper", "low", 10.0),
    ("TEMPLATE_ANOMALY", "template", "medium", 14.0),
    ("WATCHLIST_HIT", "database", "critical", 100.0),
    ("IDENTITY_CONFLICT", "graph", "high", 35.0),
    ("VISA_PASSPORT_MISMATCH", "crossdoc", "critical", 45.0),
    ("DOC_EXPIRED", "validation", "medium", 20.0),
]

COVERAGE_FLAG_POOL = ["no_biometric", "stale_watchlist", "no_ovd"]

# (system_band, weight) -- CLEAR dominates a real lane's traffic; RECAPTURE
# is excluded because it never reaches a decision (app/api/screening.py's
# decision endpoint has nothing to seal for a quality-gate failure).
BAND_WEIGHTS = [("CLEAR", 68), ("SECONDARY", 16), ("HOLD", 11), ("ABSTAIN", 5)]

DECISION_CHOICES = ["CLEAR", "SECONDARY", "HOLD", "REFER"]


def _signals_for_band(rng: random.Random, band: str) -> tuple[list[Signal], list[Signal]]:
    """Returns (signals, cross_document_signals). More/heavier signals for
    riskier bands; CLEAR usually has none, occasionally one info-only."""
    if band == "CLEAR":
        if rng.random() < 0.1:
            code, module, _sev, _w = rng.choice(SIGNAL_POOL)
            return [Signal(id=str(uuid.uuid4()), code=code, module=module, severity="info", weight=0.0,
                            detail="Reviewed, no material finding.")], []
        return [], []

    n = {"SECONDARY": rng.randint(1, 2), "HOLD": rng.randint(2, 4), "ABSTAIN": rng.randint(0, 2)}[band]
    chosen = rng.sample(SIGNAL_POOL, k=min(n, len(SIGNAL_POOL)))
    convergence_group = str(uuid.uuid4()) if (band == "HOLD" and rng.random() < 0.25 and len(chosen) >= 3) else None
    signals = [
        Signal(
            id=str(uuid.uuid4()), code=code, module=module, severity=sev, weight=weight,
            detail=f"{code.replace('_', ' ').title()} detected.",
            convergence_group=convergence_group if i < 3 else None,
        )
        for i, (code, module, sev, weight) in enumerate(chosen)
    ]
    cross_doc = []
    if band in ("HOLD", "SECONDARY") and rng.random() < 0.08:
        cross_doc = [Signal(id=str(uuid.uuid4()), code="VISA_PASSPORT_MISMATCH", module="crossdoc",
                             severity="critical", weight=45.0, detail="Visa references a different passport number.")]
    return signals, cross_doc


def _risk_for_band(rng: random.Random, band: str) -> tuple[float | None, float]:
    if band == "ABSTAIN":
        return None, round(rng.uniform(0.3, 0.64), 2)
    ranges = {"CLEAR": (0, 24), "SECONDARY": (25, 59), "HOLD": (60, 100)}
    lo, hi = ranges[band]
    return float(rng.randint(lo, hi)), round(rng.uniform(0.7, 0.99), 2)


def _decision_for(rng: random.Random, band: str) -> str:
    """Mirrors the real access pattern: an officer usually agrees with a
    concrete system band, occasionally overrides; ABSTAIN forces a choice
    (Decision has no ABSTAIN value), which is always an override by
    app/api/screening.py's own `override = decision != session.band` rule
    -- computed the same way below, not hand-picked, to stay consistent
    with the real endpoint's semantics."""
    if band in ("CLEAR", "SECONDARY", "HOLD") and rng.random() < 0.85:
        return band
    return rng.choice(DECISION_CHOICES)


def _build_session(rng: random.Random, day_offset: int) -> ScreeningSession:
    band = rng.choices([b for b, _ in BAND_WEIGHTS], weights=[w for _, w in BAND_WEIGHTS])[0]
    decision = _decision_for(rng, band)
    risk, confidence = _risk_for_band(rng, band)
    signals, cross_doc = _signals_for_band(rng, band)

    coverage_flags = sorted({f for f in COVERAGE_FLAG_POOL if rng.random() < 0.08})

    base_ms = {"CLEAR": 900, "SECONDARY": 1300, "HOLD": 1800, "ABSTAIN": 1500}[band]
    total_ms = round(base_ms + rng.uniform(-200, 700), 1)

    started_at = (
        dt.datetime.now(dt.timezone.utc)
        - dt.timedelta(days=day_offset, hours=rng.uniform(0, 23), minutes=rng.uniform(0, 59))
    ).isoformat()

    return ScreeningSession(
        session_id=str(uuid.uuid4()),
        lane_id=rng.choice(LANE_IDS),
        officer_id=rng.choice(OFFICER_IDS),
        started_at=started_at,
        band=band,
        risk=risk,
        confidence=confidence,
        abstained=(band == "ABSTAIN"),
        signals=signals,
        cross_document_signals=cross_doc,
        coverage_flags=list(coverage_flags),
        timing_ms={"total": total_ms},
        officer_decision=OfficerDecision(
            decision=decision,
            note="synthetic demo data" if decision != band else "",
            decided_at=started_at,
            override=(decision != band),
        ),
        sealed=True,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--range-days", type=int, default=30)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    async with async_session_factory() as db:
        for i in range(args.count):
            day_offset = rng.randint(0, args.range_days - 1)
            session = _build_session(rng, day_offset)
            await upsert_session(db, session)
            if (i + 1) % 50 == 0:
                print(f"seeded {i + 1}/{args.count}")
    print(f"done: {args.count} synthetic sealed sessions across the last {args.range_days} days")


if __name__ == "__main__":
    asyncio.run(main())
