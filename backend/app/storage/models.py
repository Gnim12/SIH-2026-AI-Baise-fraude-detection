"""BACKEND_BRIEF.md §8.1 defines the full relational schema (sessions,
documents, signals, decisions, audit_records, gallery, watchlist,
graph_edges) as M5 ("persistence") work -- audit chain, gallery enrolment,
identity graph, watchlist, retention.

This M1 milestone only needs `GET /api/v1/history` and
`GET /api/v1/screening/{id}` "backed by real Postgres rows" (this task's own
scope). One `sessions` table, sealed session stored as JSONB with a handful
of indexed columns for the `history` filters, is the honest M1-sized slice of
that schema -- normalizing documents/signals/decisions into their own tables
without the audit chain, gallery, or watchlist they exist to serve is placed
ahead of them for no real reason. Splitting it out is real M5 work, not
deferred by omission.
"""
from __future__ import annotations

from sqlalchemy import JSON, Boolean, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class SessionRow(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    lane_id: Mapped[str] = mapped_column(String, index=True)
    officer_id: Mapped[str] = mapped_column(String, index=True)
    # Indexed: app/storage/dashboard_repository.py range-filters on this for
    # every dashboard query. This table (all four of its indexes, not just
    # this one) previously existed only via create_all() at app startup --
    # no Alembic migration tracked it, unlike officers/officer_sessions
    # (alembic/versions/a1b2c3d4e5f6). alembic/versions/9d17cd0a007b closes
    # that gap: a fresh checkout gets this table + index from
    # `alembic upgrade head` alone now, verified against a genuinely fresh
    # DB (upgrade, downgrade, and re-upgrade all confirmed clean).
    started_at: Mapped[str] = mapped_column(String, index=True)
    band: Mapped[str | None] = mapped_column(String, nullable=True)
    sealed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict] = mapped_column(JSON)  # the full ScreeningSession, snake_case
