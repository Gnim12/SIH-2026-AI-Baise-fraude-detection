"""SQLAlchemy 2.0 async engine + session factory (BACKEND_BRIEF.md §2)."""
from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(settings.database_url, pool_pre_ping=True)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)

if engine.dialect.name == "sqlite":
    # sqlite ignores FOREIGN KEY constraints unless told otherwise -- unlike
    # Postgres (the real dev/prod DB), which enforces them. Without this, a
    # broken insert order (e.g. a join row referencing an unflushed parent
    # row) passes silently in the sqlite-backed test suite and only breaks
    # against the real database.
    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


async def get_session() -> AsyncIterator[AsyncSession]:
    async with async_session_factory() as session:
        yield session
