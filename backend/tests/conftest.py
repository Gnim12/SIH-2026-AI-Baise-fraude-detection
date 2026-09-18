"""Session-wide test config, loaded before any other test module.

Forces app.config.settings to point at a local sqlite file rather than the
production Postgres URL, so the test suite doesn't require docker-compose's
Postgres service to be running. Must be set before app.config (and anything
that imports it) is imported anywhere in the session -- conftest.py at the
tests/ root is collected first, which is what makes this reliable.
"""
import os
from pathlib import Path

_TEST_DB_PATH = Path(__file__).resolve().parent / "_test_screening.db"
os.environ.setdefault("SCREENING_DATABASE_URL", f"sqlite+aiosqlite:///{_TEST_DB_PATH}")
