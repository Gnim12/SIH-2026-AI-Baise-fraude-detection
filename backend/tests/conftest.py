"""Session-wide test config, loaded before any other test module.

Forces app.config.settings to point at a local sqlite file rather than the
production Postgres URL, so the test suite doesn't require docker-compose's
Postgres service to be running. Must be set before app.config (and anything
that imports it) is imported anywhere in the session -- conftest.py at the
tests/ root is collected first, which is what makes this reliable.
"""
import os

import pytest
from pathlib import Path

_TEST_DB_PATH = Path(__file__).resolve().parent / "_test_screening.db"
os.environ.setdefault("SCREENING_DATABASE_URL", f"sqlite+aiosqlite:///{_TEST_DB_PATH}")


@pytest.fixture(autouse=True)
def _stub_mrz_runtime(request):
    """The legacy pipeline tests decode against ground truth with a stub-mode
    MRZ reader. That is only reachable through runtime.install_reader_for_tests
    (never from app/), so it is installed here by default. Tests that exercise
    the real loading path opt out with @pytest.mark.real_mrz."""
    from app.ocr.mrz import runtime
    from app.ocr.mrz.infer import MRZReader

    runtime.reset_mrz_runtime()
    if request.node.get_closest_marker("real_mrz") is None:
        runtime.install_reader_for_tests(MRZReader(require_trained_weights=False))
    yield
    runtime.reset_mrz_runtime()
