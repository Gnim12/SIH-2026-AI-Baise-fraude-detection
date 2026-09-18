"""FastAPI app, lifespan, model loading + hash check (BACKEND_BRIEF.md §3).

Startup verifies every model hash and refuses to boot on mismatch (§1.5): a
silently swapped model in a border system is a security incident.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import b1_analyse, b1_health, b1_sessions, dashboard, health, history, screening, stream
from .api import b1_errors
from .api import b2_cases, b2_decisions
from .api.media import MEDIA_DIR
from .auth import models as auth_models  # noqa: F401 -- registers Officer/OfficerSession on Base.metadata
from .auth.routes import admin_router, router as auth_router
from .config import settings
from .pipeline.branches.ocr import warm_up as warm_up_ocr
from .registry import verify_all
from .storage import b1_models  # noqa: F401 -- registers the B1b tables on Base.metadata
from .storage import b2_models  # noqa: F401 -- registers the B2 tables on Base.metadata
from .storage.db import Base, engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _check_artefact_root_writable() -> None:
    """B1b startup check: fail loudly here, not on a traveller's first
    upload. Creates the directory if missing, then proves it's writable by
    actually writing and removing a probe file -- existence alone doesn't
    guarantee write permission."""
    root = settings.b1_artefact_root
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".write_check"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"B1b artefact directory {root} is not writable: {exc}. Refusing to boot."
        ) from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    # §1.5: refuse to boot on a hash mismatch or missing model file.
    model_versions = verify_all()
    logger.info("model registry verified: %d entries (%s)", len(model_versions), model_versions)

    # B1b: artefact storage must be writable before the first traveller
    # arrives, not discovered on their upload.
    _check_artefact_root_writable()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Build RapidOCR's/MRZReader's ONNX sessions now, not on whichever
    # request happens to hit branch_ocr() first -- see ocr.py's warm_up()
    # docstring. Off the event loop since this is blocking CPU/IO work.
    await asyncio.to_thread(warm_up_ocr)
    logger.info("OCR branch warm-up complete: RapidOCR + MRZReader sessions loaded")

    yield


app = FastAPI(title="Screening Backend", lifespan=lifespan)

# The officer console (Vite dev server) runs on a different origin, and now
# sends a session cookie -- allow_origins can no longer be "*" once
# allow_credentials=True (browsers refuse the combination), so this is an
# explicit allow-list of the frontend's dev origins (app/config.py).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(screening.router)
app.include_router(stream.router)
app.include_router(history.router)
app.include_router(dashboard.router)
app.include_router(health.router)

# B1b: separate namespace (no /v1 prefix, no officer auth -- see this
# module's lifespan docstring context and app/api/b1_*.py module docstrings).
b1_errors.register(app)
app.include_router(b1_sessions.router)
app.include_router(b1_analyse.router)
app.include_router(b1_health.router)

# B2: decision submission, case register, config audit -- same B1b
# namespace/auth decisions, built on top of it.
app.include_router(b2_decisions.router)
app.include_router(b2_cases.router)
