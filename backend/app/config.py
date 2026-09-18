"""BACKEND_BRIEF.md §2: pydantic-settings, env-driven."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCREENING_", env_file=".env", extra="ignore")

    # Host port 5433: the docker-compose Postgres container is remapped off
    # the default 5432 because this dev box also runs a native PostgreSQL
    # service bound to :5432 (see docker-compose.yml).
    database_url: str = "postgresql+asyncpg://screening:screening@localhost:5433/screening"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "screening"
    minio_secret_key: str = "screening-secret"
    minio_bucket: str = "screening-documents"
    minio_secure: bool = False

    models_dir: Path = PROJECT_ROOT / "models"
    manifest_path: Path = PROJECT_ROOT / "models" / "manifest.json"

    # BACKEND_BRIEF.md §1.4: a VLM may only ever be an optional secondary
    # cross-check, disabled by default; the system must run correctly with
    # it off.
    enable_vlm_crosscheck: bool = False

    # BACKEND_BRIEF.md §1.3: no trained CRNN checkpoint exists yet. Must stay
    # False in production; flipping it silently to True would make MRZReader
    # fabricate logprobs instead of refusing to run without weights.
    mrz_require_trained_weights: bool = True

    ws_replay_buffer_size: int = 256

    # app/auth/: session cookie auth (see app/auth/routes.py docstring for
    # the JWT-vs-server-session tradeoff this locks in).
    session_cookie_name: str = "screening_session"
    session_ttl_hours: int = 12
    # Secure cookies require https. The dev servers (Vite on 5173/5174,
    # uvicorn on 8000) run on plain http://localhost, so a Secure cookie
    # would silently never be set by the browser and every login would look
    # broken. This flag is the one deliberate, explicit relaxation for local
    # dev -- it must default False (Secure) and only ever be flipped by an
    # explicit env var on a dev box, never in a deployed environment where
    # the origin is real https.
    session_cookie_secure: bool = True
    # CORS allow-list for the officer console's dev origins. Never "*" once
    # allow_credentials=True -- the two are mutually exclusive for browsers
    # to honour a credentialed cross-origin cookie.
    frontend_origins: list[str] = ["http://localhost:5173", "http://localhost:5174"]

    # POST /api/v1/auth/reset-requests (app/auth/rate_limit.py): the one
    # unauthenticated auth endpoint, throttled per officer_id so it can't be
    # used to spam an officer or brute-force-probe officer_ids for validity.
    reset_request_rate_limit: int = 5
    reset_request_rate_window_hours: int = 1

    # BACKEND_BRIEF.md §5.3's BUDGET_MS values are production targets on
    # tuned hardware. Configurable rather than hardcoded so a slower dev/demo
    # box (cold ONNX Runtime session load + CPU inference genuinely exceeds
    # 700ms here) doesn't have to permanently report a false OCR_UNAVAILABLE
    # -- the timeout/Degraded mechanism itself is real either way (§5.3).
    ocr_budget_ms: int = 700

    # --- Milestone B1b: HTTP/WS surface for the new stage-registry pipeline
    # (app/pipeline/registry.py, app/api/b1_*.py) -- a separate namespace
    # from the BACKEND_BRIEF.md /api/v1 system above, see app/main.py.
    b1_artefact_root: Path = PROJECT_ROOT / "data" / "artefacts"
    b1_max_image_pdf_upload_mb: int = 25
    b1_max_video_upload_mb: int = 60
    b1_stage_timeout_s: float = 30.0
    # No coverage-estimation implementation exists yet (see
    # app/api/b1_sessions.py's artefact-upload handler) -- this threshold is
    # wired and enforced the moment one lands, never invented meanwhile.
    b1_ovd_angular_coverage_min_deg: float = 40.0


settings = Settings()
