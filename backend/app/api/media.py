"""Local-disk document image storage for M1.

BACKEND_BRIEF.md §2/§3 provisions MinIO (S3 API, encrypted at rest) for
originals; wiring `storage/objects.py` up to it is real work this task's
scope (contracts/orchestrator/API/model-registry) doesn't cover -- the
docker-compose service is provisioned and ready, but the app itself writes
uploaded originals to local disk and serves them back under `/media/...` for
now. Swapping this module's two functions for MinIO put/presigned-URL calls
is the entire migration; callers only ever see `image_url` strings.
"""
from __future__ import annotations

from pathlib import Path

from app.config import PROJECT_ROOT

MEDIA_DIR = PROJECT_ROOT / "data" / "uploads"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def save_document_image(document_id: str, jpeg_bytes: bytes) -> str:
    path = MEDIA_DIR / f"{document_id}.jpg"
    path.write_bytes(jpeg_bytes)
    return f"/media/{document_id}.jpg"
