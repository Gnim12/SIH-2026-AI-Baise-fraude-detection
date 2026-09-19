"""BACKEND_BRIEF.md §1.5: model registry hash verification, shared by
app/main.py's startup check and any model loader (see app/ocr/viz.py's
verify_and_resolve, which this generalises the same pattern from -- one
manifest, one verification routine, real models and not-yet-built
placeholders alike, per the same discipline).

On startup the app verifies every hash and refuses to boot on mismatch: a
silently swapped model in a border system is a security incident (§1.5).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import settings


class ModelRegistryError(RuntimeError):
    """Raised when a manifest entry is missing, unreadable, or its file's
    hash doesn't match -- the app must refuse to boot when this is raised."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    path = manifest_path or settings.manifest_path
    if not path.exists():
        raise ModelRegistryError(
            f"model manifest not found at {path}. Run scripts/fetch_models.py first."
        )
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return manifest


def verify_all(
    manifest_path: Path | None = None, *, exempt: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Verify every entry's SHA-256 against its file on disk. Returns a
    {model_key: version} map (BACKEND_BRIEF.md §8.2: this is what
    `model_versions` in every audit record and ScreeningSession comes from).
    Raises ModelRegistryError on the first missing file or hash mismatch --
    never silently skips an entry.

    `exempt` names keys verified elsewhere with a softer failure mode (the MRZ
    model: app/ocr/mrz/runtime.py reports it unavailable rather than
    refusing to boot). They are omitted from the returned map."""
    manifest = load_manifest(manifest_path)
    project_root = settings.models_dir.parent
    versions: dict[str, str] = {}

    for key, entry in manifest.items():
        if key in exempt:
            continue
        rel_path = entry["path"]
        expected_hash = entry["sha256"]
        file_path = project_root / rel_path

        if not file_path.exists():
            raise ModelRegistryError(
                f"model registry: {key!r} is listed in the manifest but the file at "
                f"{file_path} does not exist. Refusing to boot."
            )
        actual_hash = _sha256(file_path)
        if actual_hash != expected_hash:
            raise ModelRegistryError(
                f"model registry: {key!r} hash mismatch. Manifest expects "
                f"{expected_hash}, file at {file_path} hashes to {actual_hash}. "
                "A silently swapped model in a border system is a security "
                "incident -- refusing to boot."
            )
        versions[key] = entry.get("version", "unknown")

    return versions
