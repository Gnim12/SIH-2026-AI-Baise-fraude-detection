"""Process-wide MRZ model state: the one place that decides whether the MRZ
recogniser is live or unavailable.

`load_mrz_runtime()` never raises. A missing, placeholder, hash-mismatched or
otherwise unloadable model produces a `MrzModelStatus` with state
"unavailable" and a reason, and no reader. The app boots either way: a
terminal with a missing model must start and say so, not refuse to start and
leave an officer with a dead machine. What must never happen is a fallback
to stub decoding -- there is no such fallback here.

The only way to get a stub-mode reader into the pipeline is
`install_reader_for_tests`, which nothing under app/ calls.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from app.config import settings

from .infer import MRZ_PIN_NAME, MissingWeightsError, ModelIntegrityError, MRZReader

logger = logging.getLogger(__name__)

MrzState = Literal["live", "unavailable"]


@dataclass(frozen=True)
class MrzModelStatus:
    state: MrzState
    reason: Optional[str] = None
    version: Optional[str] = None

    @property
    def pin(self) -> Optional[str]:
        return f"{MRZ_PIN_NAME} {self.version}" if self.state == "live" and self.version else None


@dataclass(frozen=True)
class MrzRuntime:
    reader: Optional[MRZReader]
    status: MrzModelStatus


def _unavailable(reason: str) -> MrzRuntime:
    logger.error("MRZ recogniser UNAVAILABLE: %s", reason)
    return MrzRuntime(reader=None, status=MrzModelStatus(state="unavailable", reason=reason))


def load_mrz_runtime(manifest_path: Optional[Path] = None) -> MrzRuntime:
    path = manifest_path or settings.manifest_path
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _unavailable(f"model manifest unreadable at {path}: {exc}")

    entry = manifest.get("mrz_crnn")
    if not isinstance(entry, dict):
        return _unavailable(f"no 'mrz_crnn' entry in {path}")
    if entry.get("placeholder", True):
        return _unavailable("the MRZ model is still a placeholder: no trained weights have been installed.")

    # Same root as app/registry.py: manifest paths are relative to the
    # directory above the models dir.
    weights_path = path.parent.parent / str(entry.get("path", ""))
    try:
        reader = MRZReader(weights_path, require_trained_weights=True)
    except (MissingWeightsError, ModelIntegrityError) as exc:
        return _unavailable(str(exc))
    except Exception as exc:  # noqa: BLE001 -- corrupt ONNX / onnxruntime errors: still unavailable, never a boot failure.
        return _unavailable(f"MRZ model failed to load: {type(exc).__name__}: {exc}")

    status = MrzModelStatus(state="live", version=reader.version)
    logger.info("MRZ recogniser live: %s", status.pin)
    return MrzRuntime(reader=reader, status=status)


_runtime: Optional[MrzRuntime] = None


def get_mrz_runtime() -> MrzRuntime:
    global _runtime
    if _runtime is None:
        _runtime = load_mrz_runtime()
    return _runtime


def reset_mrz_runtime() -> None:
    """Drop the cached runtime so the next get_mrz_runtime() reloads from the
    current settings.manifest_path."""
    global _runtime
    _runtime = None


def install_reader_for_tests(reader: MRZReader) -> None:
    """TEST-ONLY. Installs `reader` (typically stub mode) as the process-wide
    reader. Never called from app/; a test asserts that."""
    global _runtime
    _runtime = MrzRuntime(reader=reader, status=MrzModelStatus(state="live", version=reader.version))
