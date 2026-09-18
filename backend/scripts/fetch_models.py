#!/usr/bin/env python
"""BACKEND_BRIEF.md §1.5 model registry: fetch once, verify SHA-256, write
models/manifest.json. Nothing downloads at *runtime* -- this script is the
one place model files are allowed to be fetched from the network, run by
hand/CI, offline from then on. app/main.py verifies every hash at boot and
refuses to start on a mismatch (a silently swapped model in a border system
is a security incident).

Two kinds of registry entry, both go through the same verify-and-record path:

- **real weights**, already present on disk (the RapidOCR det/rec/cls ONNX
  files -- see the MODEL DOWNLOAD NOTE in app/ocr/viz.py for how those got
  here by hand). This script verifies, never re-downloads them.
- **not yet built** (mrz_crnn -- CRNN not trained yet, BACKEND_BRIEF.md §1.3;
  scrfd/arcface/tamper_unet/pad -- face and forensics milestones, M4). These
  get an explicit, clearly-labelled placeholder file so the registry is one
  consistent system end to end rather than two (a "real" one for OCR and an
  ad hoc gap for everything else) -- the whole point of this M1 task item.
  A placeholder's manifest entry carries `"placeholder": true` so main.py and
  any caller can tell a stand-in from a real, load-bearing model file.

Run: python scripts/fetch_models.py
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

_PLACEHOLDER_PREAMBLE = (
    "PLACEHOLDER MODEL FILE -- NOT A TRAINED WEIGHT.\n"
    "BACKEND_BRIEF.md M1 (skeleton and contracts): {reason}\n"
    "This file exists only so the model registry (§1.5) has one hash-verified\n"
    "entry per model, covering not-yet-built models the same way as real ones,\n"
    "rather than a special case. Loading code must check manifest[key]['placeholder']\n"
    "and refuse to run inference against it.\n"
)


@dataclass
class RegistryEntry:
    key: str
    version: str
    rel_path: str
    placeholder: bool
    reason: Optional[str] = None
    source: Optional[str] = None
    note: Optional[str] = None


# BACKEND_BRIEF.md §1.5's table, field for field.
REGISTRY: list[RegistryEntry] = [
    RegistryEntry(
        key="mrz_crnn", version="1.3.0", rel_path="models/mrz_crnn/1.3.0/model.onnx",
        placeholder=True,
        reason="the CRNN (app/ocr/mrz/model.py) has no trained checkpoint yet -- "
               "training needs a GPU this environment doesn't have. MRZReader runs "
               "in stub mode (require_trained_weights=False) until this is trained.",
    ),
    RegistryEntry(
        key="rapidocr_det", version="v5", rel_path="models/rapidocr_det/v5/det.onnx",
        placeholder=False,
        source="https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
        note="Fetched once by hand, offline from here on. See app/ocr/viz.py MODEL DOWNLOAD NOTE.",
    ),
    RegistryEntry(
        key="rapidocr_rec", version="v5", rel_path="models/rapidocr_rec/v5/rec.onnx",
        placeholder=False,
        source="https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile.onnx",
        note="Character dictionary is embedded in the ONNX metadata; no separate dict.txt needed.",
    ),
    RegistryEntry(
        key="rapidocr_cls", version="v4", rel_path="models/rapidocr_cls/v4/cls.onnx",
        placeholder=False,
        source="https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        note="RapidOCR always constructs its text-orientation classifier at init "
             "regardless of use_cls, so this must be present even though VizReader disables its use.",
    ),
    RegistryEntry(
        key="scrfd", version="10g", rel_path="models/scrfd/10g/scrfd.onnx",
        placeholder=True, reason="face branch (app/pipeline/branches/face.py) is a stub until M4.",
    ),
    RegistryEntry(
        key="arcface", version="buffalo_l", rel_path="models/arcface/buffalo_l/w600k_r50.onnx",
        placeholder=True, reason="face branch (app/pipeline/branches/face.py) is a stub until M4.",
    ),
    RegistryEntry(
        key="tamper_unet", version="0.9.2", rel_path="models/tamper_unet/0.9.2/model.onnx",
        placeholder=True, reason="forensics branch (app/pipeline/branches/forensics.py) is a stub until M4.",
    ),
    RegistryEntry(
        key="pad_minifas", version="1.0", rel_path="models/pad/minifas/1.0/model.onnx",
        placeholder=True, reason="presentation-attack detection is part of the face branch, stub until M4.",
    ),
]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_placeholder(entry: RegistryEntry) -> Path:
    path = PROJECT_ROOT / entry.rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _PLACEHOLDER_PREAMBLE.format(reason=entry.reason)
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")
    return path


def fetch_all() -> dict:
    manifest: dict = {}
    for entry in REGISTRY:
        path = PROJECT_ROOT / entry.rel_path
        if entry.placeholder:
            path = _ensure_placeholder(entry)
        elif not path.exists():
            raise FileNotFoundError(
                f"{entry.key}: expected real weights at {path} -- fetch them by hand from "
                f"{entry.source} first (this script does not download at runtime; see §1.5)."
            )

        digest = _sha256_file(path)
        manifest[entry.key] = {
            "version": entry.version,
            "path": entry.rel_path.replace("\\", "/"),
            "sha256": digest,
            "placeholder": entry.placeholder,
        }
        if entry.source:
            manifest[entry.key]["source"] = entry.source
        if entry.note:
            manifest[entry.key]["note"] = entry.note
        if entry.placeholder:
            manifest[entry.key]["note"] = entry.reason

    return manifest


def main() -> None:
    manifest = fetch_all()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for key, entry in manifest.items():
        kind = "PLACEHOLDER" if entry["placeholder"] else "real"
        print(f"  {key:14s} [{kind:11s}] {entry['path']}  sha256={entry['sha256'][:12]}...")
    print(f"wrote {MANIFEST_PATH.relative_to(PROJECT_ROOT)} ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
