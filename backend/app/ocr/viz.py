"""VIZ (visual inspection zone) reader: RapidOCR wrapper + field classifier.

BACKEND_BRIEF.md §1.1/§1.2: RapidOCR (PP-OCRv5 on ONNX Runtime) reads the
printed name/date/authority fields; the MRZ strip is read separately by
mrz.infer.MRZReader and excluded here so RapidOCR never wastes time on OCR-B
text it isn't meant to read.

Model loading is offline-only (§1.5): VizReader never lets RapidOCR resolve
or download a model URL. It always passes explicit local `model_path`s and
verifies each file's SHA-256 against models/manifest.json before construction,
refusing to start on a mismatch or a missing file. See MODEL DOWNLOAD NOTE
below for how those three ONNX files got onto disk in the first place.

MODEL DOWNLOAD NOTE
====================
RapidOCR's own default config resolves and downloads model weights from
https://www.modelscope.cn -- not pypi.org -- the first time a model type is
used without an explicit path. That is a runtime download from a host outside
what §1.5 allows ("Nothing may download a model at runtime"). The det/rec/cls
ONNX files this module loads were fetched once, by hand, outside of
VizReader/RapidOCR's own auto-download path, with their SHA-256 recorded in
models/manifest.json (matching the hashes RapidOCR's own default_models.yaml
publishes for the same files, as an integrity cross-check). The real
scripts/fetch_models.py + boot-time manifest verification for *all* models
in the registry is a later milestone (§1.5); this module implements the same
verify-before-load discipline scoped to just these three files, now.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

from app.contracts import Region

from .mrz import detect as mrz_detect

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _PROJECT_ROOT / "models" / "manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    return json.loads(_MANIFEST_PATH.read_text())


def verify_and_resolve(model_key: str) -> Path:
    """Resolve a manifest entry to a local path, refusing to proceed if the
    file is missing or its hash doesn't match -- a silently swapped model in
    a border system is a security incident (§1.5)."""
    manifest = load_manifest()
    if model_key not in manifest:
        raise KeyError(f"{model_key!r} is not in {_MANIFEST_PATH}")
    entry = manifest[model_key]
    path = _PROJECT_ROOT / entry["path"]
    if not path.exists():
        raise FileNotFoundError(
            f"{model_key} weights not found at {path}. See models/manifest.json "
            "and the MODEL DOWNLOAD NOTE in app/ocr/viz.py."
        )
    actual = _sha256(path)
    if actual != entry["sha256"]:
        raise RuntimeError(
            f"{model_key} hash mismatch: manifest expects {entry['sha256']}, "
            f"file at {path} hashes to {actual}. Refusing to load a silently "
            "swapped model."
        )
    return path


@dataclass
class TextBox:
    """One raw RapidOCR detection, boxes/coords normalised to the full
    (uncropped) document image so downstream consumers never need to know
    that the MRZ band was excluded before OCR ran."""

    text: str
    confidence: float
    region: Region  # normalised 0..1 against the full document image


class VizReader:
    """RapidOCR wrapper: rectified document image -> raw text boxes.

    Excludes the MRZ band (reusing mrz.detect's band region) before running
    detection, so RapidOCR spends its whole budget on the VIZ.
    """

    def __init__(
        self,
        *,
        det_model_path: Optional[str | Path] = None,
        rec_model_path: Optional[str | Path] = None,
        cls_model_path: Optional[str | Path] = None,
    ) -> None:
        det_model_path = det_model_path or verify_and_resolve("rapidocr_det")
        rec_model_path = rec_model_path or verify_and_resolve("rapidocr_rec")
        # RapidOCR always constructs its text-orientation classifier at init
        # regardless of use_cls, so a valid local path is required even
        # though we disable its use below (VIZ crops are already upright).
        cls_model_path = cls_model_path or verify_and_resolve("rapidocr_cls")

        from rapidocr import RapidOCR  # deferred: heavy import, keeps module import cheap

        self._engine = RapidOCR(
            params={
                "Det.model_path": str(det_model_path),
                "Rec.model_path": str(rec_model_path),
                "Cls.model_path": str(cls_model_path),
                "Global.use_cls": False,
                "Global.log_level": "error",
            }
        )

    def _exclude_mrz(
        self, image: np.ndarray, mrz_band: Optional[mrz_detect.MrzBand]
    ) -> tuple[np.ndarray, int, int]:
        """Return (viz_image, y_offset, x_offset): the document image with
        the MRZ band region blanked out isn't necessary -- cropping the band
        off the bottom is enough since RapidOCR runs on the remaining region
        and produces no detections inside it. y_offset lets box coordinates
        be reported relative to the *full* document image."""
        if mrz_band is None:
            return image, 0, 0
        band_y = mrz_band.bbox[1]
        if band_y <= 0:
            return image, 0, 0
        return image[:band_y, :], 0, 0

    def read_boxes(
        self,
        image: np.ndarray,
        mrz_band: Optional[mrz_detect.MrzBand] = None,
        *,
        document_id: str = "document",
    ) -> list[TextBox]:
        full_h, full_w = image.shape[:2]
        viz_image, y_off, x_off = self._exclude_mrz(image, mrz_band)
        if viz_image.size == 0:
            return []

        result = self._engine(viz_image)
        if result.boxes is None:
            return []

        boxes: list[TextBox] = []
        for poly, text, score in zip(result.boxes, result.txts, result.scores):
            xs = poly[:, 0] + x_off
            ys = poly[:, 1] + y_off
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            region = Region(
                x=x0 / full_w, y=y0 / full_h,
                w=(x1 - x0) / full_w, h=(y1 - y0) / full_h,
                document_id=document_id,
            )
            boxes.append(TextBox(text=text, confidence=float(score), region=region))
        return boxes


# --------------------------------------------------------------------------
# Field classifier -- PLACEHOLDER, not a finished component.
#
# BACKEND_BRIEF.md §6.2 step 4 specifies a trained classifier ("[normalised
# bbox, text_embedding, template_prior] -> field label, a 3-layer MLP or
# LightGBM trained on MIDV annotations"). We don't have MIDV annotations in
# this environment, so this is a rule-based stand-in: position on the page +
# regex text-pattern heuristics. It is structured so a trained classifier can
# drop in behind the same `classify_fields(boxes) -> list[VizField]`
# interface later without callers changing.
#
# VizField is an OCR-internal type, not the app/contracts ExtractedField:
# the contract's ExtractedField (FRONTEND_BRIEF.md §3) has no `region` --
# spatial signal placement uses Signal.region instead. reader.py (the OCR
# branch's public boundary) carries VizField.region through into the
# MRZ<->VIZ cross-check's Signal.region, then converts to the real
# contracts.ExtractedField for OCRResult.fields (key/label/value/confidence/
# source, no region).
# --------------------------------------------------------------------------


@dataclass
class VizField:
    name: str
    value: str
    confidence: float
    region: Region
    source: str = "viz"

_DATE_PATTERNS = [
    re.compile(r"^(?P<d>\d{2})[./\- ](?P<m>\d{2})[./\- ](?P<y>\d{2,4})$"),
    re.compile(r"^(?P<y>\d{4})[./\-](?P<m>\d{2})[./\-](?P<d>\d{2})$"),
]
_DOC_NUMBER_PATTERN = re.compile(r"^[A-Z0-9]{6,9}$")
_NATIONALITY_PATTERN = re.compile(r"^[A-Z]{3}$")
_SEX_PATTERN = re.compile(r"^[MFX]$")
_NAME_PATTERN = re.compile(r"^[A-Z][A-Z' \-]{2,}$")


def _try_parse_date(text: str) -> Optional[date]:
    cleaned = text.strip()
    for pattern in _DATE_PATTERNS:
        m = pattern.match(cleaned)
        if not m:
            continue
        try:
            y, mth, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
            if y < 100:
                y += 2000 if y <= (date.today().year % 100) + 1 else 1900
            return date(y, mth, d)
        except ValueError:
            continue
    return None


def _classify_date(d: date, *, today: Optional[date] = None) -> str:
    """birth_date if plausibly a birth (well in the past, reasonable age),
    else expiry_date. Both are dates, so position alone can't disambiguate a
    document that only has one -- this heuristic is the fallback."""
    today = today or date.today()
    age_years = (today - d).days / 365.25
    if 0 <= age_years <= 120:
        return "birth_date"
    return "expiry_date"


def classify_fields(boxes: list[TextBox]) -> list[VizField]:
    """Heuristic VIZ field classifier. See module docstring: placeholder,
    known accuracy limits, not a finished component."""
    fields: list[VizField] = []
    if not boxes:
        return fields

    sorted_boxes = sorted(boxes, key=lambda b: b.region.y)
    name_claimed = False

    for box in sorted_boxes:
        text = box.text.strip()
        if not text:
            continue
        upper = text.upper()

        parsed_date = _try_parse_date(text)
        if parsed_date is not None:
            field_name = _classify_date(parsed_date)
            fields.append(VizField(
                name=field_name, value=parsed_date.isoformat(),
                confidence=box.confidence * 0.7, region=box.region,
            ))
            continue

        if _SEX_PATTERN.match(upper) and len(upper) == 1:
            fields.append(VizField(
                name="sex", value=upper, confidence=box.confidence * 0.5,
                region=box.region,
            ))
            continue

        if _NATIONALITY_PATTERN.match(upper) and box.region.y < 0.5:
            fields.append(VizField(
                name="nationality", value=upper, confidence=box.confidence * 0.6,
                region=box.region,
            ))
            continue

        if _DOC_NUMBER_PATTERN.match(upper) and any(c.isdigit() for c in upper):
            fields.append(VizField(
                name="doc_number", value=upper, confidence=box.confidence * 0.6,
                region=box.region,
            ))
            continue

        if not name_claimed and box.region.y < 0.35 and _NAME_PATTERN.match(upper) and len(upper) >= 4:
            fields.append(VizField(
                name="name", value=upper, confidence=box.confidence * 0.6,
                region=box.region,
            ))
            name_claimed = True
            continue

        # Unclassified text (authority lines, addresses, etc.) is dropped
        # rather than mis-labelled -- a trained classifier will do better
        # here; the heuristic's job is to not confidently guess wrong.

    return fields
