"""Shared helpers for building synthetic full-document test images: a VIZ
zone (rendered via PIL, read by the real RapidOCR-backed VizReader) stacked
above an MRZ band (rendered via mrz.synth, read by MRZReader in stub mode).
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.ocr.mrz import spec, synth
from app.ocr.mrz.infer import MRZReader
from app.ocr.reader import OCRReader
from app.ocr.viz import VizReader

VIZ_WIDTH = 700
VIZ_HEIGHT = 420


def _font(size: int):
    for candidate in ("consola.ttf", "cour.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _mrz_date_to_viz(raw_yymmdd: str, *, past_bias: bool) -> str:
    # Real VIZ printing uses a full 4-digit year (unlike the MRZ's YYMMDD),
    # so resolve the century the same way spec.py does when parsing the MRZ.
    d = spec.parse_mrz_date(raw_yymmdd, past_bias=past_bias)
    assert d is not None, raw_yymmdd
    return f"{d.day:02d}/{d.month:02d}/{d.year:04d}"


def render_viz_zone(
    record: synth.SyntheticRecord,
    *,
    given_names_display: str | None = None,
    doc_number_override: str | None = None,
    nationality_override: str | None = None,
    birth_display_override: str | None = None,
    expiry_display_override: str | None = None,
    sex_override: str | None = None,
) -> np.ndarray:
    """Render a VIZ block whose printed fields match `record` by default.
    Any *_override lets a test deliberately alter one printed field to
    diverge from the MRZ, without touching the MRZ itself."""
    img = Image.new("L", (VIZ_WIDTH, VIZ_HEIGHT), color=255)
    draw = ImageDraw.Draw(img)
    f = _font(26)

    given = given_names_display if given_names_display is not None else record.given_names.replace("<", " ")
    name_line = f"{record.surname} {given}".strip()
    doc_number = doc_number_override if doc_number_override is not None else record.doc_number
    nationality = nationality_override if nationality_override is not None else record.nationality
    birth_display = (
        birth_display_override if birth_display_override is not None
        else _mrz_date_to_viz(record.birth_date_raw, past_bias=True)
    )
    expiry_display = (
        expiry_display_override if expiry_display_override is not None
        else _mrz_date_to_viz(record.expiry_date_raw, past_bias=False)
    )
    sex = sex_override if sex_override is not None else record.sex

    draw.text((40, 30), name_line, fill=0, font=f)
    draw.text((40, 100), doc_number, fill=0, font=f)
    draw.text((40, 170), nationality, fill=0, font=f)
    draw.text((40, 240), birth_display, fill=0, font=f)
    draw.text((40, 310), expiry_display, fill=0, font=f)
    draw.text((40, 370), sex, fill=0, font=f)

    return np.array(img)


def build_document(viz_zone: np.ndarray, mrz_severity: float = 0.15, seed: int = 0) -> np.ndarray:
    """Stack a VIZ zone above a rendered MRZ band to make one full document
    image, with a rendered (but not necessarily content-accurate, since
    MRZReader stub mode ignores pixel content) MRZ band at the bottom so
    mrz.detect.find_mrz can locate it."""
    import random

    from app.ocr.mrz import synth as _synth

    rng = random.Random(seed)
    # The pixel content of the MRZ band doesn't need to match the ground
    # truth passed to stub mode -- detect.find_mrz only needs *something*
    # band-shaped to lock onto. We still render real MRZ-looking text so the
    # classical-CV detector's assumptions (dense monospace band, bottom of
    # page) hold exactly as they would for a real capture.
    placeholder_lines = ["A" * 44, "A" * 44]
    mrz_img = _synth.render_mrz_lines(placeholder_lines, char_h=48)
    mrz_img = _synth.degrade(mrz_img, rng, severity=mrz_severity)

    page_w = max(VIZ_WIDTH, mrz_img.shape[1] + 80)
    page_h = VIZ_HEIGHT + mrz_img.shape[0] + 60
    page = np.full((page_h, page_w), 255, dtype=np.uint8)
    page[0:VIZ_HEIGHT, 0:VIZ_WIDTH] = viz_zone
    y_off = page_h - mrz_img.shape[0] - 20
    page[y_off : y_off + mrz_img.shape[0], 20 : 20 + mrz_img.shape[1]] = mrz_img

    return np.stack([page] * 3, axis=-1)


@pytest.fixture(scope="session")
def viz_reader() -> VizReader:
    return VizReader()


@pytest.fixture()
def stub_mrz_reader() -> MRZReader:
    return MRZReader(require_trained_weights=False)


@pytest.fixture()
def ocr_reader(stub_mrz_reader: MRZReader, viz_reader: VizReader) -> OCRReader:
    return OCRReader(mrz_reader=stub_mrz_reader, viz_reader=viz_reader)
