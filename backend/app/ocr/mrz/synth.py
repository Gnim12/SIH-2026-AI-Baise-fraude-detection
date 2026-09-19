"""Synthetic MRZ generator + print-scan degradation.

Generates random, entirely fictional TD3 (passport) records with internally
consistent ICAO 9303 check digits, renders them as OCR-B-ish monospace text
images, and applies a print-scan degradation pipeline (blur, noise, contrast,
slight skew, resample) so the CRNN in model.py has training data that looks
like a scanned document rather than clean rendered text.

No real or scraped identity data is used anywhere here — names are random
letter sequences, not drawn from any real-name corpus.
"""
from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import spec

_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DIGITS = "0123456789"
# A pool of ISO-3166-1 alpha-3-shaped codes. These are not curated against the
# real registry -- they only need to be 3 letters for MRZ layout purposes.
_COUNTRY_POOL = ["UTO", "ZZA", "XKX", "TST", "QAX", "FAK", "SYN", "DEM"]

# The classic OCR-B confusable set -- also what decode.fake_logprobs biases
# its simulated misreads toward, and the pairs the checksum decoder exists to
# recover. Oversampling these in free-text fields gives the training set more
# examples of the exact confusions the decoder is meant to be good at.
_CONFUSABLE_CHARS = set("0O1I5S8B2Z")


def _weighted_pool(alphabet: str, confusable_bias: float) -> tuple[list[str], list[float]]:
    """Build a (population, weights) pair for `rng.choices`: characters in
    `_CONFUSABLE_CHARS` get `1 + confusable_bias` weight, everything else 1.
    `confusable_bias=0.0` reduces exactly to uniform sampling."""
    population = list(alphabet)
    weights = [1.0 + confusable_bias if c in _CONFUSABLE_CHARS else 1.0 for c in population]
    return population, weights


def _rand_letters(rng: random.Random, min_len: int, max_len: int, confusable_bias: float = 0.0) -> str:
    n = rng.randint(min_len, max_len)
    if confusable_bias <= 0.0:
        return "".join(rng.choice(_ALPHA) for _ in range(n))
    population, weights = _weighted_pool(_ALPHA, confusable_bias)
    return "".join(rng.choices(population, weights=weights, k=n))


def _rand_digits(rng: random.Random, n: int, confusable_bias: float = 0.0) -> str:
    if confusable_bias <= 0.0:
        return "".join(rng.choice(_DIGITS) for _ in range(n))
    population, weights = _weighted_pool(_DIGITS, confusable_bias)
    return "".join(rng.choices(population, weights=weights, k=n))


def _rand_doc_number(rng: random.Random, confusable_bias: float = 0.0) -> str:
    """Only the characters are biased; the document-number check digit is
    computed from the finished string in build_td3_record, so it is always
    consistent regardless of bias."""
    n = rng.randint(6, 9)
    if confusable_bias <= 0.0:
        body = "".join(rng.choice(_ALPHA + _DIGITS) for _ in range(n))
    else:
        population, weights = _weighted_pool(_ALPHA + _DIGITS, confusable_bias)
        body = "".join(rng.choices(population, weights=weights, k=n))
    return body.ljust(9, spec.FILLER)


def _rand_date(rng: random.Random, *, year_lo: int, year_hi: int) -> str:
    year = rng.randint(year_lo, year_hi)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)  # avoid month-length edge cases in synthetic data
    return f"{year % 100:02d}{month:02d}{day:02d}"


@dataclass
class SyntheticRecord:
    surname: str
    given_names: str
    doc_number: str
    nationality: str
    issuing_country: str
    birth_date_raw: str
    expiry_date_raw: str
    sex: str
    personal_number: str
    lines: list[str] = field(default_factory=list)


def _name_field(surname: str, given_names: str, width: int) -> str:
    raw = surname.replace(" ", "<") + "<<" + given_names.replace(" ", "<")
    if len(raw) > width:
        raw = raw[:width]
    return raw.ljust(width, spec.FILLER)


def build_td3_record(
    *,
    surname: str,
    given_names: str,
    doc_number: str,
    nationality: str,
    issuing_country: str,
    birth_raw: str,
    expiry_raw: str,
    sex: str,
    personal_number: str = "",
) -> SyntheticRecord:
    """Deterministically build one internally-consistent TD3 record from
    explicit field values, computing every check digit. Used both by
    `generate_record` (random inputs) and directly by tests that need exact
    control over field values."""
    doc_number = doc_number.ljust(9, spec.FILLER)
    personal_number = personal_number.ljust(14, spec.FILLER)

    l1 = "P<" + issuing_country + _name_field(surname, given_names, 39)
    assert len(l1) == 44

    doc_cd = spec.check_digit_char(doc_number)
    birth_cd = spec.check_digit_char(birth_raw)
    expiry_cd = spec.check_digit_char(expiry_raw)
    personal_cd = spec.check_digit_char(personal_number)
    composite_input = (
        doc_number + doc_cd + birth_raw + birth_cd + expiry_raw + expiry_cd + personal_number + personal_cd
    )
    composite_cd = spec.check_digit_char(composite_input)

    l2 = (
        doc_number + doc_cd + nationality + birth_raw + birth_cd + sex
        + expiry_raw + expiry_cd + personal_number + personal_cd + composite_cd
    )
    assert len(l2) == 44

    return SyntheticRecord(
        surname=surname,
        given_names=given_names.replace(" ", "<"),
        doc_number=doc_number.rstrip(spec.FILLER),
        nationality=nationality,
        issuing_country=issuing_country,
        birth_date_raw=birth_raw,
        expiry_date_raw=expiry_raw,
        sex=sex,
        personal_number=personal_number.rstrip(spec.FILLER),
        lines=[l1, l2],
    )


def generate_record(rng: random.Random, *, confusable_bias: float = 0.0) -> SyntheticRecord:
    """Build one internally-consistent, fictional TD3 MRZ record with
    randomised field values.

    `confusable_bias`: oversamples the classic OCR-B confusable characters
    (0/O, 1/I, 5/S, 8/B, 2/Z) in surname, given names, the optional
    personal-number field and the document number. Dates and country codes
    stay uniform, and every check digit is computed from the finished field
    text, so every record still self-validates regardless of the bias value.
    Default 0.0 is uniform sampling, unchanged from before this parameter
    existed.
    """
    surname = _rand_letters(rng, 3, 10, confusable_bias)
    given_names = _rand_letters(rng, 3, 10, confusable_bias) + (
        " " + _rand_letters(rng, 3, 8, confusable_bias) if rng.random() < 0.3 else ""
    )
    doc_number = _rand_doc_number(rng, confusable_bias)
    nationality = rng.choice(_COUNTRY_POOL)
    sex = rng.choice("MF")
    birth_raw = _rand_date(rng, year_lo=1950, year_hi=2010)
    expiry_raw = _rand_date(rng, year_lo=2024, year_hi=2033)
    personal_number = "" if rng.random() < 0.6 else _rand_digits(rng, rng.randint(4, 14), confusable_bias)

    return build_td3_record(
        surname=surname, given_names=given_names, doc_number=doc_number,
        nationality=nationality, issuing_country=nationality,
        birth_raw=birth_raw, expiry_raw=expiry_raw, sex=sex,
        personal_number=personal_number,
    )


# Vendored OCR-B (see assets/fonts/OCRB-LICENSE.txt for provenance/terms).
# This is the real MRZ font, not a substitute: OCR-B was designed so 0/O,
# 1/I and 5/S are visually distinct, which is exactly the confusion set the
# checksum-constrained decoder (decode.py) is meant to recover from -- a
# fallback monospace font trains the model on a differently-shaped problem.
_OCRB_PATH = Path(__file__).resolve().parents[3] / "assets" / "fonts" / "OCRB.ttf"


def _load_mono_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if _OCRB_PATH.exists():
        try:
            return ImageFont.truetype(str(_OCRB_PATH), size)
        except OSError:
            pass
    for candidate in ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


RENDER_MARGIN_PX = 4  # blank border on every side of a rendered band
FONT_SIZE_FRACTION = 0.8  # font size as a fraction of the line height


class RenderClippedError(RuntimeError):
    """A rendered MRZ line lost characters off the edge of its canvas."""


def font_advance(char_h: int) -> float:
    """Character pitch at the font size used for `char_h`: the widest advance of any MRZ character,
    measured from the font itself. Hinted fonts round advances to whole pixels and the vendored OCR-B
    varies by 1-2 px between glyphs, so it is neither a fixed ratio nor uniform; a real MRZ is fixed-pitch,
    so every character is drawn in its own cell of this width."""
    font = _load_mono_font(int(char_h * FONT_SIZE_FRACTION))
    return float(max(font.getlength(c) for c in spec.MRZ_CHARSET))


def render_mrz_lines(lines: list[str], *, char_h: int = 28) -> np.ndarray:
    """Render MRZ lines as a single grayscale image, one row of monospace text per line.

    The canvas width is computed from the font's measured advance --
    `n_chars * advance + 2 * margin` -- never a constant: a fixed 16 px pitch once cut the last ~5
    of 44 characters (including both check digits) off every training image. The result is
    checked, and RenderClippedError raised, if any text touches the canvas edge or the ink does
    not span the expected width.
    """
    n_chars = max(len(l) for l in lines)
    advance = font_advance(char_h)
    width = int(math.ceil(n_chars * advance)) + 2 * RENDER_MARGIN_PX
    height = len(lines) * char_h
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    font = _load_mono_font(int(char_h * FONT_SIZE_FRACTION))
    for i, line in enumerate(lines):
        for j, ch in enumerate(line):
            x = RENDER_MARGIN_PX + j * advance + (advance - font.getlength(ch)) / 2
            draw.text((x, i * char_h + 2), ch, fill=0, font=font)
    arr = np.array(img)
    _assert_not_clipped(arr, lines, advance, char_h)
    return arr


def _assert_not_clipped(arr: np.ndarray, lines: list[str], advance: float, char_h: int) -> None:
    ink = arr < 128
    cols = np.flatnonzero(ink.any(axis=0))
    rows = np.flatnonzero(ink.any(axis=1))
    if cols.size == 0:
        raise RenderClippedError("rendered MRZ has no ink at all")
    n_chars = max(len(l) for l in lines)
    # Every character cell must be inside the image: ink reaches into the last cell, leaves
    # the edges clear, and never touches the top/bottom rows either.
    last_cell_start = RENDER_MARGIN_PX + (n_chars - 1) * advance
    if arr.shape[1] - 1 in cols or 0 in cols or 0 in rows or arr.shape[0] - 1 in rows:
        raise RenderClippedError(f"ink touches the canvas edge (canvas {arr.shape}, char_h {char_h})")
    if cols[-1] < last_cell_start:
        raise RenderClippedError(
            f"ink ends at column {cols[-1]} but the last of {n_chars} characters starts at "
            f"{last_cell_start:.0f} (canvas {arr.shape[1]} px, advance {advance})"
        )


def degrade(image: np.ndarray, rng: random.Random, *, severity: float = 0.5) -> np.ndarray:
    """Print-scan style degradation: blur, noise, contrast/brightness jitter,
    slight resample, and mild JPEG-style compression loss. `severity` in [0, 1]."""
    from scipy.ndimage import gaussian_filter, rotate  # type: ignore[import-untyped]

    arr = image.astype(np.float32)

    angle = rng.uniform(-1.5, 1.5) * severity * 4
    if abs(angle) > 0.05:
        arr = rotate(arr, angle, reshape=False, mode="nearest", cval=255)

    sigma = 0.3 + severity * 1.4
    arr = gaussian_filter(arr, sigma=sigma)

    contrast = 1.0 - severity * rng.uniform(0.0, 0.35)
    brightness = rng.uniform(-1, 1) * severity * 25
    arr = (arr - 127.5) * contrast + 127.5 + brightness

    noise_std = severity * rng.uniform(3, 14)
    arr = arr + np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, noise_std, arr.shape)

    if severity > 0.3:
        scale = rng.uniform(0.6, 0.9)
        small = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).resize(
            (max(1, int(arr.shape[1] * scale)), max(1, int(arr.shape[0] * scale))), Image.Resampling.BILINEAR
        )
        arr = np.array(small.resize((arr.shape[1], arr.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)

    if severity > 0.5 and rng.random() < 0.4:
        buf = io.BytesIO()
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(
            buf, format="JPEG", quality=int(rng.uniform(30, 60))
        )
        buf.seek(0)
        arr = np.array(Image.open(buf).convert("L"), dtype=np.float32)

    return np.clip(arr, 0, 255).astype(np.uint8)


def generate_synthetic_page(
    rng: random.Random, *, severity: float = 0.5, confusable_bias: float = 0.0
) -> tuple[SyntheticRecord, np.ndarray]:
    """Generate one record and its degraded rendered MRZ image."""
    record = generate_record(rng, confusable_bias=confusable_bias)
    clean = render_mrz_lines(record.lines)
    return record, degrade(clean, rng, severity=severity)


if __name__ == "__main__":
    rng = random.Random(1337)
    n = 2000
    passed = 0
    for _ in range(n):
        record = generate_record(rng)
        result = spec.parse_td3(record.lines)
        if result.all_valid:
            passed += 1
        else:
            failing = [g.name for g in result.checks if not g.valid]
            print(f"FAIL: {record.lines} -> bad groups {failing}")

    print(f"synth.py self-test: {passed}/{n} generated records pass their own check digits")
    assert passed == n, f"only {passed}/{n} synthetic records were self-consistent"

    sample_rng = random.Random(42)
    record, image = generate_synthetic_page(sample_rng, severity=0.6)
    print(f"  rendered + degraded one sample page: shape={image.shape}, dtype={image.dtype}, "
          f"doc_number={record.doc_number}, surname={record.surname}")
    assert image.ndim == 2 and image.dtype == np.uint8

    # confusable_bias: every record must still self-validate, and the biased
    # free-text fields must show a measurably higher confusable-character rate.
    def _confusable_rate(records: list[SyntheticRecord]) -> float:
        text = "".join(r.surname + r.given_names + r.personal_number for r in records)
        text = text.replace(" ", "").replace("<", "")
        if not text:
            return 0.0
        return sum(1 for c in text if c in _CONFUSABLE_CHARS) / len(text)

    bias_rng = random.Random(7)
    unbiased = [generate_record(bias_rng, confusable_bias=0.0) for _ in range(200)]
    biased = [generate_record(bias_rng, confusable_bias=0.6) for _ in range(200)]

    biased_passed = sum(1 for r in biased if spec.parse_td3(r.lines).all_valid)
    assert biased_passed == 200, f"only {biased_passed}/200 confusable_bias=0.6 records self-validated"

    rate_unbiased = _confusable_rate(unbiased)
    rate_biased = _confusable_rate(biased)
    print(f"  confusable_bias=0.0 free-text confusable-char rate: {rate_unbiased:.3f}")
    print(f"  confusable_bias=0.6 free-text confusable-char rate: {rate_biased:.3f}")
    assert rate_biased > rate_unbiased, "confusable_bias=0.6 did not raise the confusable-char rate"
    print(f"  200/200 confusable_bias=0.6 records passed check-digit validation, "
          f"confusable rate {rate_unbiased:.3f} -> {rate_biased:.3f}")

    print("synth.py self-test OK")
