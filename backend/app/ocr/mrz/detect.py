"""Classical-CV MRZ band detection: locate, deskew, split into lines.

No training needed. The approach: the MRZ is a dense band of high-contrast
fixed-pitch text hugging the bottom edge of the document. A horizontal
gradient + morphological closing turns that band into one solid blob that is
easy to separate from the rest of the page with a projection profile, even
under moderate skew.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover - exercised only in envs without opencv
    _HAS_CV2 = False


# The morphological kernels below are absolute-pixel and were tuned for a page
# about this wide (an MRZ line spans ~90% of the page, so its character pitch
# comes out near 18 px). Pages are resampled to this width before detection so a
# 640 px preview and a 4000 px photo behave alike; bboxes are mapped back.
WORKING_PAGE_WIDTH = 900


class BandSplitError(RuntimeError):
    """A band was located but could not be split into the expected lines."""


@dataclass
class MrzBand:
    """A located MRZ band within a document image.

    `bbox`: (x, y, w, h) in pixels within the input image.
    `region`: normalised (x, y, w, h) in 0..1, ready to drop into a
    `contracts.Region` once that module exists (see reader.py).
    `lines`: deskewed, cropped single-line grayscale images, top to bottom.
    """

    bbox: tuple[int, int, int, int]
    region: tuple[float, float, float, float]
    angle_deg: float
    lines: list[np.ndarray]


def _require_cv2() -> None:
    if not _HAS_CV2:
        raise RuntimeError(
            "opencv-python-headless is required for app.ocr.mrz.detect "
            "(pip install opencv-python-headless)"
        )


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def find_mrz(
    image: np.ndarray,
    *,
    n_lines: int = 2,
    bottom_fraction: float = 0.45,
) -> MrzBand | None:
    """Locate the MRZ band in a rectified document image.

    Searches the bottom `bottom_fraction` of the image (MRZ always hugs the
    bottom edge on TD1/TD2/TD3), finds the widest dense high-contrast blob via
    morphological closing, deskews it, and splits it into `n_lines` equal
    horizontal strips. Returns None if no plausible band is found.
    """
    _require_cv2()
    scale = WORKING_PAGE_WIDTH / image.shape[1]
    if abs(scale - 1.0) > 0.02:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        image = cv2.resize(image, (WORKING_PAGE_WIDTH, max(1, round(image.shape[0] * scale))), interpolation=interpolation)
    else:
        scale = 1.0
    h, w = image.shape[:2]
    gray = _to_gray(image)

    search_top = int(h * (1 - bottom_fraction))
    roi = gray[search_top:h, :]

    blackhat = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)))
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    grad_max = grad.max() if grad.max() > 0 else 1.0
    grad = (grad / grad_max * 255).astype(np.uint8)

    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5)))
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 15)))
    thresh = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < roi.shape[1] * 0.5:  # MRZ spans most of the document width
            continue
        aspect = cw / max(ch, 1)
        if aspect < 4:  # must be a wide, short band, not a logo or photo
            continue
        candidates.append((x, y, cw, ch))

    if not candidates:
        return None

    # Widest candidate wins.
    x, y, cw, ch = max(candidates, key=lambda b: b[2])
    pad = int(ch * 0.15)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(roi.shape[1], x + cw + pad), min(roi.shape[0], y + ch + pad)
    band = roi[y0:y1, x0:x1]
    abs_y0 = search_top + y0

    angle = _estimate_skew(band)
    if abs(angle) > 0.2:
        # cv2.getRotationMatrix2D rotates counter-clockwise for positive angles, while _estimate_skew
        # reports the tilt of the text; undoing the tilt needs the opposite sign (B1h: the old sign doubled it).
        band = _rotate(band, -angle)

    lines = _split_lines(band, n_lines)
    bad = [i for i, ln in enumerate(lines) if ln.ndim != 2 or ln.shape[0] < 2 or ln.shape[1] < 2]
    if len(lines) != n_lines or bad:
        raise BandSplitError(
            f"MRZ band located at x={x0}, y={abs_y0}, {x1 - x0}x{y1 - y0}px, but could not be split "
            f"into {n_lines} readable lines (got {len(lines)}; unusable line indices {bad})."
        )

    bbox = (x0, abs_y0, x1 - x0, y1 - y0)
    region = (bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h)
    bbox = (round(bbox[0] / scale), round(bbox[1] / scale), round(bbox[2] / scale), round(bbox[3] / scale))
    return MrzBand(bbox=bbox, region=region, angle_deg=angle, lines=lines)


def _estimate_skew(band: np.ndarray) -> float:
    _require_cv2()
    _, thresh = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 20:
        return 0.0
    rect = cv2.minAreaRect(coords.astype(np.float32))
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    return float(angle)


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    _require_cv2()
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _split_lines(band: np.ndarray, n_lines: int) -> list[np.ndarray]:
    """Split a deskewed MRZ band into `n_lines` strips using the horizontal
    ink-density projection profile to find the gaps between text rows."""
    _require_cv2()
    _, thresh = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    row_density = thresh.sum(axis=1).astype(np.float64)

    h = band.shape[0]
    if row_density.max() == 0:
        # No ink detected -- fall back to naive equal-height slicing.
        step = h // n_lines
        return [band[i * step : (i + 1) * step] for i in range(n_lines)]

    is_text_row = row_density > (row_density.max() * 0.08)
    boundaries: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(is_text_row):
        if v and not in_run:
            start, in_run = i, True
        elif not v and in_run:
            boundaries.append((start, i))
            in_run = False
    if in_run:
        boundaries.append((start, h))

    if len(boundaries) != n_lines:
        step = h // n_lines
        return [band[i * step : (i + 1) * step] for i in range(n_lines)]

    lines = []
    pad = 3
    for start, end in boundaries:
        lines.append(band[max(0, start - pad) : min(h, end + pad)])
    return lines


if __name__ == "__main__":
    from . import synth

    _require_cv2()
    rng_seeded = __import__("random").Random(7)
    hits = 0
    n = 40
    for i in range(n):
        record, mrz_img = synth.generate_synthetic_page(rng_seeded, severity=0.3)
        # Paste the MRZ image onto a larger blank "document" so detection has
        # to actually find it, not just accept the whole frame.
        page_h, page_w = mrz_img.shape[0] * 6, mrz_img.shape[1] + 80
        page = np.full((page_h, page_w), 235, dtype=np.uint8)
        y_off = page_h - mrz_img.shape[0] - 20
        x_off = 20
        page[y_off : y_off + mrz_img.shape[0], x_off : x_off + mrz_img.shape[1]] = mrz_img
        page_bgr = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)

        band = find_mrz(page_bgr, n_lines=2)
        if band is not None and len(band.lines) == 2:
            hits += 1

    print(f"detect.py self-test: located a 2-line MRZ band in {hits}/{n} synthetic pages")
    assert hits >= n * 0.9, f"only {hits}/{n} bands located -- detector regressed"
    print("detect.py self-test OK")
