"""Single source of truth for the MRZ recogniser's input geometry.

`CANONICAL_WIDTH` must be a width whose conv-backbone output length is an
exact multiple of the line length (700 -> T=176 = 4 x 44), because the ONNX
export of AdaptiveAvgPool1d only lowers that case (scripts/export_mrz_onnx.py).
Training (data.py), export, and inference (infer.py) all import it from here
so the three can never drift apart.

The fixed-slot head reads slot n from a fixed horizontal band of the canvas, so
the *ink* -- not the crop -- has to land in the same place every time. A band
with padding, or text rendered at a different glyph height, must normalise to
the same image as a tight crop. `normalize_line` therefore finds the text's
ink extent, scales it uniformly so its width matches the ink width of the
training canvases, and places it where training ink sits. X and Y are scaled
independently: the head was trained at one glyph aspect (pitch : cap height).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

CANONICAL_WIDTH = 700
CANONICAL_HEIGHT = 32
_BACKGROUND = 255

# Where the ink of a FULL 44-character line sits, in the frame the 1.4.0 weights
# were trained on. Measured on unclipped OCR-B renders at 25 px (advance 18 px):
# ink 785 px wide (780-788), rows 7..25. The training renders were cut off at
# 704 px, so the canvas below only ever showed the first ~39 characters; this
# reproduces that exactly. When the dataset renderer is fixed and the model
# retrained (B1i), TARGET_INK_WIDTH becomes ~CANONICAL_WIDTH - 2 * TARGET_INK_LEFT.
TARGET_INK_LEFT = 6
TARGET_INK_WIDTH = 785
TARGET_INK_TOP = 7.0
TARGET_INK_HEIGHT = 19.0

# Context kept around the ink box before scaling; a constant, not a magic number.
INK_MARGIN_PX = 2

# Ink detection. The crop is lightly blurred (noise is zero-mean, strokes are not),
# then each pixel contributes its darkness beyond a noise floor: a fraction of
# the crop's dark range (background = 90th percentile, darkest = 1st percentile
# of the blurred crop, so a lone speckle moves neither). Column/row "ink mass"
# is the sum of those contributions. A soft mass, unlike a hard binary
# threshold, keeps thin, blurred '<' fillers.
_SMOOTH_RADIUS = 1.0
_NOISE_FLOOR_FRACTION = 0.2
# A column/row is inked when its mass is at least this fraction of a typical
# (90th percentile) inked column/row ...
_RELATIVE_INK = 0.10
# ... and it belongs to a run of at least this many consecutive such columns/rows.
# Together these reject an isolated speckle or a stray edge pixel.
_MIN_RUN = 3
_MIN_CONTRAST = 20.0


def _extent(mass: NDArray[np.float64]) -> tuple[int, int] | None:
    """First/last index of the first/last run (>= _MIN_RUN) of inked positions."""
    positive = mass[mass > 0]
    if positive.size == 0:
        return None
    threshold = _RELATIVE_INK * float(np.percentile(positive, 90))
    on = np.concatenate(([False], mass >= max(threshold, 1e-9), [False]))
    edges = np.flatnonzero(on[1:] != on[:-1])
    starts, ends = edges[0::2], edges[1::2]
    keep = (ends - starts) >= _MIN_RUN
    if not keep.any():
        return None
    return int(starts[keep][0]), int(ends[keep][-1]) - 1


def ink_extent(image: NDArray[np.uint8]) -> tuple[int, int, int, int] | None:
    """(x0, x1, y0, y1), inclusive, of the text ink in a grayscale crop; None when no text is found."""
    smooth = np.asarray(
        Image.fromarray(image).filter(ImageFilter.GaussianBlur(_SMOOTH_RADIUS)), dtype=np.float64
    )
    background = float(np.percentile(smooth, 90))
    dark_range = background - float(np.percentile(smooth, 1))
    if dark_range < _MIN_CONTRAST:  # no usable contrast
        return None
    ink = np.clip(background - smooth - _NOISE_FLOOR_FRACTION * dark_range, 0.0, None)
    xs = _extent(ink.sum(axis=0))
    ys = _extent(ink.sum(axis=1))
    if xs is None or ys is None:
        return None
    return xs[0], xs[1], ys[0], ys[1]


def _fit_whole_crop(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """The pre-B1h behaviour: fit the whole crop, centre-padded. Fallback when no ink is found."""
    h, w = image.shape
    scale = min(CANONICAL_HEIGHT / h, CANONICAL_WIDTH / w)
    new_h = max(1, min(CANONICAL_HEIGHT, int(round(h * scale))))
    new_w = max(1, min(CANONICAL_WIDTH, int(round(w * scale))))
    if (new_h, new_w) != (h, w):
        image = np.array(Image.fromarray(image).resize((new_w, new_h), Image.Resampling.BILINEAR))
    canvas = np.full((CANONICAL_HEIGHT, CANONICAL_WIDTH), _BACKGROUND, dtype=np.uint8)
    top = (CANONICAL_HEIGHT - new_h) // 2
    left = (CANONICAL_WIDTH - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = image
    return canvas


def normalize_line(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Map a grayscale line crop to the (CANONICAL_HEIGHT, CANONICAL_WIDTH) canvas.

    Crop to the ink extent (plus INK_MARGIN_PX), scale it so the ink box is
    TARGET_INK_WIDTH x TARGET_INK_HEIGHT (x and y independently, since a
    hinted font, a camera or a printer all change the glyph aspect), and paste
    it with the ink's top-left at (TARGET_INK_LEFT, TARGET_INK_TOP). Padding
    and glyph height in the input therefore do not change the output. Nothing
    inside the ink box is cropped before pasting; the canvas edge then clips
    the tail exactly as it did in training. The input must hold the whole line.
    Falls back to whole-crop fitting when no text is detected.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale crop, got shape {image.shape}")
    extent = ink_extent(image)
    if extent is None:
        return _fit_whole_crop(image)

    x0, x1, y0, y1 = extent
    h, w = image.shape
    cx0, cx1 = max(0, x0 - INK_MARGIN_PX), min(w, x1 + 1 + INK_MARGIN_PX)
    cy0, cy1 = max(0, y0 - INK_MARGIN_PX), min(h, y1 + 1 + INK_MARGIN_PX)
    crop = image[cy0:cy1, cx0:cx1]

    scale_x = TARGET_INK_WIDTH / (x1 - x0 + 1)
    scale_y = TARGET_INK_HEIGHT / (y1 - y0 + 1)
    new_w = max(1, int(round(crop.shape[1] * scale_x)))
    new_h = max(1, int(round(crop.shape[0] * scale_y)))
    scaled = np.array(Image.fromarray(crop).resize((new_w, new_h), Image.Resampling.BILINEAR))

    background = int(np.percentile(image, 90))
    canvas = np.full((CANONICAL_HEIGHT, CANONICAL_WIDTH), background, dtype=np.uint8)
    left = int(round(TARGET_INK_LEFT - (x0 - cx0) * scale_x))
    top = int(round(TARGET_INK_TOP - (y0 - cy0) * scale_y))
    # Clip the scaled crop to the canvas.
    sx0, sy0 = max(0, -left), max(0, -top)
    dx0, dy0 = max(0, left), max(0, top)
    cw = min(new_w - sx0, CANONICAL_WIDTH - dx0)
    ch = min(new_h - sy0, CANONICAL_HEIGHT - dy0)
    if cw > 0 and ch > 0:
        canvas[dy0 : dy0 + ch, dx0 : dx0 + cw] = scaled[sy0 : sy0 + ch, sx0 : sx0 + cw]
    return canvas
