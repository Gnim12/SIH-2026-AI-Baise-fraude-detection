"""Single source of truth for the MRZ recogniser's input geometry.

`CANONICAL_WIDTH` must be a width whose conv-backbone output length is an
exact multiple of the line length (700 -> T=176 = 4 x 44), because the ONNX
export of AdaptiveAvgPool1d only lowers that case (scripts/export_mrz_onnx.py).
Training (data.py), export, and inference (infer.py) all import it from here
so the three can never drift apart.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

CANONICAL_WIDTH = 700
CANONICAL_HEIGHT = 32
_BACKGROUND = 255


def normalize_line(image: np.ndarray) -> np.ndarray:
    """Fit a grayscale line crop into a (CANONICAL_HEIGHT, CANONICAL_WIDTH) canvas.

    One uniform scale is applied, so character shapes and the spacing between
    characters (the fixed grid the slot head relies on) are preserved. The
    scale is the smaller of the height fit and the width fit, so the whole
    crop always survives: it is brought to canonical height, and if that
    would make it wider than the canvas it is shrunk further until it fits.
    The result is centre-padded with background. Nothing is ever cropped.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale crop, got shape {image.shape}")
    h, w = image.shape
    scale = min(CANONICAL_HEIGHT / h, CANONICAL_WIDTH / w)
    new_h = max(1, min(CANONICAL_HEIGHT, int(round(h * scale))))
    new_w = max(1, min(CANONICAL_WIDTH, int(round(w * scale))))

    if (new_h, new_w) != (h, w):
        image = np.array(
            Image.fromarray(image).resize((new_w, new_h), Image.Resampling.BILINEAR)
        )

    canvas = np.full((CANONICAL_HEIGHT, CANONICAL_WIDTH), _BACKGROUND, dtype=np.uint8)
    top = (CANONICAL_HEIGHT - new_h) // 2
    left = (CANONICAL_WIDTH - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = image
    return canvas
