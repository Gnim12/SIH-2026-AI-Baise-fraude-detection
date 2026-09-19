"""Milestone B1h: ink-extent normalisation of MRZ line crops."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from app.ocr.mrz import canonical, data, synth

FULL_LINE = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
FILLER_LINE = "P<UTOERIKSSON<<ANNA<MARI".ljust(44, "<")
NUM_LINE_CHARS = 44


def render_full(text: str, size: int = 25) -> NDArray[np.uint8]:
    """One complete line on a canvas wide enough for all 44 characters."""
    font = synth._load_mono_font(size)
    width = int(font.getlength("A" * NUM_LINE_CHARS)) + 4
    img = Image.new("L", (width, 32), color=255)
    ImageDraw.Draw(img).text((2, 2), text, fill=0, font=font)
    return np.array(img)


def pad(img: NDArray[np.uint8], left: int, right: int, top: int, bottom: int, value: int = 255) -> NDArray[np.uint8]:
    return np.pad(img, ((top, bottom), (left, right)), constant_values=value)


def test_padded_and_unpadded_band_normalise_to_the_same_image():
    tight = render_full(FULL_LINE)
    # roughly 15% padding on every side, as detect.py adds
    padded = pad(tight, 100, 100, 6, 6)
    a, b = canonical.normalize_line(tight), canonical.normalize_line(padded)
    assert a.shape == b.shape == (canonical.CANONICAL_HEIGHT, canonical.CANONICAL_WIDTH)
    assert np.abs(a.astype(int) - b.astype(int)).mean() < 2.0
    assert canonical.ink_extent(a)[:2] == pytest.approx(canonical.ink_extent(b)[:2], abs=2)


def test_padding_with_paper_grey_background_is_also_invariant():
    tight = render_full(FULL_LINE)
    grey = np.clip(tight.astype(int) * 0.92, 0, 255).astype(np.uint8)  # paper at ~235
    a, b = canonical.normalize_line(tight), canonical.normalize_line(pad(grey, 90, 60, 8, 4, value=235))
    assert canonical.ink_extent(a)[:2] == pytest.approx(canonical.ink_extent(b)[:2], abs=2)


@pytest.mark.parametrize("text", [FILLER_LINE, "P<UTO" + "<" * 39, FULL_LINE[:24] + "<" * 20])
def test_line_ending_in_20_fillers_crops_to_the_last_filler(text: str):
    assert text[-20:] == "<" * 20 and len(text) == NUM_LINE_CHARS
    img = render_full(text)
    x0, x1, _, _ = canonical.ink_extent(img)
    font = synth._load_mono_font(25)
    # the last '<' starts at 43 advances; its ink must be inside the detected extent
    assert x1 >= 2 + 43 * font.getlength("A") + 4
    # and a faint, blurred version still reaches the last filler
    from scipy.ndimage import gaussian_filter

    faint = (255 - (255 - gaussian_filter(img.astype(float), 1.5)) * 0.55).astype(np.uint8)
    fx0, fx1, _, _ = canonical.ink_extent(faint)
    assert fx1 >= x1 - 6 and fx0 <= x0 + 6


def test_speckle_does_not_defeat_extent_detection():
    img = render_full(FULL_LINE)
    x0, x1, y0, y1 = canonical.ink_extent(img)
    rng = np.random.default_rng(0)
    noisy = img.copy()
    # isolated dark pixels far outside the text, at all four edges
    for y, x in [(0, 0), (31, 795), (0, 790), (16, 0), (16, 795), (31, 3)]:
        noisy[y, x] = 0
    # plus dense salt noise everywhere
    mask = rng.random(noisy.shape) < 0.01
    noisy[mask] = 0
    nx0, nx1, ny0, ny1 = canonical.ink_extent(noisy)
    assert (abs(nx0 - x0), abs(nx1 - x1), abs(ny0 - y0), abs(ny1 - y1)) <= (2, 2, 2, 2)


def test_no_character_is_lost_by_cropping():
    for text in (FULL_LINE, FILLER_LINE):
        img = render_full(text)
        x0, x1, _, _ = canonical.ink_extent(img)
        pitch = synth._load_mono_font(25).getlength("A")
        # extent spans all 44 character cells (43 pitches + a glyph)
        assert (x1 - x0 + 1) >= 43 * pitch
        # padded to the extent, nothing inside it is dropped before pasting
        assert canonical.ink_extent(pad(img, 60, 60, 5, 5))[1] - canonical.ink_extent(pad(img, 60, 60, 5, 5))[0] + 1 >= 43 * pitch


def test_glyph_height_does_not_change_the_normalised_ink_box():
    boxes = []
    for size in (19, 22, 25, 28, 32):
        n = canonical.normalize_line(render_full(FULL_LINE, size))
        x0, x1, y0, y1 = canonical.ink_extent(n)
        boxes.append((x0, y0, y1))
    assert max(b[0] for b in boxes) - min(b[0] for b in boxes) <= 3
    assert max(b[1] for b in boxes) - min(b[1] for b in boxes) <= 3
    assert max(b[2] for b in boxes) - min(b[2] for b in boxes) <= 3


def test_blank_crop_falls_back_without_error():
    out = canonical.normalize_line(np.full((30, 500), 240, dtype=np.uint8))
    assert out.shape == (canonical.CANONICAL_HEIGHT, canonical.CANONICAL_WIDTH)


def test_training_collate_and_inference_normalisation_agree(tmp_path: Path):
    rng = random.Random(11)
    line = synth.generate_record(rng).lines[1]
    image = synth.degrade(synth.render_mrz_lines([line], char_h=data.TARGET_HEIGHT), rng, severity=0.3)

    inference = canonical.normalize_line(image)
    via_collate = data.collate_batch([data.MrzSample(image=image, text=line)])["images"][0, 0].numpy()
    assert np.array_equal(np.round(via_collate * 255).astype(np.uint8), inference)

    # and through the on-disk dataset the trainer actually used (no jitter)
    split = tmp_path / "split"
    split.mkdir()
    Image.fromarray(image).save(split / "a.png")
    (split / "manifest.jsonl").write_text(json.dumps({"filename": "a.png", "line_index": 1, "label": line}) + "\n")
    sample = data.MrzDiskDataset(split, jitter_frac=0.0)[0]
    disk = data.collate_batch([sample])["images"][0, 0].numpy()
    assert np.array_equal(np.round(disk * 255).astype(np.uint8), inference)


# --- page path accuracy (needs the installed weights) -------------------------

BACKEND_ROOT = Path(__file__).resolve().parents[2]
INSTALLED_VERSIONS = sorted((BACKEND_ROOT / "models" / "mrz_crnn").glob("*/model.onnx"))
WEIGHTS = INSTALLED_VERSIONS[-1] if INSTALLED_VERSIONS else BACKEND_ROOT / "models" / "mrz_crnn" / "none" / "model.onnx"
TWO_LINE_BASELINE_CER = 0.026  # B1g: two lines at 32 px split evenly, severity 0.3


@pytest.mark.skipif(not WEIGHTS.exists(), reason="trained weights not installed")
def test_page_path_error_is_at_or_below_the_two_line_baseline():
    import importlib.util
    import sys

    import onnxruntime as ort

    from app.ocr.mrz import decode, spec

    path = BACKEND_ROOT / "scripts" / "measure_geometry.py"
    spec_ = importlib.util.spec_from_file_location("measure_geometry_t", path)
    assert spec_ is not None and spec_.loader is not None
    mod = importlib.util.module_from_spec(spec_)
    sys.modules["measure_geometry_t"] = mod
    spec_.loader.exec_module(mod)

    mod.configure(WEIGHTS)
    session = ort.InferenceSession(str(WEIGHTS), providers=["CPUExecutionProvider"])
    rng = random.Random(4242)
    wrong = total = 0
    for _ in range(60):
        rec = synth.generate_record(rng, confusable_bias=0.5)
        band = synth.degrade(mod.render(rec.lines, 32, "full"), rng, severity=0.3)
        found = mod.detect.find_mrz(mod.embed_on_page(band))
        assert found is not None
        lp = mod.infer(session, list(found.lines)).reshape(2, -1, len(spec.MRZ_CHARSET))
        d = decode.decode_mrz([lp[0], lp[1]], spec.MrzFormat.TD3)
        for got, truth in zip(d.greedy_lines, rec.lines):
            wrong += sum(a != b for a, b in zip(got, truth))
            total += len(truth)
    assert wrong / total <= TWO_LINE_BASELINE_CER, f"page-path CER {wrong / total:.2%}"


def test_detect_deskew_removes_tilt_instead_of_doubling_it():
    import sys

    from scipy.ndimage import rotate

    from app.ocr.mrz import detect

    sys.path.insert(0, str(BACKEND_ROOT / "scripts"))
    band = render_full(FULL_LINE)
    two = np.vstack([band, render_full(FILLER_LINE)])
    tilted = rotate(two.astype(np.float32), 3, reshape=False, mode="nearest", cval=255).astype(np.uint8)
    page = np.full((600, two.shape[1] + 80), 235, dtype=np.uint8)
    page[600 - two.shape[0] - 20 : 600 - 20, 40 : 40 + two.shape[1]] = tilted
    found = detect.find_mrz(np.stack([page] * 3, axis=-1))
    assert found is not None
    ref = canonical.ink_extent(band)
    assert ref is not None
    for line in found.lines:
        ext = canonical.ink_extent(line)
        assert ext is not None and (ext[1] - ext[0] + 1) >= 0.97 * (ref[1] - ref[0] + 1)


def test_target_ink_width_fits_all_44_characters_in_the_canvas():
    assert canonical.TARGET_INK_LEFT + canonical.TARGET_INK_WIDTH <= canonical.CANONICAL_WIDTH - canonical.TARGET_INK_LEFT
    assert canonical.TARGET_INK_WIDTH == 688
    out = canonical.normalize_line(render_full(FULL_LINE))
    x0, x1, _, _ = canonical.ink_extent(out)
    assert x0 >= canonical.TARGET_INK_LEFT - 2 and x1 <= canonical.CANONICAL_WIDTH - canonical.TARGET_INK_LEFT + 2
