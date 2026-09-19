"""Milestone B1i: the renderer measures its own canvas; augmentation never clips text."""
from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import ImageFont

from app.ocr.mrz import canonical, data, synth

LAST_SIX_NON_FILLER = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<ZE184226"
LINE2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
HEIGHTS = (16, 22, 28, 32, 36, 40, 44)


def test_fixture_line_really_ends_in_six_non_fillers():
    assert len(LAST_SIX_NON_FILLER) == 44 and "<" not in LAST_SIX_NON_FILLER[-6:]


@pytest.mark.parametrize("char_h", HEIGHTS)
def test_canvas_width_is_computed_from_the_measured_advance(char_h):
    advance = synth.font_advance(char_h)
    font = synth._load_mono_font(int(char_h * synth.FONT_SIZE_FRACTION))
    assert advance == max(font.getlength(c) for c in synth.spec.MRZ_CHARSET)
    img = synth.render_mrz_lines([LINE2], char_h=char_h)
    assert img.shape[1] == int(np.ceil(44 * advance)) + 2 * synth.RENDER_MARGIN_PX


@pytest.mark.parametrize("char_h", HEIGHTS)
def test_all_44_characters_are_in_the_ink_extent(char_h):
    img = synth.render_mrz_lines([LAST_SIX_NON_FILLER], char_h=char_h)
    advance = synth.font_advance(char_h)
    x0, x1, _, _ = canonical.ink_extent(img)
    # the sixth-from-last character starts at cell 38: ink must reach past it, to the last cell
    assert x1 >= synth.RENDER_MARGIN_PX + 43 * advance
    for cell in range(38, 44):  # each of the last six cells holds ink
        lo = int(synth.RENDER_MARGIN_PX + cell * advance)
        assert (img[:, lo : int(lo + advance)] < 128).any(), f"cell {cell} empty at char_h {char_h}"


def test_old_fixed_pitch_canvas_would_have_clipped_and_is_now_impossible():
    """44 x 16 px = 704 px is what the 1.4.0 set used; at 32 px the pitch is 19, so it cannot fit."""
    assert synth.font_advance(32) * 44 > 704
    assert synth.render_mrz_lines([LINE2], char_h=32).shape[1] > 704


def test_clipped_render_fails_loudly():
    img = synth.render_mrz_lines([LINE2], char_h=32)
    clipped = img[:, : img.shape[1] - 60]
    with pytest.raises(synth.RenderClippedError):
        synth._assert_not_clipped(clipped, [LINE2], synth.font_advance(32), 32)


def test_every_glyph_size_in_the_training_range_renders_without_error():
    rng = random.Random(0)
    for h in range(data.GLYPH_HEIGHT_RANGE[0], data.GLYPH_HEIGHT_RANGE[1] + 1):
        rec = synth.generate_record(rng, confusable_bias=0.5)
        synth.render_mrz_lines(rec.lines, char_h=h)


def test_rendered_characters_sit_on_a_uniform_grid():
    h = 32
    img = synth.render_mrz_lines(["<" * 44], char_h=h)
    advance = synth.font_advance(h)
    for cell in (0, 21, 43):
        lo = int(synth.RENDER_MARGIN_PX + cell * advance)
        assert (img[:, lo : int(lo + advance)] < 128).any()


def test_offset_jitter_never_clips_text():
    rng = random.Random(1)
    base = synth.render_mrz_lines([LAST_SIX_NON_FILLER], char_h=32)
    ref = canonical.ink_extent(base)
    ref_w = ref[1] - ref[0] + 1
    for _ in range(100):
        out = data.offset_jitter(base, rng)
        ext = canonical.ink_extent(out)
        assert ext is not None and ext[1] - ext[0] + 1 == ref_w  # ink width identical: nothing cut
        dy = out.shape[0] - base.shape[0]
        assert dy <= 2 * data.OFFSET_JITTER_V_PX
        assert out.shape[1] - base.shape[1] <= 2 * round(base.shape[1] * data.OFFSET_JITTER_FRAC)


def test_augmentation_ranges_match_the_measured_failure_modes():
    assert data.GLYPH_HEIGHT_RANGE == (22, 44)
    assert data.OFFSET_JITTER_FRAC == 0.04 and data.OFFSET_JITTER_V_PX == 2


def test_dataset_samples_a_range_of_glyph_sizes_and_keeps_every_character():
    ds = data.SyntheticMrzLineDataset(size=40, base_seed=7, severity_range=(0.0, 0.0))
    widths = set()
    for i in range(40):
        s = ds[i]
        widths.add(s.image.shape[1] // 44)
        ext = canonical.ink_extent(s.image)
        assert ext is not None and ext[1] - ext[0] + 1 >= 43 * 9  # >= 43 pitches at the smallest size
    assert len(widths) > 5


def test_canvas_jitter_preserves_shape_and_stays_small():
    rng = random.Random(3)
    canvas = canonical.normalize_line(synth.render_mrz_lines([LINE2], char_h=32))
    for _ in range(30):
        out = data.jitter_canvas(canvas, rng)
        assert out.shape == canvas.shape
        a, b = canonical.ink_extent(canvas), canonical.ink_extent(out)
        assert abs(a[0] - b[0]) <= data.CANVAS_SHIFT_JITTER_PX + 25 and abs(a[1] - b[1]) <= data.CANVAS_SHIFT_JITTER_PX + 25


def test_collate_canvas_jitter_is_off_by_default_and_matches_normalize():
    rng = random.Random(5)
    img = synth.render_mrz_lines([LINE2], char_h=32)
    sample = data.MrzSample(image=img, text=LINE2)
    plain = data.collate_batch([sample])["images"][0, 0].numpy()
    assert np.array_equal(np.round(plain * 255).astype(np.uint8), canonical.normalize_line(img))
    jittered = data.collate_batch([sample], canvas_jitter=True)["images"][0, 0].numpy()
    assert jittered.shape == plain.shape


def test_gen_dataset_rejects_the_invalid_seed_and_summary_is_versioned():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "gen_mrz_dataset.py"
    spec_ = importlib.util.spec_from_file_location("gen_mrz_dataset_t", path)
    assert spec_ is not None and spec_.loader is not None
    mod = importlib.util.module_from_spec(spec_)
    import sys

    sys.modules["gen_mrz_dataset_t"] = mod
    spec_.loader.exec_module(mod)
    assert mod.LEGACY_FREE_SEED != 42 and mod.RENDER_VERSION == 2
    src = path.read_text(encoding="utf-8")
    assert '"render_version": RENDER_VERSION' in src and "seed == 42" in src
    ktrain = (path.parent / "kaggle_train.py").read_text(encoding="utf-8")
    assert 'info.get("render_version") == RENDER_VERSION' in ktrain and 'info.get("seed") == args.seed' in ktrain
