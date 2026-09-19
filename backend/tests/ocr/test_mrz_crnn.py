"""M-PREP acceptance tests: fixed-slot head, OCR-B font, confusable biasing,
pre-generated dataset, ONNX export, and infer.py's registry-checked loading.
"""
from __future__ import annotations

import json
import random
import sys

import numpy as np
import pytest
import torch

from app.ocr.mrz import data, decode, spec, synth
from app.ocr.mrz.model import NUM_CLASSES, MrzCRNN

# ---------------------------------------------------------------------------
# model.py: fixed-slot head
# ---------------------------------------------------------------------------

def test_model_output_shape_td3():
    model = MrzCRNN()
    out = model(torch.randn(3, 1, 32, 640))
    assert out.shape == (3, 44, 37)


def test_model_has_no_ctc_blank():
    assert not hasattr(MrzCRNN, "BLANK_IDX")
    import app.ocr.mrz.model as model_mod

    assert not hasattr(model_mod, "BLANK_IDX")
    assert NUM_CLASSES == len(spec.MRZ_CHARSET) == 37


@pytest.mark.parametrize("line_len", [36, 30])
def test_model_alternate_line_lengths(line_len):
    model = MrzCRNN(line_len=line_len)
    out = model(torch.randn(2, 1, 32, line_len * 16))
    assert out.shape == (2, line_len, 37)


def test_cross_entropy_against_integer_targets():
    model = MrzCRNN()
    out = model(torch.randn(2, 1, 32, 640))  # (B, 44, 37), log-probabilities
    targets = torch.randint(0, 37, (2, 44))
    loss = torch.nn.functional.cross_entropy(out.reshape(-1, 37), targets.reshape(-1))
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# data.py: fixed-length targets, jitter augmentation
# ---------------------------------------------------------------------------

def test_encode_text_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        data.encode_text("TOOSHORT", 44)


def test_encode_text_matches_length():
    text = "P" + "<" * 43
    encoded = data.encode_text(text, 44)
    assert encoded.shape == (44,)


def test_horizontal_jitter_stays_within_bound():
    # Background fill is 255 (white, matching real MRZ crops), so the marker
    # must use a distinct value or it's indistinguishable from the padding
    # horizontal_jitter introduces at the exposed edge.
    rng = random.Random(11)
    w = 700
    image = np.full((32, w), 255, dtype=np.uint8)
    image[:, w // 2] = 0  # single marker column at the centre
    max_frac = 0.03
    max_px = int(round(w * max_frac))

    for _ in range(200):
        shifted = data.horizontal_jitter(image, rng, max_frac=max_frac)
        assert shifted.shape == image.shape
        marker_cols = np.where(shifted[16] == 0)[0]
        assert marker_cols.size == 1
        delta = int(marker_cols[0]) - w // 2
        assert abs(delta) <= max_px


# ---------------------------------------------------------------------------
# synth.py: confusable-character biasing
# ---------------------------------------------------------------------------

def _confusable_rate(records) -> float:
    text = "".join(r.surname + r.given_names + r.personal_number for r in records)
    text = text.replace(" ", "").replace("<", "")
    if not text:
        return 0.0
    confusable = set("0O1I5S8B2Z")
    return sum(1 for c in text if c in confusable) / len(text)


def test_confusable_bias_all_records_self_validate():
    rng = random.Random(123)
    records = [synth.generate_record(rng, confusable_bias=0.6) for _ in range(200)]
    for record in records:
        result = spec.parse_td3(record.lines)
        assert result.all_valid, record.lines


def test_confusable_bias_raises_frequency():
    rng = random.Random(321)
    unbiased = [synth.generate_record(rng, confusable_bias=0.0) for _ in range(200)]
    biased = [synth.generate_record(rng, confusable_bias=0.6) for _ in range(200)]
    assert _confusable_rate(biased) > _confusable_rate(unbiased)


def _doc_number_confusable_rate(records) -> float:
    text = "".join(r.doc_number for r in records)
    confusable = set("0O1I5S8B2Z")
    return sum(1 for c in text if c in confusable) / len(text)


def test_doc_number_bias_raises_rate_and_records_still_validate():
    rng = random.Random(2024)
    unbiased = [synth.generate_record(rng, confusable_bias=0.0) for _ in range(200)]
    biased = [synth.generate_record(rng, confusable_bias=0.6) for _ in range(200)]

    for record in biased:
        assert spec.parse_td3(record.lines).all_valid, record.lines

    base, raised = _doc_number_confusable_rate(unbiased), _doc_number_confusable_rate(biased)
    # Uniform over 36 alphanumerics -> ~0.278; weight 1.6 on 10 of them -> ~0.381.
    assert raised > base + 0.05, (base, raised)


def test_bias_leaves_dates_and_country_codes_unbiased():
    rng = random.Random(9)
    records = [synth.generate_record(rng, confusable_bias=0.9) for _ in range(300)]
    for record in records:
        assert record.birth_date_raw.isdigit() and len(record.birth_date_raw) == 6
        assert record.expiry_date_raw.isdigit() and len(record.expiry_date_raw) == 6
        assert record.nationality in synth._COUNTRY_POOL


# ---------------------------------------------------------------------------
# canonical geometry: width guard, never-crop normalisation
# ---------------------------------------------------------------------------

def _ink_runs(image) -> int:
    """Number of separate dark column-runs in a line image."""
    dark = (image < 128).any(axis=0)
    return int(np.count_nonzero(dark[1:] & ~dark[:-1]) + (1 if dark[0] else 0))


@pytest.mark.parametrize("width", [500, 700, 900, 1400])
def test_normalize_keeps_all_44_characters(width):
    """A 44-cell band of any width must come out of normalisation with all 44
    characters intact -- one ink block per cell, none cropped away."""
    from app.ocr.mrz.canonical import CANONICAL_HEIGHT, CANONICAL_WIDTH, normalize_line

    cell = width // 44
    image = np.full((28, cell * 44), 255, dtype=np.uint8)
    for i in range(44):
        x0 = i * cell + cell // 4
        image[4:24, x0 : x0 + max(2, cell // 2)] = 0

    assert _ink_runs(image) == 44
    out = normalize_line(image)

    assert out.shape == (CANONICAL_HEIGHT, CANONICAL_WIDTH)
    assert _ink_runs(out) == 44, f"characters lost normalising a {width}px band"
    assert (out < 128).any(axis=0)[0] == False and (out < 128).any(axis=0)[-1] == False  # noqa: E712


def test_normalize_preserves_character_grid():
    """Uniform scale: the spacing between character centres stays uniform."""
    from app.ocr.mrz.canonical import normalize_line

    cell = 20
    image = np.full((28, cell * 44), 255, dtype=np.uint8)
    for i in range(44):
        image[4:24, i * cell + 5 : i * cell + 15] = 0
    out = normalize_line(image)
    dark = (out < 128).any(axis=0)
    starts = np.flatnonzero(dark[1:] & ~dark[:-1])
    gaps = np.diff(starts)
    assert gaps.max() - gaps.min() <= 2, gaps


def test_infer_normalises_wide_band_without_losing_characters(exported_reader_setup):
    """End to end: a 900px-wide band goes through MRZReader._run_model and
    still yields one (line_len, 37) matrix per line."""
    from app.ocr.mrz import detect
    from app.ocr.mrz.infer import MRZReader

    _models_dir, onnx_path = exported_reader_setup
    reader = MRZReader(weights_path=onnx_path)
    wide = np.full((30, 900), 255, dtype=np.uint8)
    band = detect.MrzBand(bbox=(0, 0, 900, 60), region=(0.0, 0.0, 1.0, 1.0), angle_deg=0.0, lines=[wide, wide])
    out = reader._run_model(band)
    assert [o.shape for o in out] == [(44, 37), (44, 37)]


def test_onnx_width_guard_rejects_mismatched_canonical_width(exported_reader_setup, monkeypatch):
    from app.ocr.mrz import infer
    from app.ocr.mrz.infer import ModelIntegrityError, MRZReader

    _models_dir, onnx_path = exported_reader_setup
    monkeypatch.setattr(infer, "CANONICAL_WIDTH", 704)
    with pytest.raises(ModelIntegrityError, match="expects input"):
        MRZReader(weights_path=onnx_path)


def test_data_export_and_infer_share_one_canonical_width():
    import importlib.util
    from pathlib import Path

    from app.ocr.mrz import canonical, infer

    assert data.CANONICAL_WIDTH is canonical.CANONICAL_WIDTH or data.CANONICAL_WIDTH == 700
    assert infer.CANONICAL_WIDTH == canonical.CANONICAL_WIDTH == 700
    script = Path(__file__).resolve().parents[2] / "scripts" / "export_mrz_onnx.py"
    spec_ = importlib.util.spec_from_file_location("export_mrz_onnx_w", script)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    assert mod.CANONICAL_WIDTH == canonical.CANONICAL_WIDTH


# ---------------------------------------------------------------------------
# scripts/gen_mrz_dataset.py: manifest integrity + split disjointness
# ---------------------------------------------------------------------------

def _import_gen_script():
    import importlib.util
    from pathlib import Path

    backend_root = Path(__file__).resolve().parents[2]
    script_path = backend_root / "scripts" / "gen_mrz_dataset.py"
    spec_ = importlib.util.spec_from_file_location("gen_mrz_dataset", script_path)
    module = importlib.util.module_from_spec(spec_)
    sys.modules["gen_mrz_dataset"] = module
    spec_.loader.exec_module(module)
    return module


def test_dataset_manifest_resolves_and_splits_are_disjoint(tmp_path):
    gen = _import_gen_script()
    gen._worker_init()

    out_dir = tmp_path / "mrz_dataset_test"
    all_keys = {}
    offset = 0
    for split_name, n in (("train", 12), ("val", 4), ("test", 4)):
        rows, keys = gen.generate_split(
            split_name=split_name, n_records=n, record_seed_offset=offset,
            base_seed=99, confusable_bias=0.3, out_dir=out_dir, workers=1,
        )
        offset += n
        all_keys[split_name] = keys

        split_dir = out_dir / split_name
        manifest_rows = [json.loads(line) for line in (split_dir / "manifest.jsonl").open()]
        assert len(manifest_rows) == 2 * n
        for row in manifest_rows:
            assert (split_dir / row["filename"]).exists()

    assert not (all_keys["train"] & all_keys["val"])
    assert not (all_keys["train"] & all_keys["test"])
    assert not (all_keys["val"] & all_keys["test"])


# ---------------------------------------------------------------------------
# ONNX export: PyTorch vs ONNX Runtime parity
# ---------------------------------------------------------------------------

def test_onnx_export_matches_pytorch(tmp_path):
    import importlib.util
    from pathlib import Path

    backend_root = Path(__file__).resolve().parents[2]
    script_path = backend_root / "scripts" / "export_mrz_onnx.py"
    spec_ = importlib.util.spec_from_file_location("export_mrz_onnx", script_path)
    export_mod = importlib.util.module_from_spec(spec_)
    sys.modules["export_mrz_onnx"] = export_mod
    spec_.loader.exec_module(export_mod)

    torch.manual_seed(0)
    model = MrzCRNN(line_len=44)
    ckpt_path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "line_len": 44, "char_accuracy": 0.0}, ckpt_path)

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    manifest = {"mrz_crnn": {"version": "0.0.0", "path": "", "placeholder": True, "sha256": "0" * 64}}
    (models_dir / "manifest.json").write_text(json.dumps(manifest))

    onnx_path = export_mod.export(
        checkpoint_path=ckpt_path, version="0.0.1-test", models_dir=models_dir,
        training_config={}, extra_metrics={},
    )
    assert onnx_path.exists()
    metadata = json.loads((onnx_path.parent / "metadata.json").read_text())
    assert metadata["onnx_vs_pytorch_max_abs_diff"] < 1e-4


# ---------------------------------------------------------------------------
# infer.py: registry-checked ONNX loading, decode_mrz round-trip
# ---------------------------------------------------------------------------

@pytest.fixture()
def exported_reader_setup(tmp_path):
    """Exports a small MrzCRNN to a tmp models/ dir with a matching manifest,
    mirroring what scripts/export_mrz_onnx.py produces in production."""
    import importlib.util
    from pathlib import Path

    backend_root = Path(__file__).resolve().parents[2]
    script_path = backend_root / "scripts" / "export_mrz_onnx.py"
    spec_ = importlib.util.spec_from_file_location("export_mrz_onnx_fixture", script_path)
    export_mod = importlib.util.module_from_spec(spec_)
    sys.modules["export_mrz_onnx_fixture"] = export_mod
    spec_.loader.exec_module(export_mod)

    torch.manual_seed(1)
    model = MrzCRNN(line_len=44)
    ckpt_path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "line_len": 44, "char_accuracy": 0.0}, ckpt_path)

    models_dir = tmp_path / "models"
    manifest = {
        "mrz_crnn": {
            "version": "1.3.0", "path": "models/mrz_crnn/1.3.0/model.onnx",
            "placeholder": True, "sha256": "0" * 64,
        }
    }
    models_dir.mkdir(parents=True)
    (models_dir / "manifest.json").write_text(json.dumps(manifest))

    onnx_path = export_mod.export(
        checkpoint_path=ckpt_path, version="1.3.0", models_dir=models_dir,
        training_config={}, extra_metrics={},
    )
    return models_dir, onnx_path


def test_infer_raises_on_missing_file(tmp_path):
    from app.ocr.mrz.infer import MissingWeightsError, MRZReader

    with pytest.raises(MissingWeightsError):
        MRZReader(weights_path=tmp_path / "does_not_exist.onnx")


def test_infer_raises_on_hash_mismatch(tmp_path):
    from app.ocr.mrz.infer import ModelIntegrityError, MRZReader

    models_dir = tmp_path / "models"
    (models_dir / "mrz_crnn" / "1.3.0").mkdir(parents=True)
    weights_path = models_dir / "mrz_crnn" / "1.3.0" / "model.onnx"
    weights_path.write_bytes(b"not a real onnx file")
    manifest = {
        "mrz_crnn": {
            "version": "1.3.0", "path": "models/mrz_crnn/1.3.0/model.onnx",
            "placeholder": False, "sha256": "f" * 64,
        }
    }
    (models_dir / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ModelIntegrityError, match="SHA-256 mismatch"):
        MRZReader(weights_path=weights_path)


def test_infer_raises_on_placeholder_still_in_place():
    """Exercised against the real, still-unplaced-holder repo state: the
    checked-in models/mrz_crnn/1.3.0/model.onnx is a placeholder and must be
    refused, not silently loaded."""
    from pathlib import Path

    from app.ocr.mrz.infer import ModelIntegrityError, MRZReader

    backend_root = Path(__file__).resolve().parents[2]
    real_weights = backend_root / "models" / "mrz_crnn" / "1.3.0" / "model.onnx"
    if not real_weights.exists():
        pytest.skip("repo placeholder weights file not present")

    with pytest.raises(ModelIntegrityError, match="placeholder"):
        MRZReader(weights_path=real_weights)


def test_infer_output_feeds_decode_mrz_without_shape_errors(exported_reader_setup):
    from app.ocr.mrz.infer import MRZReader

    _models_dir, onnx_path = exported_reader_setup
    reader = MRZReader(weights_path=onnx_path)
    assert not reader.stub_mode

    rng = random.Random(2024)
    record = synth.generate_record(rng)
    clean = synth.render_mrz_lines(record.lines)
    degraded = synth.degrade(clean, rng, severity=0.2)

    page_h, page_w = degraded.shape[0] * 6, degraded.shape[1] + 80
    page = np.full((page_h, page_w), 235, dtype=np.uint8)
    y_off, x_off = page_h - degraded.shape[0] - 20, 20
    page[y_off : y_off + degraded.shape[0], x_off : x_off + degraded.shape[1]] = degraded
    import cv2

    page_bgr = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)

    result = reader.read(page_bgr)
    assert result.parsed is not None
    assert isinstance(result.status, decode.DecodeStatus)


def test_stub_ground_truth_unreachable_from_api_path(exported_reader_setup):
    """A real (non-stub) reader must refuse stub_ground_truth outright, not
    silently ignore it -- that's what keeps the test-only fixture from ever
    leaking into a production call path."""
    from app.ocr.mrz.infer import MRZReader

    _models_dir, onnx_path = exported_reader_setup
    reader = MRZReader(weights_path=onnx_path)

    dummy = np.full((200, 400, 3), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="stub_ground_truth"):
        reader.read(dummy, stub_ground_truth=["A" * 44, "A" * 44])
