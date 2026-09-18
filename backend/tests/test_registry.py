"""BACKEND_BRIEF.md §1.5: the app must verify every model hash at startup and
refuse to boot on mismatch. Tested by deliberately corrupting a file."""
from __future__ import annotations

import json

import pytest

from app.registry import ModelRegistryError, verify_all


def test_verify_all_passes_against_the_real_manifest():
    versions = verify_all()
    assert "rapidocr_det" in versions
    assert "mrz_crnn" in versions  # placeholder entry, still hash-verified


def test_verify_all_refuses_to_boot_on_corrupted_model_file(tmp_path):
    # Build an isolated manifest + model dir so this test never touches the
    # real models/ directory other tests and app startup depend on.
    # verify_all resolves manifest paths relative to models_dir.parent (the
    # project root), so lay this out the same way: <tmp>/models/model.onnx.
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / "model.onnx"
    model_path.write_bytes(b"genuine weights, honest")
    import hashlib

    real_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "some_model": {"version": "1.0", "path": "models/model.onnx", "sha256": real_hash},
    }))

    from app.config import settings
    original_models_dir = settings.models_dir
    settings.models_dir = models_dir
    try:
        # Sanity: verification passes before corruption.
        assert verify_all(manifest_path)["some_model"] == "1.0"

        # Corrupt the file in place -- a silently swapped model.
        model_path.write_bytes(b"a swapped, malicious weight file")

        with pytest.raises(ModelRegistryError, match="hash mismatch"):
            verify_all(manifest_path)
    finally:
        settings.models_dir = original_models_dir


def test_verify_all_refuses_to_boot_on_missing_model_file(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "ghost_model": {"version": "1.0", "path": "models/does_not_exist.onnx", "sha256": "deadbeef"},
    }))

    from app.config import settings
    original_models_dir = settings.models_dir
    settings.models_dir = models_dir
    try:
        with pytest.raises(ModelRegistryError, match="does not exist"):
            verify_all(manifest_path)
    finally:
        settings.models_dir = original_models_dir
