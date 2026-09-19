"""Milestone B1g: the trained MRZ weights are installed, verified, and pinned.

Uses the real files under models/mrz_crnn/1.4.0/ (gitignored *.onnx: the
tests that need the weights skip when they are not present on this machine).
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import random
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.ocr.mrz import runtime, synth
from app.ocr.mrz.infer import MRZ_PIN_NAME, ModelIntegrityError, MRZReader

BACKEND_ROOT = Path(__file__).resolve().parents[2]
MODELS = BACKEND_ROOT / "models"
INSTALLED = MODELS / "mrz_crnn" / "1.4.0"
needs_weights = pytest.mark.skipif(not (INSTALLED / "model.onnx").exists(), reason="trained weights not installed")


def _manifest_entry() -> dict:
    return json.loads((MODELS / "manifest.json").read_text())["mrz_crnn"]


@pytest.fixture()
def fresh_runtime(monkeypatch):
    monkeypatch.setattr(runtime, "_runtime", None)
    yield
    runtime.reset_mrz_runtime()


@needs_weights
def test_manifest_hash_matches_installed_file_and_metadata():
    entry = _manifest_entry()
    actual = hashlib.sha256((INSTALLED / "model.onnx").read_bytes()).hexdigest()
    assert entry["placeholder"] is False
    assert entry["version"] == "1.4.0" and entry["path"] == "models/mrz_crnn/1.4.0/model.onnx"
    assert entry["sha256"] == actual == json.loads((INSTALLED / "metadata.json").read_text())["sha256"]


def test_other_manifest_entries_survive_the_merge():
    manifest = json.loads((MODELS / "manifest.json").read_text())
    assert {"arcface", "scrfd", "rapidocr_det", "rapidocr_rec", "rapidocr_cls", "tamper_unet", "pad_minifas"} <= set(manifest)


@needs_weights
def test_pin_version_comes_from_metadata_not_a_constant(tmp_path, monkeypatch, fresh_runtime):
    from app.storage import b2_repositories as repo

    src = (BACKEND_ROOT / "app/storage/b2_repositories.py").read_text(encoding="utf-8")
    assert '"mrz-crnn-slot": "' not in src and "1.2.0" not in src

    live = runtime.get_mrz_runtime()
    metadata = json.loads((INSTALLED / "metadata.json").read_text())
    assert live.status.state == "live"
    assert repo.current_model_pins()[MRZ_PIN_NAME] == metadata["version"] == live.status.version

    # Change only metadata.json in a copy: the pin must follow it.
    root = tmp_path / "models"
    (root / "mrz_crnn" / "1.4.0").mkdir(parents=True)
    shutil.copy(INSTALLED / "model.onnx", root / "mrz_crnn/1.4.0/model.onnx")
    (root / "manifest.json").write_text(json.dumps({"mrz_crnn": _manifest_entry()}))
    (root / "mrz_crnn/1.4.0/metadata.json").write_text(json.dumps({**metadata, "version": "8.8.8-meta"}))
    monkeypatch.setattr(settings, "manifest_path", root / "manifest.json")
    runtime.reset_mrz_runtime()
    assert repo.current_model_pins()[MRZ_PIN_NAME] == "8.8.8-meta"


def test_pin_is_unavailable_never_a_stale_version_when_model_not_live(monkeypatch):
    from app.storage import b2_repositories as repo

    monkeypatch.setattr(runtime, "_runtime", runtime._unavailable("test: not loaded"))
    assert repo.current_model_pins()[MRZ_PIN_NAME] == "unavailable"


@needs_weights
def test_health_reports_mrz_live_with_installed_version(fresh_runtime):
    from app.main import app

    with TestClient(app) as client:
        body = client.get("/api/health").json()
    mrz = next(n for n in body["nodes"] if n["node"] == "mrz")
    assert mrz["status"] == "live"
    assert mrz["modelPin"] == "mrz-crnn-slot 1.4.0" and mrz["modelVersion"] == "1.4.0"


@needs_weights
def test_corrupted_model_file_raises_on_load(tmp_path):
    root = tmp_path / "models"
    (root / "mrz_crnn/1.4.0").mkdir(parents=True)
    shutil.copy(MODELS / "manifest.json", root / "manifest.json")
    weights = root / "mrz_crnn/1.4.0/model.onnx"
    shutil.copy(INSTALLED / "model.onnx", weights)
    shutil.copy(INSTALLED / "metadata.json", weights.parent / "metadata.json")
    MRZReader(weights)  # intact: loads
    weights.write_bytes(weights.read_bytes() + b"\x00")
    with pytest.raises(ModelIntegrityError, match="SHA-256 mismatch"):
        MRZReader(weights)
    assert runtime.load_mrz_runtime(root / "manifest.json").status.state == "unavailable"


def _page(band: np.ndarray) -> np.ndarray:
    h, w = band.shape
    page = np.full((h * 6, w + 80), 235, dtype=np.uint8)
    page[h * 6 - h - 20 : h * 6 - 20, 20 : 20 + w] = band
    return np.stack([page] * 3, axis=-1)


@needs_weights
def test_twelve_synthetic_documents_produce_signals_from_the_live_model(fresh_runtime):
    """Provenance, not accuracy: every one of the 12 documents goes through the
    real ONNX model and comes out with recoveryStats and MRZ signals pinned to
    1.4.0. (Whether the readings are *right* is measured by measure_recovery.py;
    see the B1g report for what the page path currently does to them.)"""
    from app.pipeline.analyse import run_analysis

    async def run_all():
        out = []
        for i, sev in enumerate(np.linspace(0, 1, 12)):
            rng = random.Random(9000 + i)
            _, img = synth.generate_synthetic_page(rng, severity=float(sev))
            out.append(await run_analysis(f"b1g-{i}", _page(img)))
        return out

    results = asyncio.run(run_all())
    assert len(results) == 12
    for res in results:
        assert res.mrz is not None and res.mrz.model_pin == "mrz-crnn-slot 1.4.0"
        stats = res.mrz.recovery_stats
        assert stats is not None and stats.total_characters == 88
        codes = {s.signal_id for f in res.findings for s in f.signals}
        assert any(c.startswith("MRZ_") for c in codes)


def _load_script(name: str):
    path = BACKEND_ROOT / "scripts" / f"{name}.py"
    spec_ = importlib.util.spec_from_file_location(name, path)
    assert spec_ is not None and spec_.loader is not None
    mod = importlib.util.module_from_spec(spec_)
    sys.modules[name] = mod
    spec_.loader.exec_module(mod)
    return mod


@needs_weights
def test_measure_recovery_runs_and_writes_report(tmp_path):
    mod = _load_script("measure_recovery")
    out = tmp_path / "report.json"
    args = Namespace(records=40, seed=123, confusable_bias=0.5, weights=str(INSTALLED / "model.onnx"), out=str(out))
    report = mod.measure(args)
    out.write_text(json.dumps(report))
    assert json.loads(out.read_text())["records"] == 40
    a, s = report["all_characters"], report["checksum_guarded_span"]
    assert a["chars"] == 40 * 88 and s["chars"] == 40 * 24
    assert a["corrected_chars"] == s["corrected_chars"] and a["harmed_chars"] == s["harmed_chars"]
    assert "0" in report["per_character_guarded_span"]
    assert "8" in report["per_character_guarded_span"] and "2" in report["per_character_guarded_span"]
    assert "records" in mod.summarise(report) or "model" in mod.summarise(report)


def test_measure_recovery_refuses_training_seed_and_small_sets():
    mod = _load_script("measure_recovery")
    with pytest.raises(SystemExit):
        mod.measure(Namespace(records=2000, seed=mod.TRAIN_SEED, confusable_bias=0.5, weights="x", out="y"))
