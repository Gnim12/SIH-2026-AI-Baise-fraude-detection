"""Milestone B1c: the pipeline runs the real MRZ model or reports it unavailable.

A tiny untrained MrzCRNN is exported to a tmp models dir (session-scoped) so
these tests do not depend on scratch/ or on real weights; its accuracy is
noise, which is fine -- the tests are about wiring.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.api.schemas import MrzCharRole, StageState
from app.config import settings
from app.ocr.mrz import canonical, decode, detect, infer, runtime, spec, synth
from app.ocr.mrz.infer import MRZ_PIN_NAME, MRZReader, MrzReadResult
from app.ocr.mrz.model import MrzCRNN
from app.pipeline.registry import StageId
from app.pipeline.runner import StageContext
from app.pipeline.stages.gate1 import Gate1Stage
from app.pipeline.stages.mrz_read import MrzReadStage

BACKEND_ROOT = Path(__file__).resolve().parents[2]
ICAO_L1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
ICAO_L2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
VERSION = "9.9.9-test"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def exported_models(tmp_path_factory) -> Path:
    """A models dir holding a real (untrained) ONNX export + verified manifest."""
    script = BACKEND_ROOT / "scripts" / "export_mrz_onnx.py"
    spec_ = importlib.util.spec_from_file_location("export_mrz_onnx_b1c", script)
    assert spec_ is not None and spec_.loader is not None
    export_mod = importlib.util.module_from_spec(spec_)
    sys.modules["export_mrz_onnx_b1c"] = export_mod
    spec_.loader.exec_module(export_mod)

    root = tmp_path_factory.mktemp("b1c")
    torch.manual_seed(1)
    ckpt = root / "model.pt"
    torch.save({"state_dict": MrzCRNN(line_len=44).state_dict(), "line_len": 44, "char_accuracy": 0.0}, ckpt)
    models_dir = root / "models"
    models_dir.mkdir()
    (models_dir / "manifest.json").write_text(json.dumps({
        "mrz_crnn": {
            "version": VERSION, "path": f"models/mrz_crnn/{VERSION}/model.onnx",
            "placeholder": True, "sha256": "0" * 64,
        }
    }))
    export_mod.export(
        checkpoint_path=ckpt, version=VERSION, models_dir=models_dir, training_config={}, extra_metrics={},
    )
    return models_dir


@pytest.fixture()
def models_dir(exported_models: Path, tmp_path: Path) -> Path:
    """A private copy per test, so tests can corrupt/alter it."""
    dest = tmp_path / "models"
    shutil.copytree(exported_models, dest)
    return dest


@pytest.fixture()
def use_manifest(monkeypatch):
    def _use(models: Path) -> None:
        monkeypatch.setattr(settings, "manifest_path", models / "manifest.json")
        runtime.reset_mrz_runtime()
    return _use


def _page(lines: list[str], seed: int = 3) -> np.ndarray:
    rec_rng = random.Random(seed)
    img = synth.degrade(synth.render_mrz_lines(lines, char_h=48), rec_rng, severity=0.2)
    page_h, page_w = img.shape[0] * 6, img.shape[1] + 80
    page = np.full((page_h, page_w), 235, dtype=np.uint8)
    page[page_h - img.shape[0] - 20 : page_h - 20, 20 : 20 + img.shape[1]] = img
    return np.stack([page] * 3, axis=-1)


class CraftedReader(MRZReader):
    """Non-stub reader that decodes hand-built log-probabilities, so a test
    can dictate exactly what the network 'saw'. Test-only, like all stubs."""

    def __init__(self, lp_lines: list[np.ndarray], version: str = VERSION) -> None:
        super().__init__(require_trained_weights=False)
        self.stub_mode = False
        self._lp = lp_lines
        self._v = version

    @property
    def version(self) -> str:
        return self._v

    def read(self, image, *, stub_ground_truth=None, mrz_format=spec.MrzFormat.TD3) -> MrzReadResult:
        band = detect.find_mrz(image, n_lines=2)
        if band is None:
            return MrzReadResult(status=decode.DecodeStatus.UNRECOVERABLE, parsed=None)
        return self._to_result(decode.decode_mrz(self._lp, mrz_format), band)


async def _run_stage(reader: MRZReader, image: Optional[np.ndarray] = None):
    runtime.install_reader_for_tests(reader)
    ctx = StageContext(run_id="b1c", inputs={"document_image": image if image is not None else _page([ICAO_L1, ICAO_L2])})
    return await MrzReadStage().run(ctx)


def _lp(line: str, *, conf: float = 0.97, seed: int = 1, errors: Optional[dict[int, str]] = None) -> np.ndarray:
    return decode.fake_logprobs(line, confidence=conf, seed=seed, error_positions=errors)


# ---------------------------------------------------------------------------
# 1. real weights required; unavailable states
# ---------------------------------------------------------------------------

def test_pipeline_loader_requires_trained_weights(models_dir, monkeypatch):
    seen: list[dict] = []
    real_cls = runtime.MRZReader

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real_cls(*args, **kwargs)

    monkeypatch.setattr(runtime, "MRZReader", spy)
    rt = runtime.load_mrz_runtime(models_dir / "manifest.json")
    assert rt.status.state == "live"
    assert seen and seen[0]["require_trained_weights"] is True


@pytest.mark.real_mrz
async def test_placeholder_model_yields_unavailable_never_passed_or_failed(use_manifest, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "manifest.json").write_text(json.dumps({
        "mrz_crnn": {"version": "1.3.0", "path": "models/mrz_crnn/1.3.0/model.onnx",
                     "placeholder": True, "sha256": "0" * 64}
    }))
    use_manifest(models)
    assert runtime.get_mrz_runtime().status.state == "unavailable"

    result = await MrzReadStage().run(StageContext(run_id="x", inputs={"document_image": _page([ICAO_L1, ICAO_L2])}))
    assert result.state == StageState.UNAVAILABLE
    assert result.state not in (StageState.PASSED, StageState.FAILED)
    assert not result.signals  # nothing ran, nothing to attribute a signal to

    gate = await Gate1Stage().run(StageContext(
        run_id="x", artefacts={StageId.MRZ_READ: result.artefacts, StageId.VIZ_READ: {"viz_fields": []}},
    ))
    assert gate.state == StageState.UNAVAILABLE


@pytest.mark.real_mrz
def test_missing_and_mismatched_weights_are_unavailable(models_dir):
    onnx = next(models_dir.rglob("model.onnx"))
    manifest = models_dir / "manifest.json"

    assert runtime.load_mrz_runtime(manifest).status.state == "live"

    onnx.write_bytes(onnx.read_bytes() + b"tamper")
    mismatched = runtime.load_mrz_runtime(manifest).status
    assert mismatched.state == "unavailable" and "SHA-256" in (mismatched.reason or "")

    onnx.unlink()
    missing = runtime.load_mrz_runtime(manifest).status
    assert missing.state == "unavailable"
    assert runtime.load_mrz_runtime(models_dir / "nope.json").status.state == "unavailable"


@pytest.mark.real_mrz
def test_app_boots_with_weights_missing_and_reports_node_unavailable():
    # The repo manifest's mrz_crnn entry is a placeholder: nothing to load.
    runtime.reset_mrz_runtime()
    from app.main import app

    with TestClient(app) as client:
        body = client.get("/api/health").json()
    mrz = next(n for n in body["nodes"] if n["node"] == "mrz")
    assert mrz["status"] == "unavailable"
    assert mrz["modelPin"] is None
    assert "unavailable" in mrz["detail"]


@pytest.mark.real_mrz
def test_health_live_on_verified_model_and_reports_pin_and_version(models_dir, use_manifest):
    use_manifest(models_dir)
    from app.main import app

    with TestClient(app) as client:
        body = client.get("/api/health").json()
    mrz = next(n for n in body["nodes"] if n["node"] == "mrz")
    assert mrz["status"] == "live"
    assert mrz["modelPin"] == f"{MRZ_PIN_NAME} {VERSION}"
    assert mrz["modelVersion"] == VERSION


@pytest.mark.real_mrz
def test_health_never_live_on_placeholder(models_dir, use_manifest):
    manifest = models_dir / "manifest.json"
    data = json.loads(manifest.read_text())
    data["mrz_crnn"]["placeholder"] = True
    manifest.write_text(json.dumps(data))
    use_manifest(models_dir)
    from app.main import app

    with TestClient(app) as client:
        body = client.get("/api/health").json()
    mrz = next(n for n in body["nodes"] if n["node"] == "mrz")
    assert mrz["status"] == "unavailable" and mrz["modelPin"] is None


def test_stub_mode_unreachable_from_app_code():
    offenders_stub, offenders_install = [], []
    for path in (BACKEND_ROOT / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(BACKEND_ROOT).as_posix()
        # code (not docs) constructing a stub reader
        if re.search(r"MRZReader\([^)]*require_trained_weights=False", text) and rel != "app/ocr/mrz/infer.py":
            offenders_stub.append(rel)
        if "install_reader_for_tests(" in text and rel != "app/ocr/mrz/runtime.py":
            offenders_install.append(rel)
    assert not offenders_stub and not offenders_install
    # and the runtime loader itself never builds a stub reader
    assert "require_trained_weights=False" not in (BACKEND_ROOT / "app/ocr/mrz/runtime.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. pin
# ---------------------------------------------------------------------------

@pytest.mark.real_mrz
async def test_pin_is_mrz_crnn_slot_with_version_from_metadata_and_on_every_signal(models_dir, use_manifest):
    use_manifest(models_dir)
    rt = runtime.get_mrz_runtime()
    assert rt.reader is not None
    metadata = json.loads(next(models_dir.rglob("metadata.json")).read_text())
    assert rt.reader.pin == f"mrz-crnn-slot {metadata['version']}" == f"mrz-crnn-slot {VERSION}"
    assert "ctc" not in rt.reader.pin.lower()

    # An untrained model produces noise: low confidence + failed checks, i.e.
    # several signals to inspect.
    result = await MrzReadStage().run(StageContext(run_id="p", inputs={"document_image": _page([ICAO_L1, ICAO_L2])}))
    assert result.signals
    assert all(s.model_pin == rt.reader.pin for s in result.signals)
    ribbon = result.artefacts.get("mrz_schema")
    if ribbon is not None:
        assert ribbon.model_pin == rt.reader.pin


@pytest.mark.real_mrz
def test_version_is_read_from_metadata_not_hardcoded(models_dir, use_manifest):
    meta = next(models_dir.rglob("metadata.json"))
    data = json.loads(meta.read_text())
    data["version"] = "7.7.7-from-metadata"
    meta.write_text(json.dumps(data))
    use_manifest(models_dir)
    assert runtime.get_mrz_runtime().status.pin == "mrz-crnn-slot 7.7.7-from-metadata"


# ---------------------------------------------------------------------------
# 3. recoveredCharacters
# ---------------------------------------------------------------------------

def test_decode_populates_recovered_when_verified_differs_from_greedy():
    # 0/O confusion in the expiry field: the network favours 'O', the checksum wants '0'.
    d = decode.decode_mrz(
        [_lp(ICAO_L1, seed=1), _lp(ICAO_L2, conf=0.6, seed=3, errors={23: "O"})], spec.MrzFormat.TD3,
    )
    assert d.groups["expiry_date"].status is decode.DecodeStatus.VERIFIED
    assert len(d.recovered) == 1
    rc = d.recovered[0]
    assert (rc.line, rc.position, rc.raw, rc.recovered) == (1, 23, "O", "0")
    assert 0.0 < rc.raw_confidence <= 1.0 and 0.0 < rc.recovered_confidence <= 1.0
    assert d.greedy_lines[1][23] == "O" and d.lines[1][23] == "0"


async def test_recovered_reaches_the_ribbon_and_wire_shape():
    lp = [_lp(ICAO_L1, seed=1), _lp(ICAO_L2, conf=0.6, seed=3, errors={23: "O"})]
    result = await _run_stage(CraftedReader(lp))
    ribbon = result.artefacts["mrz_schema"]
    assert len(ribbon.recovered_characters) == 1
    wire = ribbon.model_dump(mode="json", by_alias=True)["recoveredCharacters"][0]
    assert wire["lineIndex"] == 1 and wire["position"] == 23
    assert wire["raw"] == "O" and wire["recovered"] == "0"
    assert {"rawConfidence", "recoveredConfidence"} <= wire.keys()
    tinted = [c for line in ribbon.lines for c in line if c.role == MrzCharRole.RECOVERED]
    assert [c.index for c in tinted] == [23]


async def test_no_recovery_emits_empty_recovered_characters_and_no_tint():
    lp = [_lp(ICAO_L1, seed=1), _lp(ICAO_L2, seed=2)]
    d = decode.decode_mrz(lp, spec.MrzFormat.TD3)
    assert d.recovered == []
    result = await _run_stage(CraftedReader(lp))
    ribbon = result.artefacts["mrz_schema"]
    assert ribbon.recovered_characters == []
    assert not any(c.role == MrzCharRole.RECOVERED for line in ribbon.lines for c in line)


# ---------------------------------------------------------------------------
# 4. honest signals
# ---------------------------------------------------------------------------

async def test_low_mean_confidence_emits_signal_and_high_does_not():
    low = await _run_stage(CraftedReader([_lp(ICAO_L1, conf=0.3), _lp(ICAO_L2, conf=0.3, seed=2)]))
    assert "MRZ_LOW_CONFIDENCE" in [s.signal_id for s in low.signals]
    high = await _run_stage(CraftedReader([_lp(ICAO_L1, conf=0.97), _lp(ICAO_L2, conf=0.97, seed=2)]))
    assert "MRZ_LOW_CONFIDENCE" not in [s.signal_id for s in high.signals]
    assert high.state == StageState.PASSED


def test_low_confidence_threshold_lives_in_thresholds_not_inline():
    from app.pipeline.thresholds import DEFAULT_THRESHOLDS

    assert 0.0 < DEFAULT_THRESHOLDS.mrz_low_confidence_mean < 1.0
    stage_src = (BACKEND_ROOT / "app/pipeline/stages/mrz_read.py").read_text(encoding="utf-8")
    assert "DEFAULT_THRESHOLDS.mrz_low_confidence_mean" in stage_src
    assert "LOW_CONFIDENCE_THRESHOLD" not in stage_src


async def test_unreadable_field_is_absent_with_signal_never_guessed():
    # An unrelated high-confidence misread in the birth-date span cannot be
    # recovered: birth_date must be absent, and the reason must be signalled.
    lp = [_lp(ICAO_L1, seed=1), _lp(ICAO_L2, conf=0.9, seed=4, errors={14: "5"})]
    result = await _run_stage(CraftedReader(lp))
    ribbon = result.artefacts["mrz_schema"]
    assert result.state == StageState.FAILED
    assert "birth_date" not in {f.field for f in ribbon.fields}
    parsed_result = result.artefacts["mrz_read_result"]
    assert "birth_date" not in parsed_result.fields
    codes = [s.signal_id for s in result.signals]
    assert "MRZ_CHECKSUM_FAIL_LINE2" in codes
    assert "MRZ_DECODE_UNRECOVERABLE" in codes
    # fields that verified are still reported
    assert "doc_number" in parsed_result.fields


async def test_unrecoverable_emits_its_signal_and_does_not_crash_the_run():
    from app.pipeline.analyse import run_analysis

    lp = [_lp(ICAO_L1, seed=1), _lp(ICAO_L2, conf=0.9, seed=4, errors={14: "5"})]
    runtime.install_reader_for_tests(CraftedReader(lp))
    result = await run_analysis("b1c-unrec", _page([ICAO_L1, ICAO_L2]))
    assert result.session_id == "b1c-unrec"
    codes = {s.signals[0].signal_id for s in result.findings if s.signals}
    assert codes  # findings were still built, run completed
    all_codes = {sig.signal_id for f in result.findings for sig in f.signals}
    assert "MRZ_DECODE_UNRECOVERABLE" in all_codes


async def test_checksum_fail_composite_code():
    # Corrupt only a composite-covered field the per-field groups don't beam-search (personal number).
    bad_l2 = ICAO_L2[:28] + "Q" + ICAO_L2[29:]
    result = await _run_stage(CraftedReader([_lp(ICAO_L1, seed=1), _lp(bad_l2, seed=2)]))
    assert "MRZ_CHECKSUM_FAIL_COMPOSITE" in [s.signal_id for s in result.signals]


async def test_band_not_found_signal():
    blank = np.full((200, 400, 3), 255, dtype=np.uint8)
    result = await _run_stage(CraftedReader([_lp(ICAO_L1), _lp(ICAO_L2)]), blank)
    assert result.state == StageState.FAILED
    assert "MRZ_BAND_NOT_FOUND" in [s.signal_id for s in result.signals]


# ---------------------------------------------------------------------------
# 5. capture path: detect -> normalize_line -> ONNX
# ---------------------------------------------------------------------------

@pytest.mark.real_mrz
async def test_band_found_but_split_failed_is_band_not_found_with_detail(models_dir, use_manifest, monkeypatch):
    use_manifest(models_dir)
    monkeypatch.setattr(detect, "_split_lines", lambda band, n: [band[:0, :]] * n)  # zero-height lines
    result = await MrzReadStage().run(StageContext(run_id="s", inputs={"document_image": _page([ICAO_L1, ICAO_L2])}))
    assert result.state == StageState.FAILED
    sig = next(s for s in result.signals if s.signal_id == "MRZ_BAND_NOT_FOUND")
    assert "could not be split" in sig.detail and "band located" in sig.detail
    assert sig.model_pin.startswith(MRZ_PIN_NAME)


@pytest.mark.real_mrz
def test_pipeline_uses_normalize_line_and_canonical_geometry(models_dir, use_manifest, monkeypatch):
    use_manifest(models_dir)
    reader = runtime.get_mrz_runtime().reader
    assert reader is not None

    assert infer.normalize_line is canonical.normalize_line
    calls: list[tuple[int, ...]] = []
    real = canonical.normalize_line

    def spy(img: np.ndarray) -> np.ndarray:
        out = real(img)
        calls.append(out.shape)
        return out

    monkeypatch.setattr(infer, "normalize_line", spy)
    fed: list[tuple[int, ...]] = []
    real_run = reader._model.session.run  # type: ignore[union-attr]

    class SessionSpy:
        def run(self, names, feed):
            fed.append(feed["images"].shape)
            return real_run(names, feed)

        def __getattr__(self, name):
            return getattr(reader._model.session, name)  # type: ignore[union-attr]

    monkeypatch.setattr(reader._model, "session", SessionSpy())  # type: ignore[arg-type]
    result = reader.read(_page([ICAO_L1, ICAO_L2]))
    assert result.band is not None
    assert calls == [(canonical.CANONICAL_HEIGHT, canonical.CANONICAL_WIDTH)] * 2
    assert fed == [(1, 1, canonical.CANONICAL_HEIGHT, canonical.CANONICAL_WIDTH)] * 2


def test_no_divergent_resize_in_inference_path():
    src = (BACKEND_ROOT / "app/ocr/mrz/infer.py").read_text(encoding="utf-8")
    assert "cv2.resize" not in src and ".resize(" not in src


@pytest.mark.real_mrz
def test_smoke_export_end_to_end_if_present(monkeypatch):
    smoke = BACKEND_ROOT / "scratch" / "smoke_models"
    if not (smoke / "manifest.json").exists():
        pytest.skip("scratch/smoke_models not present")
    rt = runtime.load_mrz_runtime(smoke / "manifest.json")
    assert rt.status.state == "live"
    assert rt.status.pin == "mrz-crnn-slot 1.4.0-smoke"
    assert rt.reader is not None
    result = rt.reader.read(_page([ICAO_L1, ICAO_L2]))
    assert result.band is not None and result.decoded is not None
    assert hashlib.sha256(Path(smoke / "mrz_crnn/1.4.0-smoke/model.onnx").read_bytes()).hexdigest()
