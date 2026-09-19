"""B1f: no wrong architecture name in backend/app, and the suspect-recovery guard."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.api.schemas import StageState
from app.ocr.mrz import spec
from app.pipeline.registry import REGISTRY_BY_ID, StageId
from app.pipeline.thresholds import DEFAULT_THRESHOLDS
from tests.pipeline.test_b1c_mrz_real_weights import ICAO_L1, ICAO_L2, CraftedReader, _lp, _run_stage

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CHARSET = spec.MRZ_CHARSET


def _line2(raw_p: float, rec_p: float) -> np.ndarray:
    """Line 2 where slot 23 (a '0' in the expiry date) is read as 'O' with
    probability raw_p and '0' with rec_p; the checksum must pick '0'."""
    lp = _lp(ICAO_L2, seed=2)
    row = np.full(len(CHARSET), 1e-4)
    row[CHARSET.index("O")] = raw_p
    row[CHARSET.index("0")] = rec_p
    lp[23] = np.log(row / row.sum())
    return lp


async def _run(raw_p: float, rec_p: float):
    return await _run_stage(CraftedReader([_lp(ICAO_L1, seed=1), _line2(raw_p, rec_p)]))


def test_no_ctc_architecture_name_in_backend_app():
    needles = ["crnn-ctc", "crnn_ctc"]
    hits = [
        str(p.relative_to(BACKEND_ROOT))
        for p in (BACKEND_ROOT / "app").rglob("*.py")
        if any(n in p.read_text(encoding="utf-8").lower() for n in needles)
    ]
    assert hits == []


def test_registry_label_matches_frontend_and_model_pins_name():
    assert REGISTRY_BY_ID[StageId.MRZ_READ].label == "MRZ read · fixed-slot CRNN"
    from app.storage.b2_repositories import MODEL_PINS

    assert "mrz-crnn-slot" in MODEL_PINS and not any("ctc" in k for k in MODEL_PINS)


def test_margin_lives_in_thresholds():
    assert DEFAULT_THRESHOLDS.mrz_recovery_confidence_margin == 0.30
    src = (BACKEND_ROOT / "app/pipeline/stages/mrz_read.py").read_text(encoding="utf-8")
    assert "DEFAULT_THRESHOLDS.mrz_recovery_confidence_margin" in src


async def test_gap_above_threshold_is_suspect_and_still_reported():
    result = await _run(0.70, 0.28)
    ribbon = result.artefacts["mrz_schema"]
    assert len(ribbon.recovered_characters) == 1  # not suppressed
    rc = ribbon.recovered_characters[0]
    assert (rc.raw, rc.recovered, rc.suspect) == ("O", "0", True)
    assert ribbon.model_dump(mode="json", by_alias=True)["recoveredCharacters"][0]["suspect"] is True
    sig = next(s for s in result.signals if s.signal_id == "MRZ_RECOVERY_SUSPECT")
    assert "line 2 position 23" in sig.detail
    assert result.state == StageState.PASSED  # flagged, not failed or hidden


async def test_gap_within_threshold_is_not_suspect_and_emits_no_signal():
    result = await _run(0.45, 0.40)
    rc = result.artefacts["mrz_schema"].recovered_characters
    assert len(rc) == 1 and rc[0].suspect is False
    assert "MRZ_RECOVERY_SUSPECT" not in [s.signal_id for s in result.signals]


async def test_statistics_match_the_recovered_array():
    result = await _run(0.70, 0.28)
    ribbon = result.artefacts["mrz_schema"]
    stats = ribbon.recovery_stats
    rcs = ribbon.recovered_characters
    assert stats is not None
    assert stats.total_characters == 88
    assert stats.recovered == len(rcs) and stats.suspect == sum(r.suspect for r in rcs)
    assert stats.mean_raw_confidence == sum(r.raw_confidence for r in rcs) / len(rcs)
    assert stats.mean_recovered_confidence == sum(r.recovered_confidence for r in rcs) / len(rcs)
    wire = ribbon.model_dump(mode="json", by_alias=True)["recoveryStats"]
    assert {"totalCharacters", "recovered", "suspect", "meanRawConfidence", "meanRecoveredConfidence"} <= wire.keys()


async def test_statistics_with_no_recovery_have_null_means_not_zero():
    result = await _run_stage(CraftedReader([_lp(ICAO_L1, seed=1), _lp(ICAO_L2, seed=2)]))
    stats = result.artefacts["mrz_schema"].recovery_stats
    assert stats is not None
    assert (stats.total_characters, stats.recovered, stats.suspect) == (88, 0, 0)
    assert stats.mean_raw_confidence is None and stats.mean_recovered_confidence is None
