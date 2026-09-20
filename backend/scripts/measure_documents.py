"""Per-document readout of the 12 synthetic documents through the full pipeline (real ONNX model).

For each document: character error of the final MRZ text vs ground truth, whether Gate 1 passed,
recoveryStats, and every MRZ signal that fired. Run once per model version; the version's manifest
entry is swapped in on the fly so both can be compared without touching models/manifest.json.

    python scripts/measure_documents.py --version 1.5.0 --out scripts/documents_report_1.5.0.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.api.schemas import StageState  # noqa: E402
from app.config import settings  # noqa: E402
from app.ocr.mrz import runtime, synth  # noqa: E402
from app.pipeline.analyse import default_stages  # noqa: E402
from app.pipeline.registry import StageId  # noqa: E402
from app.pipeline.runner import StageContext, run_dag  # noqa: E402

N_DOCS = 12
_TMP = tempfile.TemporaryDirectory()


def _page(band: np.ndarray) -> np.ndarray:
    h, w = band.shape
    page = np.full((h * 6, w + 80), 235, dtype=np.uint8)
    page[h * 6 - h - 20 : h * 6 - 20, 20 : 20 + w] = band
    return np.stack([page] * 3, axis=-1)


def use_version(version: str) -> None:
    """Point the MRZ runtime at models/mrz_crnn/<version>/ through a temporary manifest."""
    # MRZReader verifies against the models/manifest.json beside the weights, so build a throwaway tree.
    src = BACKEND_ROOT / "models" / "mrz_crnn" / version
    root = Path(_TMP.name) / "models"
    shutil.copytree(src, root / "mrz_crnn" / version)
    manifest = json.loads((BACKEND_ROOT / "models" / "manifest.json").read_text())
    manifest["mrz_crnn"].update(
        version=version, path=f"models/mrz_crnn/{version}/model.onnx",
        sha256=hashlib.sha256((src / "model.onnx").read_bytes()).hexdigest())
    (root / "manifest.json").write_text(json.dumps(manifest))
    settings.manifest_path = root / "manifest.json"
    runtime.reset_mrz_runtime()
    assert runtime.get_mrz_runtime().status.version == version


async def one(i: int, severity: float) -> dict:
    rng = random.Random(9000 + i)
    rec, img = synth.generate_synthetic_page(rng, severity=severity)
    ctx = StageContext(run_id=f"doc-{i}", inputs={"document_image": _page(img), "document_id": "document",
                                                  "mrz_ground_truth": None}, timeout_s=60.0)
    out = await run_dag(default_stages(), ctx)
    mrz = out.stages[StageId.MRZ_READ].artefacts.get("mrz_schema")
    gate1 = out.stages[StageId.GATE_1]
    read = ["".join(c.char for c in line) for line in mrz.lines] if mrz else []
    wrong = sum(a != b for r, t in zip(read, rec.lines) for a, b in zip(r, t)) + \
        sum(len(t) for t in rec.lines[len(read):])
    stats = mrz.recovery_stats if mrz else None
    return {
        "doc": i, "severity": round(severity, 3),
        "chars_wrong": wrong, "cer": round(wrong / 88, 4), "lines_exact": [r == t for r, t in zip(read, rec.lines)],
        "gate1_passed": gate1.state == StageState.PASSED, "gate1_state": gate1.state.value,
        "recovery_stats": stats.model_dump(by_alias=True) if stats else None,
        "signals": sorted(s.signal_id for s in out.all_signals if s.signal_id.startswith("MRZ_")),
        "model_pin": mrz.model_pin if mrz else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    use_version(args.version)

    async def run_all():
        return [await one(i, float(s)) for i, s in enumerate(np.linspace(0, 1, N_DOCS))]

    docs = asyncio.run(run_all())
    n_pass = sum(d["gate1_passed"] for d in docs)
    composite = sum("MRZ_CHECKSUM_FAIL_COMPOSITE" in d["signals"] for d in docs)
    report = {"version": args.version, "documents": docs, "gate1_pass": f"{n_pass}/{N_DOCS}",
              "composite_fail_fired": f"{composite}/{N_DOCS}",
              "mean_cer": round(sum(d["cer"] for d in docs) / N_DOCS, 4)}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"model {args.version}: Gate 1 {n_pass}/{N_DOCS}  COMPOSITE fired {composite}/{N_DOCS}  mean CER {report['mean_cer']:.2%}")
    for d in docs:
        rs = d["recovery_stats"] or {}
        print(f" doc{d['doc']:>2} sev {d['severity']:.2f} CER {d['cer']:.1%} ({d['chars_wrong']:>2}/88) "
              f"gate1={'PASS' if d['gate1_passed'] else 'fail'} recovered {rs.get('recovered')} suspect {rs.get('suspect')} "
              f"{','.join(d['signals']) or '-'}")


if __name__ == "__main__":
    main()
