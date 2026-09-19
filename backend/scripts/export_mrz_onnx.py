#!/usr/bin/env python
"""Export a trained MrzCRNN .pt checkpoint to ONNX for infer.py.

Run: python scripts/export_mrz_onnx.py --checkpoint models/mrz_crnn/model.pt \
    --version 1.4.0 --training-config '{"epochs": 30, "batch": 64}'

Produces:
  models/mrz_crnn/{version}/model.onnx
  models/mrz_crnn/{version}/metadata.json
and updates models/manifest.json's mrz_crnn entry (sha256, placeholder=false).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr.mrz.canonical import CANONICAL_HEIGHT, CANONICAL_WIDTH  # noqa: E402
from app.ocr.mrz.model import MrzCRNN, NUM_CLASSES, _ConvBackbone  # noqa: E402

INPUT_NAME = "images"
OUTPUT_NAME = "log_probs"
TARGET_HEIGHT = CANONICAL_HEIGHT
OPSET = 17


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_exportable(line_len: int) -> None:
    """The legacy TorchScript exporter only lowers AdaptiveAvgPool1d when the
    backbone's output length T is an exact multiple of `line_len`. Check that
    CANONICAL_WIDTH satisfies it for this checkpoint's line_len, and fail with
    the reason rather than an opaque exporter error."""
    backbone = _ConvBackbone()
    backbone.eval()
    with torch.no_grad():
        t = backbone(torch.randn(1, 1, TARGET_HEIGHT, CANONICAL_WIDTH)).shape[3]
    if t % line_len != 0:
        raise RuntimeError(
            f"CANONICAL_WIDTH={CANONICAL_WIDTH} gives backbone output length T={t}, "
            f"which is not a multiple of line_len={line_len}; the ONNX export of "
            "AdaptiveAvgPool1d needs T % line_len == 0. Choose a canonical width "
            "that satisfies it for every supported line length, or export a "
            "separate model per format."
        )


def export(
    *, checkpoint_path: Path, version: str, models_dir: Path,
    training_config: dict[str, Any], extra_metrics: dict[str, Any],
) -> Path:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    line_len = ckpt.get("line_len", 44)

    model = MrzCRNN(line_len=line_len)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out_dir = models_dir / "mrz_crnn" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    # Width is exported FIXED at CANONICAL_WIDTH (app/ocr/mrz/canonical.py),
    # not dynamic. In eager PyTorch AdaptiveAvgPool1d accepts any width, but
    # the legacy TorchScript exporter can only lower it to a plain AveragePool,
    # which needs the backbone output length T to be a static multiple of
    # line_len. infer.py applies the same normalize_line() used in training
    # (uniform scale + pad, never crop), so the fixed width costs nothing.
    _check_exportable(line_len)
    width = CANONICAL_WIDTH
    dummy = torch.randn(1, 1, TARGET_HEIGHT, width)
    export_kwargs: dict[str, Any] = dict(
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=OPSET,
    )
    try:
        # dynamo=False selects the legacy TorchScript exporter, which needs no
        # `onnxscript` dependency; older torch (e.g. Kaggle images) predates the
        # kwarg and only has that exporter anyway.
        torch.onnx.export(model, (dummy,), str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, (dummy,), str(onnx_path), **export_kwargs)

    # Verify ONNX Runtime output matches PyTorch on fresh samples, all at the
    # one fixed width the exported graph accepts (batch is still exercised).
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    max_abs_diff = 0.0
    n_check = 20
    latencies = []
    for i in range(n_check):
        sample = rng.standard_normal((1, 1, TARGET_HEIGHT, width)).astype(np.float32)
        with torch.no_grad():
            torch_out = model(torch.from_numpy(sample)).numpy()

        t0 = time.perf_counter()
        onnx_out = session.run([OUTPUT_NAME], {INPUT_NAME: sample})[0]
        latencies.append(time.perf_counter() - t0)

        max_abs_diff = max(max_abs_diff, float(np.max(np.abs(torch_out - onnx_out))))

    assert max_abs_diff < 1e-4, f"ONNX output diverges from PyTorch by {max_abs_diff} (>1e-4)"

    sha256 = _sha256(onnx_path)
    metadata = {
        "version": version,
        "line_len": line_len,
        "num_classes": NUM_CLASSES,
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "input_height": TARGET_HEIGHT,
        "input_width": width,
        "opset": OPSET,
        "sha256": sha256,
        "training_config": training_config,
        "metrics": {
            **{k: ckpt.get(k) for k in ("char_accuracy", "line_accuracy", "confusable_accuracy", "epoch")},
            **extra_metrics,
        },
        "onnx_vs_pytorch_max_abs_diff": max_abs_diff,
        "cpu_latency_ms_per_crop_mean": float(np.mean(latencies) * 1000),
        "cpu_latency_ms_per_crop_p95": float(np.percentile(latencies, 95) * 1000),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    _update_manifest(models_dir, version, onnx_path, sha256)

    print(f"exported {onnx_path}")
    print(f"onnx vs pytorch max abs diff: {max_abs_diff:.2e}")
    print(f"cpu latency: {metadata['cpu_latency_ms_per_crop_mean']:.2f}ms mean, "
          f"{metadata['cpu_latency_ms_per_crop_p95']:.2f}ms p95")
    return onnx_path


def _update_manifest(models_dir: Path, version: str, onnx_path: Path, sha256: str) -> None:
    manifest_path = models_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["mrz_crnn"]

    entry["path"] = str(onnx_path.relative_to(models_dir.parent)).replace("\\", "/")
    entry["sha256"] = sha256
    entry["version"] = version
    entry["placeholder"] = False
    entry.pop("note", None)

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"updated {manifest_path}: placeholder=false, sha256={sha256[:12]}...")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="models/mrz_crnn/model.pt")
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--training-config", type=str, default="{}", help="JSON string")
    parser.add_argument("--extra-metrics", type=str, default="{}", help="JSON string")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export(
        checkpoint_path=PROJECT_ROOT / args.checkpoint,
        version=args.version,
        models_dir=PROJECT_ROOT / args.models_dir,
        training_config=json.loads(args.training_config),
        extra_metrics=json.loads(args.extra_metrics),
    )
