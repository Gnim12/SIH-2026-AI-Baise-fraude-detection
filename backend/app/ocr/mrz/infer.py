"""MRZReader: image in, fields + signals out.

Ties together detect.py (locate + split the MRZ band), model.py (CRNN
inference, when real weights exist), and decode.py (checksum-constrained
beam search) into one call.

STUB MODE -- read this before touching require_trained_weights
================================================================
The CRNN has no trained weights yet: training needs a GPU, which this
environment doesn't have (BACKEND_BRIEF.md §1.3). Until a real checkpoint
exists, MRZReader can run in "stub mode": instead of a forward pass through
the network, it replays `decode.fake_logprobs` seeded from a ground-truth
string supplied by the caller. This lets every layer above MRZReader --
checksum-constrained decoding, the MRZ<->VIZ cross-check, signal emission --
be fully exercised without a trained model.

This must never happen silently in production:
  - `require_trained_weights` defaults to True. With no `weights_path`, the
    constructor raises `MissingWeightsError` instead of quietly falling back
    to stub mode.
  - Stub mode is reachable only by explicitly passing
    `require_trained_weights=False` *and* `stub_ground_truth=...` (or
    per-call via `read(..., stub_ground_truth=...)`).
  - Every stub-mode init logs a loud warning naming itself as unsafe for
    production so it can't be missed in logs.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from . import decode, detect, spec
from .canonical import CANONICAL_HEIGHT, CANONICAL_WIDTH, normalize_line

logger = logging.getLogger(__name__)

# The audit-record name of this model. The recogniser is a fixed-slot head, not
# CTC; the pin names what actually runs. The version comes from the export's
# metadata.json, never from here.
MRZ_PIN_NAME = "mrz-crnn-slot"

_INPUT_NAME = "images"
_OUTPUT_NAME = "log_probs"


class MissingWeightsError(RuntimeError):
    pass


class ModelIntegrityError(RuntimeError):
    """Raised when the on-disk ONNX weights don't match what the model
    registry (models/manifest.json) says should be running -- a placeholder
    still in place, a missing file, or a hash mismatch. Never caught and
    silently downgraded to stub mode: a swapped or corrupt model in a border
    system is a security incident, not a fallback path."""


@dataclass
class _OnnxModel:
    session: ort.InferenceSession
    line_len: int
    version: str


@dataclass
class MrzField:
    name: str
    value: str
    confidence: float


@dataclass
class MrzSignal:
    code: str
    severity: str
    detail: str
    region: Optional[tuple[float, float, float, float]] = None


@dataclass
class MrzReadResult:
    status: decode.DecodeStatus
    parsed: Optional[spec.MrzResult]
    fields: dict[str, MrzField] = field(default_factory=dict)
    signals: list[MrzSignal] = field(default_factory=list)
    groups: dict[str, decode.GroupDecodeResult] = field(default_factory=dict)
    band: Optional[detect.MrzBand] = None
    stub_mode: bool = False
    # Set when a band was located but could not be turned into lines.
    band_failure_detail: Optional[str] = None
    decoded: Optional[decode.MrzDecodeResult] = None
    # Mean, over every character of the final decoded lines, of the network's
    # probability for that character. None when nothing was decoded.
    mean_confidence: Optional[float] = None

    @property
    def recovered(self) -> list[decode.RecoveredCharacter]:
        return self.decoded.recovered if self.decoded is not None else []


class MRZReader:
    """MRZ band -> checksum-constrained fields + signals.

    Production usage:
        reader = MRZReader(weights_path="models/mrz_crnn/1.3.0/model.onnx")
        result = reader.read(document_image)

    Test usage (explicit opt-in, no trained model required):
        reader = MRZReader(require_trained_weights=False)
        result = reader.read(document_image, stub_ground_truth=[line1, line2])
    """

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        *,
        require_trained_weights: bool = True,
        stub_confidence: float = 0.9,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self.stub_confidence = stub_confidence
        self.stub_mode = False
        self._model: Optional[_OnnxModel] = None

        if self.weights_path is not None and self.weights_path.exists():
            self._model = self._load_model(self.weights_path)
            return

        if require_trained_weights:
            raise MissingWeightsError(
                "MRZReader has no trained weights ("
                f"weights_path={self.weights_path!r}) and require_trained_weights=True "
                "(the production default). Train the CRNN (see train.py) or, for "
                "tests only, pass require_trained_weights=False explicitly."
            )

        self.stub_mode = True
        logger.warning(
            "MRZReader is running in STUB MODE: no trained CRNN weights, decoding "
            "fabricated logprobs from decode.fake_logprobs instead of real inference. "
            "This must never run in production -- it requires "
            "require_trained_weights=False to have been passed explicitly. "
            "If you are seeing this outside a test, something is misconfigured."
        )

    @property
    def version(self) -> str:
        """Model version as recorded in the export's metadata.json."""
        return self._model.version if self._model is not None else "stub"

    @property
    def pin(self) -> str:
        return f"{MRZ_PIN_NAME} {self.version}"

    def _load_model(self, weights_path: Path) -> "_OnnxModel":
        """Verify the on-disk ONNX weights against models/manifest.json, then
        create one onnxruntime.InferenceSession, reused for every subsequent
        `read()` call -- session creation is not cheap and must not happen
        per-request."""
        # weights_path is models/mrz_crnn/{version}/model.onnx; the registry
        # lives two directories up, at models/manifest.json.
        models_dir = weights_path.parents[2]
        manifest_path = models_dir / "manifest.json"
        if not manifest_path.exists():
            raise ModelIntegrityError(f"model registry not found at {manifest_path}")

        manifest = json.loads(manifest_path.read_text())
        entry = manifest.get("mrz_crnn")
        if entry is None:
            raise ModelIntegrityError(f"no 'mrz_crnn' entry in {manifest_path}")

        if entry.get("placeholder", True):
            raise ModelIntegrityError(
                f"models/manifest.json marks mrz_crnn as a placeholder (version="
                f"{entry.get('version')!r}) -- there is no trained model to load. "
                "Train the CRNN (train.py), export it (scripts/export_mrz_onnx.py), "
                "which clears the placeholder flag, before constructing MRZReader "
                "with require_trained_weights=True."
            )

        if not weights_path.exists():
            raise ModelIntegrityError(f"weights file missing: {weights_path}")

        actual_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        expected_sha256 = entry.get("sha256")
        if actual_sha256 != expected_sha256:
            raise ModelIntegrityError(
                f"SHA-256 mismatch for {weights_path}: manifest expects "
                f"{expected_sha256!r}, file hashes to {actual_sha256!r}. Refusing to "
                "load -- a silently swapped model file is a security incident, not "
                "something to load anyway."
            )

        session = ort.InferenceSession(str(weights_path), providers=["CPUExecutionProvider"])

        # Width guard: a dynamic axis reads back as a string/None, a fixed one
        # as an int -- anything other than exactly canonical geometry is refused.
        input_shape = tuple(session.get_inputs()[0].shape)
        if input_shape[2:] != (CANONICAL_HEIGHT, CANONICAL_WIDTH):
            raise ModelIntegrityError(
                f"{weights_path} expects input (height, width)={input_shape[2:]} but this "
                f"code normalises crops to ({CANONICAL_HEIGHT}, {CANONICAL_WIDTH}) "
                "(app/ocr/mrz/canonical.py). The model was exported for a different "
                "geometry; re-export it or fix CANONICAL_WIDTH. Refusing to feed it "
                "mis-sized crops."
            )

        metadata_path = weights_path.parent / "metadata.json"
        if not metadata_path.exists():
            raise ModelIntegrityError(
                f"{metadata_path} is missing: the model version is read from it and cannot be guessed."
            )
        metadata = json.loads(metadata_path.read_text())
        version = metadata.get("version")
        if not isinstance(version, str) or not version:
            raise ModelIntegrityError(f"{metadata_path} has no 'version'.")
        line_len = int(metadata.get("line_len", 44))

        logger.info("MRZReader loaded ONNX weights from %s (sha256=%s...)", weights_path, actual_sha256[:12])
        return _OnnxModel(session=session, line_len=line_len, version=version)

    def read(
        self,
        image: np.ndarray,
        *,
        stub_ground_truth: Optional[list[str]] = None,
        mrz_format: spec.MrzFormat = spec.MrzFormat.TD3,
    ) -> MrzReadResult:
        if stub_ground_truth is not None and not self.stub_mode:
            raise ValueError(
                "stub_ground_truth is only usable when MRZReader was constructed with "
                "require_trained_weights=False (test-only). Passing it against a real "
                "model is refused rather than silently ignored, so a test fixture can "
                "never accidentally leak into a production call path."
            )

        n_lines = spec.LINE_SHAPE[mrz_format][0]
        try:
            band = detect.find_mrz(image, n_lines=n_lines)
        except detect.BandSplitError as exc:
            return MrzReadResult(
                status=decode.DecodeStatus.UNRECOVERABLE,
                parsed=None,
                signals=[MrzSignal(code="MRZ_NOT_FOUND", severity="high", detail=str(exc))],
                stub_mode=self.stub_mode,
                band_failure_detail=str(exc),
            )
        if band is None:
            return MrzReadResult(
                status=decode.DecodeStatus.UNRECOVERABLE,
                parsed=None,
                signals=[
                    MrzSignal(
                        code="MRZ_NOT_FOUND",
                        severity="high",
                        detail="No machine-readable zone could be located on this document.",
                    )
                ],
                stub_mode=self.stub_mode,
            )

        if self.stub_mode:
            if stub_ground_truth is None:
                raise ValueError(
                    "MRZReader is in stub mode and requires stub_ground_truth=[line1, line2, ...] "
                    "to fabricate logprobs from -- there is no real model to run."
                )
            logprobs_lines = [
                decode.fake_logprobs(line, confidence=self.stub_confidence, seed=i)
                for i, line in enumerate(stub_ground_truth)
            ]
        else:
            logprobs_lines = self._run_model(band)

        decoded = decode.decode_mrz(logprobs_lines, mrz_format)
        return self._to_result(decoded, band)

    def _run_model(self, band: detect.MrzBand) -> list[np.ndarray]:
        """Run the loaded ONNX session over each line of `band`, returning
        one (line_len, 37) log-probability array per line -- exactly the
        shape decode.decode_mrz expects, no CTC collapse needed."""
        assert self._model is not None
        results = []
        for line_img in band.lines:
            resized = normalize_line(line_img)
            batch = resized.astype(np.float32)[None, None, :, :] / 255.0
            log_probs = self._model.session.run([_OUTPUT_NAME], {_INPUT_NAME: batch})[0]
            results.append(log_probs[0])  # drop the batch dim -> (line_len, 37)
        return results

    def _to_result(self, decoded: decode.MrzDecodeResult, band: detect.MrzBand) -> MrzReadResult:
        parsed = decoded.parsed
        confs = decoded.char_confidences
        all_confs = [c for line in confs for c in line]
        mean_confidence = float(sum(all_confs) / len(all_confs)) if all_confs else 0.0

        def span_conf(line: int, start: int, end: int) -> float:
            vals = confs[line][start:end]
            return float(sum(vals) / len(vals)) if vals else 0.0

        # Fields guarded by a check digit are only reported when that check
        # digit validated. A field whose check failed is absent -- the
        # signals say why -- never a guess presented as a reading.
        check_ok = {g.name: g.valid for g in parsed.checks}
        fields: dict[str, MrzField] = {}
        if decoded.format is spec.MrzFormat.TD3:
            if check_ok.get("doc_number", False):
                fields["doc_number"] = MrzField("doc_number", parsed.doc_number, span_conf(1, 0, 10))
            fields["surname"] = MrzField("surname", parsed.surname, span_conf(0, 5, 44))
            fields["given_names"] = MrzField("given_names", parsed.given_names, span_conf(0, 5, 44))
            fields["nationality"] = MrzField("nationality", parsed.nationality, span_conf(1, 10, 13))
            fields["sex"] = MrzField("sex", parsed.sex, span_conf(1, 20, 21))
            if check_ok.get("birth_date", False):
                fields["birth_date"] = MrzField(
                    "birth_date", parsed.birth_date.isoformat() if parsed.birth_date else parsed.birth_date_raw,
                    span_conf(1, 13, 20),
                )
            if check_ok.get("expiry_date", False):
                fields["expiry_date"] = MrzField(
                    "expiry_date", parsed.expiry_date.isoformat() if parsed.expiry_date else parsed.expiry_date_raw,
                    span_conf(1, 21, 28),
                )

        signals: list[MrzSignal] = []
        for name, group in decoded.groups.items():
            if group.status is decode.DecodeStatus.UNRECOVERABLE:
                signals.append(
                    MrzSignal(
                        code="MRZ_CHECKDIGIT_UNRECOVERABLE",
                        severity="high",
                        detail=(
                            f"MRZ {name.replace('_', ' ')} check digit does not validate against "
                            f"any hypothesis in the decoder's beam (read as {group.text!r}, "
                            f"check digit {group.check_char!r}); this cannot be explained by an "
                            "ordinary misread and is treated as a tamper signal, not an OCR error."
                        ),
                        region=band.region,
                    )
                )

        return MrzReadResult(
            status=decoded.status,
            parsed=parsed,
            fields=fields,
            signals=signals,
            groups=decoded.groups,
            band=band,
            stub_mode=self.stub_mode,
            decoded=decoded,
            mean_confidence=mean_confidence,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-decoder", action="store_true",
        help="run the stub-mode read() path end to end against a synthetic page, no trained model required",
    )
    args = parser.parse_args()

    if not args.demo_decoder:
        parser.print_help()
        raise SystemExit(0)

    import random

    from . import synth

    # require_trained_weights=True (the default) must refuse to run without weights.
    try:
        MRZReader()
        raise AssertionError("MRZReader() with no weights should have raised MissingWeightsError")
    except MissingWeightsError as exc:
        print(f"infer.py: default constructor correctly refuses stub mode: {exc}")

    reader = MRZReader(require_trained_weights=False)
    assert reader.stub_mode

    rng = random.Random(2024)
    record = synth.generate_record(rng)
    clean = synth.render_mrz_lines(record.lines)
    degraded = synth.degrade(clean, rng, severity=0.3)

    page_h, page_w = degraded.shape[0] * 6, degraded.shape[1] + 80
    page = np.full((page_h, page_w), 235, dtype=np.uint8)
    y_off, x_off = page_h - degraded.shape[0] - 20, 20
    page[y_off : y_off + degraded.shape[0], x_off : x_off + degraded.shape[1]] = degraded
    import cv2

    page_bgr = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)

    result = reader.read(page_bgr, stub_ground_truth=record.lines)
    print(f"infer.py: stub-mode read() -> status={result.status}, "
          f"doc_number={result.fields['doc_number'].value}, "
          f"surname={result.fields['surname'].value}, "
          f"signals={[s.code for s in result.signals]}")
    assert result.status is decode.DecodeStatus.VERIFIED
    assert result.fields["doc_number"].value == record.doc_number
    assert not result.signals

    print("infer.py --demo-decoder self-test OK")
