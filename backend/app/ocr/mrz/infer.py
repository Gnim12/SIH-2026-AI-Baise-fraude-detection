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

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import decode, detect, spec

logger = logging.getLogger(__name__)


class MissingWeightsError(RuntimeError):
    pass


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
        self._model = None

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

    def _load_model(self, weights_path: Path):
        raise NotImplementedError(
            "real CRNN checkpoint loading is not implemented yet -- "
            f"no trained weights exist to load from {weights_path} (see BACKEND_BRIEF.md §1.3)"
        )

    def read(
        self,
        image: np.ndarray,
        *,
        stub_ground_truth: Optional[list[str]] = None,
        mrz_format: spec.MrzFormat = spec.MrzFormat.TD3,
    ) -> MrzReadResult:
        band = detect.find_mrz(image, n_lines=spec.LINE_SHAPE[mrz_format][0])
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
        raise NotImplementedError(
            "real CRNN forward pass is not implemented yet -- no trained weights (see BACKEND_BRIEF.md §1.3)"
        )

    def _to_result(self, decoded: decode.MrzDecodeResult, band: detect.MrzBand) -> MrzReadResult:
        parsed = decoded.parsed
        fields: dict[str, MrzField] = {}

        def conf_for(group_name: str) -> float:
            group = decoded.groups.get(group_name)
            if group is None:
                return 1.0
            return 1.0 if group.status is decode.DecodeStatus.VERIFIED else 0.3

        fields["doc_number"] = MrzField("doc_number", parsed.doc_number, conf_for("doc_number"))
        fields["surname"] = MrzField("surname", parsed.surname, 0.9)
        fields["given_names"] = MrzField("given_names", parsed.given_names, 0.9)
        fields["nationality"] = MrzField("nationality", parsed.nationality, 0.9)
        fields["sex"] = MrzField("sex", parsed.sex, 0.9)
        fields["birth_date"] = MrzField(
            "birth_date", parsed.birth_date.isoformat() if parsed.birth_date else parsed.birth_date_raw,
            conf_for("birth_date"),
        )
        fields["expiry_date"] = MrzField(
            "expiry_date", parsed.expiry_date.isoformat() if parsed.expiry_date else parsed.expiry_date_raw,
            conf_for("expiry_date"),
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
