"""Model pin registry, single source of truth for every signal's `modelPin`.

Deviation from the B1 layout (`app/config/pins.py`, `app/config/thresholds.py`
as a package): `app/config.py` already exists as a module in this repo
(pydantic-settings `Settings`), so a sibling `app/config/` package would
collide with it. Pins and thresholds live under `app/pipeline/` instead;
flagged in the milestone report rather than silently renamed on either side.
"""
from __future__ import annotations

from app.ocr.mrz.runtime import get_mrz_runtime
from app.registry import ModelRegistryError, load_manifest

# Wave 2's four stages (tamper/ovd-sweep/face-verify/identity-graph) are
# `unavailable` in B1 -- they have no pin because nothing runs. The manifest
# already marks their weight files "placeholder": true (see
# models/manifest.json); this set is the pipeline-facing mirror of that.
UNAVAILABLE_MODEL_KEYS = frozenset({"tamper_unet", "scrfd", "arcface", "pad_minifas"})

WAVE1_MODEL_KEYS = {"mrz_crnn": "mrz_crnn", "rapidocr_det": "rapidocr_det", "rapidocr_rec": "rapidocr_rec"}


def resolve_pins() -> dict[str, str]:
    """{model_key: 'name version'} for every model this build can actually
    run, sourced from models/manifest.json (never hardcoded in a stage)."""
    try:
        manifest = load_manifest()
    except ModelRegistryError:
        return {}
    pins: dict[str, str] = {}
    for key, entry in manifest.items():
        if key == "mrz_crnn":
            continue  # pin comes from the model that actually loaded, below
        version = entry.get("version", "unknown")
        pins[key] = f"{key} {version}"
    # "mrz-crnn-slot <version from metadata.json>", and only while the model
    # is live: an unavailable model emitted nothing, so has no pin to record.
    mrz_pin = get_mrz_runtime().status.pin
    if mrz_pin is not None:
        pins["mrz_crnn"] = mrz_pin
    return pins
