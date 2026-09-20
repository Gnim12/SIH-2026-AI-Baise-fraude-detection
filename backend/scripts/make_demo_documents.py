"""DEMO-ONLY: build full synthetic documents (VIZ zone + MRZ band) and push each through the live HTTP API.

The MRZ bands are the same pixels the 12-document measurement used (seed 9000+i, severity linspace 0..1),
so the two reports line up. Two extra documents carry a deliberately altered VIZ field (a MRZ/VIZ
disagreement, the one tampering-style finding the current build can genuinely produce).

    python scripts/make_demo_documents.py --out ../demo_docs --api http://localhost:8001
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.ocr.mrz import synth  # noqa: E402
from tests.ocr.conftest import VIZ_HEIGHT, VIZ_WIDTH, render_viz_zone  # noqa: E402


def stack(viz: np.ndarray, band: np.ndarray) -> np.ndarray:
    page_w = max(VIZ_WIDTH, band.shape[1] + 80)
    page_h = VIZ_HEIGHT + band.shape[0] + 60
    page = np.full((page_h, page_w), 255, dtype=np.uint8)
    page[0:VIZ_HEIGHT, 0:VIZ_WIDTH] = viz
    y = page_h - band.shape[0] - 20
    page[y : y + band.shape[0], 20 : 20 + band.shape[1]] = band
    return page


def build(i: int, severity: float, **viz_overrides):
    rng = random.Random(9000 + i)
    rec, band = synth.generate_synthetic_page(rng, severity=severity)
    return rec, stack(render_viz_zone(rec, **viz_overrides), band)


def analyse(api: str, path: Path, truth: list[str]) -> dict:
    with httpx.Client(base_url=api, timeout=120) as c:
        sid = c.post("/api/sessions").json()["sessionId"]
        c.post(f"/api/sessions/{sid}/artefacts", data={"kind": "document-still"},
               files={"file": (path.name, path.read_bytes(), "image/png")}).raise_for_status()
        c.post(f"/api/sessions/{sid}/analyse").raise_for_status()
        for _ in range(120):
            r = c.get(f"/api/sessions/{sid}/result")
            if r.status_code == 200:
                break
            time.sleep(1)
        body = r.json()
    mrz = body.get("mrz") or {}
    read = ["".join(ch["char"] for ch in line) for line in mrz.get("lines", [])]
    wrong = sum(a != b for x, t in zip(read, truth) for a, b in zip(x, t))
    codes = sorted({s["signalId"] for f in body.get("findings", []) for s in f.get("signals", [])})
    return {"file": path.name, "sessionId": sid, "mrz_chars_wrong": wrong, "verdict": body.get("verdict"),
            "reason": body.get("reason"), "signals": codes,
            "finding_titles": [f.get("title") for f in body.get("findings", [])],
            "recovery_stats": mrz.get("recoveryStats"),
            "cross_check": [(x["field"], x["status"]) for x in body.get("crossCheck", [])]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--api", required=True)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, sev in enumerate(np.linspace(0, 1, 12)):
        rec, page = build(i, float(sev))
        jobs.append((f"doc{i:02d}_sev{sev:.2f}.png", page, rec.lines))
    rec, page = build(0, 0.0, given_names_display="XAVIER MORGAN")
    jobs.append(("tamper_A_altered_given_names.png", page, rec.lines))
    rec, page = build(0, 0.0, birth_display_override="01/01/1970")
    jobs.append(("tamper_B_altered_birth_date.png", page, rec.lines))
    rec, page = build(0, 0.0, doc_number_override="ZZ9999999")
    jobs.append(("tamper_C_altered_doc_number.png", page, rec.lines))
    results = []
    for name, page, truth in jobs:
        path = out / name
        Image.fromarray(page).save(path)
        res = analyse(args.api, path, truth)
        results.append(res)
        print(f"{name:40s} wrong {res['mrz_chars_wrong']:>2}/88  verdict={res['verdict']}  "
              f"{','.join(s for s in res['signals'] if not s.startswith('MRZ_VIZ_FIELD_UNREADABLE')) or '-'}")
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
