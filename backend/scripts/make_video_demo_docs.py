"""DEMO-ONLY: three clean synthetic passports (VIZ zone + MRZ band) plus one whose VIZ surname is altered.

    python scripts/make_video_demo_docs.py --out scratch/demo_docs --api http://localhost:8000
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

from PIL import Image

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(BACKEND_ROOT / "scripts"))

from app.ocr.mrz import synth  # noqa: E402
from make_demo_documents import analyse, stack  # noqa: E402
from tests.ocr.conftest import render_viz_zone  # noqa: E402

PEOPLE = {
    "demo_clean_1_okonkwo": dict(surname="OKONKWO", given_names="AMARA GRACE", doc_number="P48213975", nationality="UTO",
                                 issuing_country="UTO", birth_raw="900315", expiry_raw="330612", sex="F"),
    "demo_clean_2_lindqvist": dict(surname="LINDQVIST", given_names="ERIK JOHAN", doc_number="P70581342", nationality="UTO",
                                   issuing_country="UTO", birth_raw="850922", expiry_raw="311104", sex="M"),
    "demo_clean_3_tanaka": dict(surname="TANAKA", given_names="YUKI", doc_number="P25907116", nationality="UTO",
                                issuing_country="UTO", birth_raw="980207", expiry_raw="340830", sex="F"),
}


def page(fields: dict, viz_surname: str | None = None):
    rec = synth.build_td3_record(**fields)
    band = synth.degrade(synth.render_mrz_lines(rec.lines), random.Random(9000), severity=0.0)
    viz_rec = dataclasses.replace(rec, surname=viz_surname) if viz_surname else rec
    return rec, stack(render_viz_zone(viz_rec), band)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--api", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, fields in PEOPLE.items():
        rec, img = page(fields)
        jobs.append((name, rec, img))
    rec, img = page(PEOPLE["demo_clean_1_okonkwo"], viz_surname="OKONKWU")
    jobs.append(("demo_ALTERED_surname", rec, img))
    results = []
    for name, rec, img in jobs:
        path = out / f"{name}.png"
        Image.fromarray(img).save(path)
        res = analyse(a.api, path, rec.lines)
        res["mrz_surname"] = rec.surname
        results.append(res)
        print(name, res["verdict"], "wrong=", res["mrz_chars_wrong"], res["signals"])
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
