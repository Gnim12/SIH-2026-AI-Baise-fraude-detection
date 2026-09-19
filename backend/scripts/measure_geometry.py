#!/usr/bin/env python
"""Character error of the MRZ model by input path and glyph height.

Paths (each rendered at glyph heights 24/28/32/36/40 px, severity uniform 0-1):

  crop_training  ground-truth single-line crops, rendered as in training
  band_split     a rendered two-line band split evenly in half
  page_detect    the two-line band pasted on a page, found by detect.find_mrz

Every path goes through canonical.normalize_line (whatever it currently does)
and the installed ONNX model, then decode.decode_mrz (greedy and constrained).
Run once before a geometry change with --label before and once after with
--label after; both land in geometry_report.json.

Run: python scripts/measure_geometry.py --label before [--records 200]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.ocr.mrz import canonical, data, decode, detect, spec, synth  # noqa: E402

TRAIN_SEED = 42
B1G_RECOVERY_SEED = 20260919
DEFAULT_SEED = 777001
HEIGHTS = (24, 28, 32, 36, 40)
PATHS = ("crop_training", "band_split", "page_detect")
BATCH = 64


def embed_on_page(band: np.ndarray) -> np.ndarray:
    h, w = band.shape
    page = np.full((h * 6, w + 80), 235, dtype=np.uint8)
    page[h * 6 - h - 20 : h * 6 - 20, 20 : 20 + w] = band
    return np.stack([page] * 3, axis=-1)


VISIBLE = 39  # characters that fit in the 704 px canvas the 1.4.0 training set was rendered on


def render(lines: list[str], height: int, mode: str) -> np.ndarray:
    """mode 'training': synth.render_mrz_lines as the training set was made (44*16 px wide, so the
    trailing characters of an 18 px-advance font are clipped). 'full': the same font and size on a
    canvas wide enough for all 44 characters, i.e. what a real, complete MRZ line looks like."""
    if mode == "training":
        return synth.render_mrz_lines(lines, char_h=height)
    font = synth._load_mono_font(int(height * 0.8))
    width = int(font.getlength("A" * 44)) + 4
    img = Image.new("L", (width, len(lines) * height), color=255)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((2, i * height + 2), line, fill=0, font=font)
    return np.array(img)


def make_lines(path: str, lines: list[str], height: int, rng: random.Random, mode: str) -> list[np.ndarray] | None:
    """Two line images for one record, or None when the path failed to produce them."""
    sev = rng.uniform(0.0, 1.0)
    if path == "crop_training":
        out = []
        for text in lines:
            img = synth.degrade(render([text], height, mode), rng, severity=sev)
            out.append(data.horizontal_jitter(img, rng, max_frac=0.03))
        return out
    band = synth.degrade(render(lines, height, mode), rng, severity=sev)
    if path == "band_split":
        half = band.shape[0] // 2
        return [band[:half], band[half:]]
    try:
        found = detect.find_mrz(embed_on_page(band))
    except detect.BandSplitError:
        return None
    return list(found.lines) if found is not None else None


def infer(session, images: list[np.ndarray]) -> np.ndarray:
    outs = []
    for i in range(0, len(images), BATCH):
        batch = np.stack([canonical.normalize_line(im) for im in images[i : i + BATCH]])
        outs.append(session.run(["log_probs"], {"images": batch.astype(np.float32)[:, None] / 255.0})[0])
    return np.concatenate(outs)


def evaluate(session, path: str, height: int, records: int, seed: int, mode: str) -> dict:
    rng = random.Random(seed * 1000 + height)
    recs = [synth.generate_record(rng, confusable_bias=0.5) for _ in range(records)]
    per_rec = [make_lines(path, r.lines, height, rng, mode) for r in recs]
    ok = [(r, imgs) for r, imgs in zip(recs, per_rec) if imgs is not None and len(imgs) == 2]
    failed = records - len(ok)
    c = collections.Counter()
    if ok:
        lp = infer(session, [im for _, imgs in ok for im in imgs]).reshape(len(ok), 2, -1, len(spec.MRZ_CHARSET))
        for (rec, _), rec_lp in zip(ok, lp):
            d = decode.decode_mrz([rec_lp[0], rec_lp[1]], spec.MrzFormat.TD3)
            for li, truth in enumerate(rec.lines):
                g, k = d.greedy_lines[li], d.lines[li]
                c["lines"] += 1
                c["exact_greedy"] += g == truth
                c["exact_constrained"] += k == truth
                c["wrong_greedy"] += sum(a != b for a, b in zip(g, truth))
                c["wrong_constrained"] += sum(a != b for a, b in zip(k, truth))
                c["wrong_visible"] += sum(a != b for a, b in zip(g[:VISIBLE], truth[:VISIBLE]))
    chars = c["lines"] * 44 + failed * 88  # a path that produced no lines scores every char wrong
    return {
        "records": records, "path_failures": failed,
        "cer_greedy": round((c["wrong_greedy"] + failed * 88) / (records * 88), 5),
        "cer_constrained": round((c["wrong_constrained"] + failed * 88) / (records * 88), 5),
        "cer_greedy_first39": round((c["wrong_visible"] + failed * 2 * VISIBLE) / (records * 2 * VISIBLE), 5),
        "line_exact_greedy": round(c["exact_greedy"] / max(1, records * 2), 4),
        "chars_scored": chars,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True, help="key in the report, e.g. before, after_full")
    p.add_argument("--render", choices=("training", "full"), default="training")
    p.add_argument("--normalizer", choices=("current", "legacy"), default="current",
                   help="legacy = the pre-B1h whole-crop fit, for the 'before' rows")
    p.add_argument("--paths", default=",".join(PATHS), help="comma-separated subset of paths")
    p.add_argument("--records", type=int, default=200)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--weights", default=str(BACKEND_ROOT / "models/mrz_crnn/1.4.0/model.onnx"))
    p.add_argument("--out", default=str(BACKEND_ROOT / "scripts/geometry_report.json"))
    args = p.parse_args()
    if args.records < 200:
        raise SystemExit("--records must be at least 200")
    if args.seed in (TRAIN_SEED, B1G_RECOVERY_SEED):
        raise SystemExit("seed must differ from the training seed and the B1g recovery-set seed")

    import onnxruntime as ort

    if args.normalizer == "legacy":
        canonical.normalize_line = canonical._fit_whole_crop  # type: ignore[assignment]
    session = ort.InferenceSession(args.weights, providers=["CPUExecutionProvider"])
    result: dict = {}
    t0 = time.perf_counter()
    for path in args.paths.split(","):
        result[path] = {}
        for h in HEIGHTS:
            result[path][str(h)] = evaluate(session, path, h, args.records, args.seed, args.render)
            r = result[path][str(h)]
            print(f"{args.label:6s} {path:14s} h={h}: CER greedy {r['cer_greedy']:.2%} "
                  f"first39 {r['cer_greedy_first39']:.2%} constrained {r['cer_constrained']:.2%} exact-line {r['line_exact_greedy']:.1%} "
                  f"failures {r['path_failures']}", flush=True)
    out = Path(args.out)
    report = json.loads(out.read_text()) if out.exists() else {}
    report.setdefault(args.label, {}).update(result)
    report["meta"] = {"records": args.records, "seed": args.seed, "severity": "uniform(0,1)",
                      "heights": list(HEIGHTS), "visible_chars": VISIBLE, "note": "CER = wrong chars / all 88 chars per record; "
                      "path failures count as fully wrong"}
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"done in {time.perf_counter() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
