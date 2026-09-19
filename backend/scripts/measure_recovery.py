#!/usr/bin/env python
"""Measure what checksum-constrained decoding buys over plain greedy argmax.

Generates a fresh synthetic TD3 test set (seed distinct from training's 42),
renders each line exactly as the training/eval pipeline does (single line at
data.TARGET_HEIGHT, degraded, horizontally jittered, normalize_line), runs the
installed ONNX model once, then decodes every record twice from the same
log-probabilities:

  greedy       decode.decode_mrz(...).greedy_lines  (per-slot argmax)
  constrained  decode.decode_mrz(...).lines         (checksum beam search)

and compares both to ground truth. Everything that could make constrained
decoding look better than it is gets its own number: the harm rate (a correct
greedy character turned wrong), wrong->different-wrong changes, and the
correctness of suspect-flagged recoveries. Metrics are reported over all
characters AND over the checksum-guarded span only, since constrained decoding
cannot touch anything else.

Scope caveat, printed in the report: this measures the decoder on line crops in
the training geometry. It says nothing about the detect.py -> normalize_line
path on whole pages (see the B1g report).

Run: python scripts/measure_recovery.py [--records 2000] [--seed 20260919]
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

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_geometry as geometry  # noqa: E402
from app.ocr.mrz import canonical, data, decode, spec, synth  # noqa: E402
from app.pipeline.thresholds import DEFAULT_THRESHOLDS  # noqa: E402

TRAIN_SEED = 42
BATCH = 64


def guarded_positions() -> set[tuple[int, int]]:
    """(line, position) of every character a constrained group may rewrite."""
    out: set[tuple[int, int]] = set()
    for lay in decode.LAYOUTS[spec.MrzFormat.TD3]:
        for pos in range(lay.field_start, lay.check_digit_index + 1):
            out.add((lay.line, pos))
    return out


def render_line(text: str, rng: random.Random, severity_range: tuple[float, float]) -> np.ndarray:
    clean = synth.render_mrz_lines([text], char_h=rng.randint(*data.GLYPH_HEIGHT_RANGE))
    degraded = synth.degrade(clean, rng, severity=rng.uniform(*severity_range))
    return data.offset_jitter(degraded, rng)


def load_session(weights: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])


def run_model(session, images: list[np.ndarray], normalize=None) -> np.ndarray:
    normalize = normalize or geometry._NORMALIZE
    out = []
    for i in range(0, len(images), BATCH):
        chunk = images[i : i + BATCH]
        batch = np.stack([normalize(im) for im in chunk]).astype(np.float32)[:, None] / 255.0
        out.append(session.run(["log_probs"], {"images": batch})[0])
    return np.concatenate(out)


def measure(args: argparse.Namespace) -> dict:
    if args.seed == TRAIN_SEED:
        raise SystemExit(f"--seed must differ from the training seed ({TRAIN_SEED})")
    weights = Path(args.weights)
    session = load_session(weights)
    meta = geometry.configure(weights)
    guarded = guarded_positions()
    margin = DEFAULT_THRESHOLDS.mrz_recovery_confidence_margin

    t0 = time.perf_counter()
    rng = random.Random(args.seed)
    records = [synth.generate_record(rng, confusable_bias=args.confusable_bias) for _ in range(args.records)]
    path = getattr(args, "path", "crops")
    page_failures = 0
    if path == "crops":
        # Single-line crops from the training-style renderer (glyph height 22-44, offset jitter).
        images = [render_line(line, rng, (0.0, 1.0)) for r in records for line in r.lines]
        lp = run_model(session, images).reshape(args.records, 2, -1, len(spec.MRZ_CHARSET))
    else:
        # Full pages: complete two-line band, severity uniform 0-1, pasted on a page, found by
        # detect.find_mrz, normalised by the current canonical.normalize_line.
        per_rec = [geometry.make_lines("page_detect", r.lines, 32, rng, "full") for r in records]
        kept = [(r, im) for r, im in zip(records, per_rec) if im is not None and len(im) == 2]
        page_failures = len(records) - len(kept)
        records = [r for r, _ in kept]
        images = [x for _, im in kept for x in im]
        lp = run_model(session, images).reshape(len(records), 2, -1, len(spec.MRZ_CHARSET))
    print(f"[{path}] generated + inferred {len(records)} records in {time.perf_counter() - t0:.1f}s"
          f" ({page_failures} page-path failures)")

    tot = collections.Counter()
    span = collections.Counter()
    per_char = collections.defaultdict(collections.Counter)  # truth char -> counts (guarded span only)
    per_char_all = collections.defaultdict(collections.Counter)
    suspect = collections.Counter()
    status = collections.Counter()
    pos_wrong = np.zeros((2, 44))
    n_records = len(records)

    for rec, rec_lp in zip(records, lp):
        d = decode.decode_mrz([rec_lp[0], rec_lp[1]], spec.MrzFormat.TD3)
        status[d.status.value] += 1
        for li, truth in enumerate(rec.lines):
            g, c = d.greedy_lines[li], d.lines[li]
            tot["lines"] += 1
            tot["line_exact_greedy"] += g == truth
            tot["line_exact_constrained"] += c == truth
            pos_wrong[li] += [a != b for a, b in zip(g, truth)]
            for pos, (tc, gc, cc) in enumerate(zip(truth, g, c)):
                bucket = [tot] + ([span] if (li, pos) in guarded else [])
                for b in bucket:
                    b["chars"] += 1
                    b["greedy_wrong"] += gc != tc
                    b["constrained_wrong"] += cc != tc
                    b["corrected"] += gc != tc and cc == tc            # greedy wrong -> constrained right
                    b["harmed"] += gc == tc and cc != tc               # greedy right -> constrained wrong
                    b["wrong_to_other_wrong"] += gc != tc and cc != tc and gc != cc
                    b["changed"] += gc != cc
                for store in ([per_char_all] + ([per_char] if (li, pos) in guarded else [])):
                    s = store[tc]
                    s["n"] += 1
                    s["greedy_wrong"] += gc != tc
                    s["corrected"] += gc != tc and cc == tc
                    s["harmed"] += gc == tc and cc != tc
        for rc in d.recovered:
            truth_ch = rec.lines[rc.line][rc.position]
            is_suspect = (rc.raw_confidence - rc.recovered_confidence) > margin
            key = "suspect" if is_suspect else "unsuspected"
            suspect[f"{key}_recoveries"] += 1
            suspect[f"{key}_correct"] += rc.recovered == truth_ch
            suspect[f"{key}_raw_was_right"] += rc.raw == truth_ch

    def rate(a: int, b: int) -> float | None:
        return round(a / b, 6) if b else None

    def block(c: collections.Counter) -> dict:
        return {
            "chars": c["chars"],
            "cer_greedy": rate(c["greedy_wrong"], c["chars"]),
            "cer_constrained": rate(c["constrained_wrong"], c["chars"]),
            "greedy_wrong_chars": c["greedy_wrong"],
            "constrained_wrong_chars": c["constrained_wrong"],
            "corrected_chars": c["corrected"],
            "recovery_rate": rate(c["corrected"], c["greedy_wrong"]),
            "harmed_chars": c["harmed"],
            "harm_rate_of_correct_greedy": rate(c["harmed"], c["chars"] - c["greedy_wrong"]),
            "wrong_to_other_wrong": c["wrong_to_other_wrong"],
            "net_chars_fixed": c["corrected"] - c["harmed"],
        }

    def per_char_block(store) -> dict:
        rows = {}
        for ch in sorted(store):
            s = store[ch]
            rows[ch] = {"n": s["n"], "greedy_wrong": s["greedy_wrong"], "corrected": s["corrected"],
                        "recovery_rate": rate(s["corrected"], s["greedy_wrong"]), "harmed": s["harmed"]}
        return rows

    lines = tot["lines"]
    return {
        "model_version": meta["version"],
        "model_sha256": meta["sha256"],
        "path": path, "page_path_failures": page_failures,
        "records": len(records), "lines": lines, "seed": args.seed, "training_seed": TRAIN_SEED,
        "confusable_bias": args.confusable_bias, "severity": "uniform(0,1)",
        "geometry": ("single-line crops from the training renderer (glyph 22-44 px, offset jitter)"
                     if path == "crops" else "full page -> detect.find_mrz -> normalize_line (ink extent)"),
        "all_characters": block(tot),
        "checksum_guarded_span": block(span),
        "line_exact_match": {
            "greedy": rate(tot["line_exact_greedy"], lines),
            "constrained": rate(tot["line_exact_constrained"], lines),
        },
        "records_status": dict(status),
        "per_position_error_greedy": {
            f"line{li + 1}": {str(p): round(float(pos_wrong[li][p] / n_records), 4) for p in range(44)}
            for li in range(2)
        },
        "per_character_guarded_span": per_char_block(per_char),
        "per_character_all": per_char_block(per_char_all),
        "recoveries": {
            "total": suspect["suspect_recoveries"] + suspect["unsuspected_recoveries"],
            "suspect_margin": margin, **dict(suspect),
        },
    }


def summarise(r: dict) -> str:
    a, s = r["all_characters"], r["checksum_guarded_span"]
    lines = [
        f"[{r['path']}] model {r['model_version']} | {r['records']} records / {r['lines']} lines | seed {r['seed']} (train {r['training_seed']})",
        f"[all chars]   CER greedy {a['cer_greedy']:.4%}  constrained {a['cer_constrained']:.4%}",
        f"              greedy-wrong {a['greedy_wrong_chars']}  corrected {a['corrected_chars']} "
        f"(recovery {a['recovery_rate'] if a['recovery_rate'] is None else format(a['recovery_rate'], '.2%')})  "
        f"harmed {a['harmed_chars']}  net {a['net_chars_fixed']:+d}",
        f"[guarded]     CER greedy {s['cer_greedy']:.4%}  constrained {s['cer_constrained']:.4%}  "
        f"({s['chars']} chars) recovery {s['recovery_rate'] if s['recovery_rate'] is None else format(s['recovery_rate'], '.2%')}  "
        f"harmed {s['harmed_chars']}",
        f"line exact    greedy {r['line_exact_match']['greedy']:.2%}  constrained {r['line_exact_match']['constrained']:.2%}",
        "per-char (guarded span): char  n  greedy_wrong  corrected  recovery  harmed",
    ]
    for ch, v in r["per_character_guarded_span"].items():
        if v["greedy_wrong"] or v["harmed"]:
            rr = "  -  " if v["recovery_rate"] is None else f"{v['recovery_rate']:.0%}"
            lines.append(f"   {ch!r:>5} {v['n']:6d} {v['greedy_wrong']:6d} {v['corrected']:6d}  {rr:>5} {v['harmed']:5d}")
    pp = r["per_position_error_greedy"]
    lines.append("per-position greedy error, positions 38-43:  " + "  ".join(
        f"L{li}:" + ",".join(f"{pp[f'line{li}'][str(p)]:.1%}" for p in range(38, 44)) for li in (1, 2)))
    rc = r["recoveries"]
    lines.append(f"recoveries: {rc['total']} total | suspect {rc.get('suspect_recoveries', 0)} "
                 f"(correct {rc.get('suspect_correct', 0)}) | unsuspected {rc.get('unsuspected_recoveries', 0)} "
                 f"(correct {rc.get('unsuspected_correct', 0)})")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--confusable-bias", type=float, default=0.5, help="matches the training set's bias")
    p.add_argument("--weights", default=str(BACKEND_ROOT / "models/mrz_crnn/1.4.0/model.onnx"))
    p.add_argument("--out", default=str(BACKEND_ROOT / "scripts/recovery_report.json"))
    args = p.parse_args()
    if args.records < 2000:
        raise SystemExit("--records must be at least 2000")
    report = {}
    for path in ("page", "crops"):
        args.path = path
        report[path] = measure(args)
        print(summarise(report[path]))
        print()
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
