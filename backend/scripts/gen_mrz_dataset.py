#!/usr/bin/env python
"""Pre-generate a fixed MRZ training/val/test set so train.py stops paying
synth.py's generation cost on every epoch.

Writes to `<out>/{train,val,test}/` -- PNG crops plus a `manifest.jsonl` of
one `{filename, line_index, label, severity, seed}` row per line-crop.

Split boundary is the *source record*, not the individual line sample: each
TD3 record produces two line crops (line 0: name/doc-type line, line 1: the
checksum-bearing line), and both must land in the same split. Splitting the
derived line samples independently would let the same record's line 1 appear
in train while its sibling line 0 appears in test -- the model would then be
evaluated on a record it saw a highly correlated view of during training.
This script generates whole records, assigns each record's *index* to a
split up front, and only then renders+writes its lines -- so the split is
enforced before any sample exists, not filtered after the fact.

Run: python scripts/gen_mrz_dataset.py --train 120000 --val 12000 --test 12000 \
    --confusable-bias 0.5 --seed 42 --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
synth: ModuleType
spec: ModuleType


@dataclass
class ManifestRow:
    filename: str
    line_index: int
    label: str
    severity: float
    seed: int


def _worker_init() -> None:
    # Re-import inside each worker process rather than at module scope: on
    # Windows, multiprocessing spawns fresh interpreters that re-exec this
    # module, and importing app.ocr.mrz before sys.path is set up (spawn
    # start method) would fail.
    global synth, spec  # noqa: PLW0603
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from app.ocr.mrz import spec as _spec  # noqa: PLC0415
    from app.ocr.mrz import synth as _synth  # noqa: PLC0415

    synth, spec = _synth, _spec


def _generate_one(args: tuple[int, int, float, Path]) -> tuple[list[ManifestRow], str]:
    """Generate one record, render+degrade both lines, write PNGs.

    Returns (manifest_rows, record_key) where `record_key` is the joined
    ground-truth text -- used by the caller to assert split disjointness.
    """
    record_seed, base_seed, confusable_bias, split_dir = args
    rng = random.Random(base_seed * 7_919 + record_seed)
    record = synth.generate_record(rng, confusable_bias=confusable_bias)

    parsed = spec.parse_td3(record.lines)
    if not parsed.all_valid:
        raise AssertionError(f"generated record failed self-validation: {record.lines}")

    severity = rng.uniform(0.0, 1.0)
    rows: list[ManifestRow] = []
    for line_index, line_text in enumerate(record.lines):
        clean = synth.render_mrz_lines([line_text], char_h=32)
        degraded = synth.degrade(clean, rng, severity=severity)
        filename = f"{record_seed:08d}_{line_index}.png"
        Image.fromarray(degraded).save(split_dir / filename)
        rows.append(
            ManifestRow(
                filename=filename, line_index=line_index, label=line_text,
                severity=severity, seed=record_seed,
            )
        )
    record_key = "|".join(record.lines)
    return rows, record_key


def generate_split(
    *, split_name: str, n_records: int, record_seed_offset: int, base_seed: int,
    confusable_bias: float, out_dir: Path, workers: int,
) -> tuple[list[ManifestRow], set[str]]:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (record_seed_offset + i, base_seed, confusable_bias, split_dir)
        for i in range(n_records)
    ]

    all_rows: list[ManifestRow] = []
    record_keys: set[str] = set()

    if workers <= 1:
        _worker_init()
        results = [_generate_one(t) for t in tasks]
    else:
        with mp.Pool(processes=workers, initializer=_worker_init) as pool:
            results = pool.map(_generate_one, tasks, chunksize=max(1, n_records // (workers * 8) or 1))

    for rows, record_key in results:
        all_rows.extend(rows)
        record_keys.add(record_key)

    manifest_path = split_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(asdict(row)) + "\n")

    return all_rows, record_keys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=120_000, help="number of source records for the train split")
    parser.add_argument("--val", type=int, default=12_000)
    parser.add_argument("--test", type=int, default=12_000)
    parser.add_argument("--confusable-bias", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 1) - 1))
    parser.add_argument("--out", type=str, default="data/mrz")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = PROJECT_ROOT / args.out

    splits = [("train", args.train), ("val", args.val), ("test", args.test)]
    record_seed_offset = 0
    all_keys: dict[str, set[str]] = {}
    severities: dict[str, list[float]] = {}

    t0 = time.time()
    total_records = sum(n for _, n in splits)
    for split_name, n_records in splits:
        rows, keys = generate_split(
            split_name=split_name, n_records=n_records, record_seed_offset=record_seed_offset,
            base_seed=args.seed, confusable_bias=args.confusable_bias, out_dir=out_dir, workers=args.workers,
        )
        record_seed_offset += n_records
        all_keys[split_name] = keys
        severities[split_name] = [r.severity for r in rows]
        print(f"{split_name}: {n_records} records -> {len(rows)} line samples written to {out_dir / split_name}")

    elapsed = time.time() - t0
    n_line_samples = sum(2 * n for _, n in splits)
    print(f"generated {total_records} records ({n_line_samples} line samples) in {elapsed:.1f}s "
          f"-> {n_line_samples / elapsed:.1f} samples/sec with {args.workers} workers")

    # Split-disjointness assertion: since record seeds are drawn from
    # disjoint, non-overlapping ranges per split, and generate_record is a
    # pure function of its rng draw sequence, distinct seeds essentially
    # never collide -- but assert it directly on content, not just on seed
    # ranges, so a future refactor that breaks the seed-disjointness
    # invariant fails loudly here instead of silently leaking.
    train_val = all_keys["train"] & all_keys["val"]
    train_test = all_keys["train"] & all_keys["test"]
    val_test = all_keys["val"] & all_keys["test"]
    assert not train_val, f"{len(train_val)} records leaked between train and val"
    assert not train_test, f"{len(train_test)} records leaked between train and test"
    assert not val_test, f"{len(val_test)} records leaked between val and test"
    print("split-disjointness check OK: no source record appears in two splits")

    manifest_summary = {
        "seed": args.seed,
        "confusable_bias": args.confusable_bias,
        "splits": {
            name: {
                "n_records": n,
                "n_line_samples": 2 * n,
                "severity_min": min(severities[name]) if severities[name] else None,
                "severity_max": max(severities[name]) if severities[name] else None,
                "severity_mean": float(np.mean(severities[name])) if severities[name] else None,
            }
            for name, n in splits
        },
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(manifest_summary, indent=2))
    print(f"wrote {out_dir / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
