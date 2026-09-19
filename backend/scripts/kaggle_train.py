#!/usr/bin/env python
"""Unattended MRZ recogniser training for Kaggle (or any GPU box).

One command, no interaction:

    python kaggle_train.py                       # full run: 120k/12k/12k, 30 epochs
    python kaggle_train.py --smoke               # ~minutes, CPU-friendly plumbing check

Stages (each prints a banner and is skipped if its output already exists, so
re-running after a disconnect picks up where the last run stopped):

  1. clone     git clone the repo (skipped with --repo-dir or if already inside it)
  2. deps      pip install only what Kaggle's image lacks (never touches torch)
  3. dataset   scripts/gen_mrz_dataset.py with all CPU cores
  4. train     app.ocr.mrz.train --disk: severity curriculum, checkpoint every
               epoch to <work>/checkpoints, resumes from last.pt, logs the
               confusable metric every epoch
  5. export    scripts/export_mrz_onnx.py -> ONNX + metadata.json + manifest.json
               copied to <work>/artifacts

Outputs land in /kaggle/working (the only directory Kaggle preserves) when it
exists, otherwise ./kaggle_work. The bulky PNG dataset goes to /kaggle/temp so
it is not counted as notebook output.

This script uses only the standard library until the deps stage, so it can
start on a bare interpreter.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/Gnim12/SIH-2026-AI-Baise-fraude-detection.git"


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n[{time.strftime('%H:%M:%S')}] {text}\n{'=' * 78}", flush=True)


def run(cmd: list[str], *, cwd: Path | None = None, log: Path | None = None) -> None:
    """Run a command, stream its output live, tee it to `log`, raise on failure."""
    print("$ " + " ".join(cmd), flush=True)
    with subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    ) as proc:
        assert proc.stdout is not None
        log_f = log.open("a", encoding="utf-8") if log else None
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                if log_f:
                    log_f.write(line)
                    log_f.flush()
        finally:
            if log_f:
                log_f.close()
        if proc.wait() != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def locate_backend(start: Path) -> Path | None:
    for candidate in (start, start / "backend", *start.parents):
        if (candidate / "app" / "ocr" / "mrz" / "model.py").exists():
            return candidate
        if (candidate / "backend" / "app" / "ocr" / "mrz" / "model.py").exists():
            return candidate / "backend"
    return None


def stage_clone(args: argparse.Namespace, work: Path) -> Path:
    banner("1/5 clone")
    if args.repo_dir:
        backend = locate_backend(Path(args.repo_dir).resolve())
    else:
        backend = locate_backend(Path(__file__).resolve().parent)
        if backend is None:
            dest = work / "repo"
            if not dest.exists():
                cmd = ["git", "clone", "--depth", "1"]
                if args.branch:
                    cmd += ["--branch", args.branch]
                run(cmd + [args.repo_url, str(dest)])
            backend = locate_backend(dest)
    if backend is None:
        sys.exit("could not find backend/app/ocr/mrz in the checkout -- is the M-PREP work pushed?")
    if not (backend / "assets" / "fonts" / "OCRB.ttf").exists():
        sys.exit(f"{backend}/assets/fonts/OCRB.ttf is missing -- the checkout predates the OCR-B "
                 "font commit; push it before training (synthetic data would silently use a different font)")
    print(f"backend root: {backend}", flush=True)
    return backend


def stage_deps() -> None:
    banner("2/5 deps")
    needed = {"onnx": "onnx", "onnxruntime": "onnxruntime", "PIL": "pillow", "scipy": "scipy", "numpy": "numpy"}
    missing = [pkg for mod, pkg in needed.items() if importlib.util.find_spec(mod) is None]
    if missing:
        run([sys.executable, "-m", "pip", "install", "-q", *missing])
    else:
        print("all dependencies already present")
    import torch  # noqa: PLC0415

    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}", flush=True)


def stage_dataset(args: argparse.Namespace, backend: Path, data_dir: Path) -> None:
    banner("3/5 dataset")
    summary = data_dir / "dataset_summary.json"
    if summary.exists():
        done = json.loads(summary.read_text())["splits"]
        if (done["train"]["n_records"], done["val"]["n_records"], done["test"]["n_records"]) == (
            args.train_records, args.val_records, args.test_records,
        ):
            print(f"dataset already generated at {data_dir}, skipping")
            return
        shutil.rmtree(data_dir)
    run([
        sys.executable, "scripts/gen_mrz_dataset.py",
        "--train", str(args.train_records), "--val", str(args.val_records), "--test", str(args.test_records),
        "--confusable-bias", str(args.confusable_bias), "--seed", str(args.seed),
        "--workers", str(args.workers), "--out", os.path.relpath(data_dir, backend),
    ], cwd=backend)


def stage_train(args: argparse.Namespace, backend: Path, data_dir: Path, ckpt_dir: Path) -> bool:
    banner("4/5 train (resumes from last.pt if present)")
    cmd = [
        sys.executable, "-u", "-m", "app.ocr.mrz.train", "--disk",
        "--train-dir", str(data_dir / "train"), "--val-dir", str(data_dir / "val"),
        "--checkpoint-dir", str(ckpt_dir), "--epochs", str(args.epochs), "--batch", str(args.batch),
        "--workers", str(args.loader_workers),
    ]
    if args.max_steps_per_epoch:
        cmd += ["--max-steps-per-epoch", str(args.max_steps_per_epoch)]
    if args.time_budget_hours:
        cmd += ["--time-budget-hours", str(args.time_budget_hours)]
    try:
        run(cmd, cwd=backend, log=ckpt_dir / "train.log")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"!! training exited with {exc.returncode}; will still export best.pt if one exists", flush=True)
        return False


def stage_export(args: argparse.Namespace, backend: Path, ckpt_dir: Path, artifacts: Path) -> None:
    banner("5/5 export")
    best = ckpt_dir / "best.pt"
    if not best.exists():
        sys.exit("no best.pt to export -- training produced no checkpoint")
    models_dir = artifacts / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(backend / "models" / "manifest.json", models_dir / "manifest.json")
    config = {
        "epochs": args.epochs, "batch": args.batch, "confusable_bias": args.confusable_bias, "seed": args.seed,
        "train_records": args.train_records, "val_records": args.val_records, "test_records": args.test_records,
        "smoke": args.smoke,
    }
    run([
        sys.executable, "scripts/export_mrz_onnx.py", "--checkpoint", os.path.relpath(best, backend),
        "--version", args.version, "--models-dir", os.path.relpath(models_dir, backend),
        "--training-config", json.dumps(config),
    ], cwd=backend)
    for name in ("metrics.jsonl", "train.log"):
        if (ckpt_dir / name).exists():
            shutil.copy(ckpt_dir / name, artifacts / name)
    print(f"\nartifacts in {artifacts}:", flush=True)
    for path in sorted(artifacts.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(artifacts)}  ({path.stat().st_size:,} bytes)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end plumbing run (not a real training run)")
    p.add_argument("--repo-url", default=REPO_URL)
    p.add_argument("--branch", default=None)
    p.add_argument("--repo-dir", default=None, help="use an existing checkout instead of cloning")
    p.add_argument("--work-dir", default=None, help="default: /kaggle/working if it exists, else ./kaggle_work")
    p.add_argument("--version", default="1.4.0")
    p.add_argument("--train-records", type=int, default=120_000)
    p.add_argument("--val-records", type=int, default=12_000)
    p.add_argument("--test-records", type=int, default=12_000)
    p.add_argument("--confusable-bias", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2, help="dataset generation processes")
    p.add_argument("--loader-workers", type=int, default=2, help="DataLoader workers during training")
    p.add_argument("--max-steps-per-epoch", type=int, default=None)
    p.add_argument("--time-budget-hours", type=float, default=10.5,
                   help="stop starting new epochs after this long so export still happens inside Kaggle's 12h limit")
    p.add_argument("--allow-cpu", action="store_true", help="permit a full run without a GPU (very slow)")
    args = p.parse_args()

    if args.smoke:
        args.train_records, args.val_records, args.test_records = 60, 12, 12
        args.epochs, args.batch, args.loader_workers, args.max_steps_per_epoch = 2, 8, 0, 2
        args.workers = min(args.workers, 2)
        args.version = "0.0.0-smoke"
        args.time_budget_hours = None

    kaggle = Path("/kaggle/working").exists()
    work = Path(args.work_dir) if args.work_dir else (Path("/kaggle/working") if kaggle else Path("kaggle_work"))
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    data_root = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else work / "data"
    data_dir = (data_root / ("mrz_smoke" if args.smoke else "mrz")).resolve()
    ckpt_dir, artifacts = work / "checkpoints", work / "artifacts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    backend = stage_clone(args, work)
    stage_deps()

    import torch  # noqa: PLC0415

    if not torch.cuda.is_available() and not (args.smoke or args.allow_cpu):
        sys.exit("no GPU detected: a full run on CPU would take days. Enable a GPU accelerator "
                 "in the Kaggle notebook settings, or pass --allow-cpu to override.")

    stage_dataset(args, backend, data_dir)
    train_ok = stage_train(args, backend, data_dir, ckpt_dir)
    stage_export(args, backend, ckpt_dir, artifacts)

    banner(f"done in {(time.time() - started) / 60:.1f} min" + ("" if train_ok else " (training exited abnormally)"))
    sys.exit(0 if train_ok else 1)


if __name__ == "__main__":
    main()
