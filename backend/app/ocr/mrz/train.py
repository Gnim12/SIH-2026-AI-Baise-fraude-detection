"""Training loop for MrzCRNN (fixed-slot head), with a severity curriculum.

Curriculum: early steps train mostly on lightly-degraded (easy) synthetic
lines so the model first learns the charset and basic shapes, then the
degradation severity range widens over the course of training so it also
learns to be robust to the print-scan noise, blur and skew it will see on
real captures. This is a training loop, not run as part of the module vendor
self-test suite (no GPU in this environment -- see BACKEND_BRIEF.md §1.3);
its self-test below runs a handful of CPU steps on a tiny synthetic dataset
just to prove the loop is wired correctly end to end.

Loss is per-slot cross-entropy, not CTC: the fixed-slot head (model.py)
already emits exactly `line_len` class distributions, one per character
position, so there's no alignment to marginalize over.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset

from functools import partial

from .data import MrzDiskDataset, MrzSample, SyntheticMrzLineDataset, WidthBucketedSampler, collate_batch
from .model import MrzCRNN
from .spec import MRZ_CHARSET

_CONFUSABLE_CHARS = set("0O1I5S8B2Z")
_CONFUSABLE_IDX = {MRZ_CHARSET.index(c) for c in _CONFUSABLE_CHARS}


def severity_for_step(step: int, total_steps: int, *, start: tuple[float, float] = (0.0, 0.3), end: tuple[float, float] = (0.1, 0.9)) -> tuple[float, float]:
    """Linear curriculum: severity range widens from `start` to `end` over training."""
    t = min(1.0, step / max(total_steps, 1))
    lo = start[0] + (end[0] - start[0]) * t
    hi = start[1] + (end[1] - start[1]) * t
    return (lo, hi)


class RunningMetrics:
    """Accumulates per-character accuracy, full-line exact match, and
    confusable-set accuracy across a validation pass."""

    def __init__(self) -> None:
        self.n_chars = 0
        self.n_chars_correct = 0
        self.n_lines = 0
        self.n_lines_correct = 0
        self.n_confusable = 0
        self.n_confusable_correct = 0
        self.per_char_total: dict[int, int] = {i: 0 for i in _CONFUSABLE_IDX}
        self.per_char_correct: dict[int, int] = {i: 0 for i in _CONFUSABLE_IDX}

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        # preds, targets: (B, line_len)
        correct = preds == targets
        self.n_chars += targets.numel()
        self.n_chars_correct += int(correct.sum().item())
        self.n_lines += targets.shape[0]
        self.n_lines_correct += int(correct.all(dim=1).sum().item())

        confusable_mask = torch.zeros_like(targets, dtype=torch.bool)
        for idx in _CONFUSABLE_IDX:
            is_idx = targets == idx
            confusable_mask |= is_idx
            self.per_char_total[idx] += int(is_idx.sum().item())
            self.per_char_correct[idx] += int((correct & is_idx).sum().item())
        self.n_confusable += int(confusable_mask.sum().item())
        self.n_confusable_correct += int((correct & confusable_mask).sum().item())

    @property
    def char_accuracy(self) -> float:
        return self.n_chars_correct / max(self.n_chars, 1)

    @property
    def line_accuracy(self) -> float:
        return self.n_lines_correct / max(self.n_lines, 1)

    @property
    def confusable_accuracy(self) -> float:
        return self.n_confusable_correct / max(self.n_confusable, 1)

    @property
    def per_confusable_char(self) -> dict[str, float]:
        """Accuracy for each of 0 O 1 I 5 S 8 B 2 Z on its own, so a collapse
        on one pair (e.g. 0 vs O) is visible instead of averaged away."""
        return {
            MRZ_CHARSET[i]: self.per_char_correct[i] / max(self.per_char_total[i], 1)
            for i in sorted(_CONFUSABLE_IDX)
        }


@torch.no_grad()
def evaluate(model: MrzCRNN, loader: "DataLoader[MrzSample]", device: str) -> RunningMetrics:
    model.eval()
    metrics = RunningMetrics()
    for batch in loader:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        log_probs = model(images)  # (B, line_len, C)
        preds = log_probs.argmax(dim=-1)
        metrics.update(preds, targets)
    return metrics


def train(
    *,
    epochs: int = 30,
    batch_size: int = 64,
    steps_per_epoch: int = 400,
    lr: float = 3e-4,
    line_len: int = 44,
    jitter_frac: float = 0.03,
    val_size: int = 1000,
    checkpoint_dir: str | Path = "models/mrz_crnn",
    device: str | None = None,
) -> MrzCRNN:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MrzCRNN(line_len=line_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    val_dataset = SyntheticMrzLineDataset(
        size=val_size, base_seed=-1, severity_range=(0.0, 1.0), jitter_frac=jitter_frac
    )
    val_loader = DataLoader[MrzSample](val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    total_steps = epochs * steps_per_epoch
    global_step = 0
    best_char_accuracy = -1.0
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "model.pt"

    for epoch in range(epochs):
        severity_range = severity_for_step(global_step, total_steps)
        dataset = SyntheticMrzLineDataset(
            size=steps_per_epoch * batch_size, base_seed=epoch,
            severity_range=severity_range, jitter_frac=jitter_frac,
        )
        sampler = WidthBucketedSampler(dataset, batch_size=batch_size)
        loader = DataLoader[MrzSample](dataset, batch_sampler=sampler, collate_fn=partial(collate_batch, canvas_jitter=True))

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        for batch in loader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)  # (B, line_len)

            log_probs = model(images)  # (B, line_len, C)
            loss = criterion(log_probs.reshape(-1, log_probs.shape[-1]), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        elapsed = time.time() - start_time
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch + 1}/{epochs}  loss={epoch_loss / max(n_batches, 1):.4f}  "
            f"severity={severity_range}  char_acc={val_metrics.char_accuracy:.4f}  "
            f"line_acc={val_metrics.line_accuracy:.4f}  "
            f"confusable_acc={val_metrics.confusable_accuracy:.4f}  {elapsed:.1f}s"
        )

        if val_metrics.char_accuracy > best_char_accuracy:
            best_char_accuracy = val_metrics.char_accuracy
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "line_len": line_len,
                    "char_accuracy": best_char_accuracy,
                    "line_accuracy": val_metrics.line_accuracy,
                    "confusable_accuracy": val_metrics.confusable_accuracy,
                    "epoch": epoch + 1,
                },
                best_ckpt_path,
            )
            print(f"  new best val char_accuracy={best_char_accuracy:.4f} -> saved {best_ckpt_path}")

    print(f"training complete, best checkpoint at {best_ckpt_path} (char_accuracy={best_char_accuracy:.4f})")
    return model


def train_from_disk(
    *,
    train_dir: str | Path,
    val_dir: str | Path,
    max_steps: int,
    batch_size: int = 32,
    lr: float = 3e-4,
    line_len: int = 44,
    checkpoint_dir: str | Path = "models/mrz_crnn",
    device: str | None = None,
) -> tuple[MrzCRNN, list[float]]:
    """Train against a pre-generated dataset (scripts/gen_mrz_dataset.py)
    instead of generating on the fly, capped at `max_steps` batches. Used for
    the M-PREP smoke run: prove the loop is wired end to end (loss decreases,
    metrics compute, a checkpoint saves) on a small fixed set, not a real
    training run.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MrzCRNN(line_len=line_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    train_dataset = MrzDiskDataset(train_dir)
    val_dataset = MrzDiskDataset(val_dir, jitter_frac=0.0)
    train_loader = DataLoader[MrzSample](
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_batch, canvas_jitter=True)
    )
    val_loader = DataLoader[MrzSample](val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "model.pt"

    loss_history: list[float] = []
    best_char_accuracy = -1.0
    global_step = 0
    model.train()
    start_time = time.time()

    while global_step < max_steps:
        for batch in train_loader:
            if global_step >= max_steps:
                break
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)

            log_probs = model(images)
            loss = criterion(log_probs.reshape(-1, log_probs.shape[-1]), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_history.append(loss.item())
            global_step += 1

            if global_step % 50 == 0 or global_step == max_steps:
                val_metrics = evaluate(model, val_loader, device)
                model.train()
                print(
                    f"step {global_step}/{max_steps}  loss={loss.item():.4f}  "
                    f"char_acc={val_metrics.char_accuracy:.4f}  "
                    f"line_acc={val_metrics.line_accuracy:.4f}  "
                    f"confusable_acc={val_metrics.confusable_accuracy:.4f}"
                )
                if val_metrics.char_accuracy > best_char_accuracy:
                    best_char_accuracy = val_metrics.char_accuracy
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "line_len": line_len,
                            "char_accuracy": best_char_accuracy,
                            "line_accuracy": val_metrics.line_accuracy,
                            "confusable_accuracy": val_metrics.confusable_accuracy,
                            "step": global_step,
                        },
                        ckpt_path,
                    )

    elapsed = time.time() - start_time
    print(f"smoke run: {global_step} steps in {elapsed:.1f}s, best char_accuracy={best_char_accuracy:.4f} "
          f"(noise at this scale -- not a trained-model result), checkpoint at {ckpt_path}")
    return model, loss_history


def _atomic_save(obj: dict[str, Any], path: Path) -> None:
    """Write-then-rename so a disconnect mid-save never leaves a truncated
    checkpoint that a resume would then choke on."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_disk_curriculum(
    *,
    train_dir: str | Path,
    val_dir: str | Path,
    checkpoint_dir: str | Path,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 3e-4,
    line_len: int = 44,
    num_workers: int = 2,
    max_steps_per_epoch: int | None = None,
    time_budget_s: float | None = None,
    device: str | None = None,
) -> Path:
    """Unattended, resumable training over a pre-generated dataset.

    Severity curriculum: the manifest records each sample's severity, so epoch
    `e` trains only on samples whose severity is inside a window that widens
    from [0, 0.3] to [0, 1.0] -- easy shapes first, full print-scan damage by
    the end.

    Every epoch: `last.pt` (model + optimiser + progress) is rewritten
    atomically so a disconnect loses at most one epoch and a re-run resumes;
    `best.pt` is rewritten when validation per-character accuracy improves;
    and one line of metrics -- including the confusable-set accuracy and the
    per-character breakdown -- goes to stdout and `metrics.jsonl`.

    Returns the path of `best.pt`, the checkpoint export_mrz_onnx.py consumes.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"
    metrics_path = checkpoint_dir / "metrics.jsonl"

    model = MrzCRNN(line_len=line_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    train_ds = MrzDiskDataset(train_dir)
    val_ds = MrzDiskDataset(val_dir, jitter_frac=0.0)
    severities = [float(r["severity"]) for r in train_ds.rows]
    val_loader = DataLoader[MrzSample](
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=num_workers,
    )

    start_epoch, best_char_acc = 0, -1.0
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_char_acc = int(state["epoch"]), float(state["best_char_accuracy"])
        print(f"RESUMING from {last_path}: completed {start_epoch}/{epochs} epochs, "
              f"best char_acc so far {best_char_acc:.4f}", flush=True)

    run_start = time.time()
    for epoch in range(start_epoch, epochs):
        if time_budget_s is not None and time.time() - run_start > time_budget_s:
            print(f"time budget {time_budget_s:.0f}s exhausted before epoch {epoch + 1}; "
                  "stopping early so the export still happens", flush=True)
            break

        lo, hi = severity_for_step(epoch, max(epochs - 1, 1), start=(0.0, 0.3), end=(0.0, 1.0))
        indices = [i for i, sv in enumerate(severities) if lo <= sv <= hi]
        loader = DataLoader[MrzSample](
            Subset(train_ds, indices), batch_size=batch_size, shuffle=True, collate_fn=partial(collate_batch, canvas_jitter=True),
            num_workers=num_workers, drop_last=True, pin_memory=device == "cuda",
        )

        model.train()
        epoch_start, loss_sum, n_batches = time.time(), 0.0, 0
        for step, batch in enumerate(loader):
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break
            images, targets = batch["images"].to(device), batch["targets"].to(device)
            log_probs = model(images)
            loss = criterion(log_probs.reshape(-1, log_probs.shape[-1]), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1

        val = evaluate(model, val_loader, device)
        improved = val.char_accuracy > best_char_acc
        seconds = round(time.time() - epoch_start, 1)
        record: dict[str, Any] = {
            "epoch": epoch + 1, "epochs": epochs, "severity_window": [round(lo, 3), round(hi, 3)],
            "train_samples": len(indices), "steps": n_batches,
            "loss": loss_sum / max(n_batches, 1),
            "char_acc": val.char_accuracy, "line_acc": val.line_accuracy,
            "confusable_acc": val.confusable_accuracy,
            "confusable_per_char": val.per_confusable_char,
            "best": improved, "seconds": seconds,
        }
        per_char = " ".join(f"{c}={a:.3f}" for c, a in val.per_confusable_char.items())
        marker = "*best*" if improved else ""
        print(
            f"epoch {epoch + 1}/{epochs}  loss={record['loss']:.4f}  window=[{lo:.2f},{hi:.2f}]  "
            f"char_acc={val.char_accuracy:.4f}  line_acc={val.line_accuracy:.4f}  "
            f"CONFUSABLE_ACC={val.confusable_accuracy:.4f}  {marker}  {seconds}s\n"
            f"    per-char: {per_char}",
            flush=True,
        )
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if improved:
            best_char_acc = val.char_accuracy
            _atomic_save(
                {"state_dict": model.state_dict(), "line_len": line_len, "epoch": epoch + 1,
                 "char_accuracy": val.char_accuracy, "line_accuracy": val.line_accuracy,
                 "confusable_accuracy": val.confusable_accuracy},
                best_path,
            )
        _atomic_save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "epoch": epoch + 1, "best_char_accuracy": best_char_acc},
            last_path,
        )

    if not best_path.exists():
        raise RuntimeError("training produced no checkpoint (zero epochs ran?)")
    return best_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=400, help="steps (batches) per epoch")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--line-len", type=int, default=44)
    parser.add_argument("--checkpoint-dir", type=str, default="models/mrz_crnn")
    parser.add_argument("--self-test", action="store_true", help="run a tiny CPU smoke test instead of real training")
    parser.add_argument("--disk", action="store_true", help="train from a pre-generated dataset (resumable, unattended)")
    parser.add_argument("--train-dir", type=str, default="data/mrz/train")
    parser.add_argument("--val-dir", type=str, default="data/mrz/val")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--time-budget-hours", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.disk:
        train_disk_curriculum(
            train_dir=args.train_dir, val_dir=args.val_dir, checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs, batch_size=args.batch, lr=args.lr, line_len=args.line_len,
            num_workers=args.workers, max_steps_per_epoch=args.max_steps_per_epoch,
            time_budget_s=args.time_budget_hours * 3600 if args.time_budget_hours else None,
        )
    elif args.self_test:
        print("train.py self-test: 2 tiny epochs on synthetic data, CPU, no real convergence expected")
        model = train(epochs=2, batch_size=4, steps_per_epoch=3, val_size=16, checkpoint_dir="_selftest_ckpt")
        assert any(p.requires_grad for p in model.parameters())
        import shutil

        shutil.rmtree("_selftest_ckpt", ignore_errors=True)
        print("train.py self-test OK (loop runs end to end; real training needs --epochs 30 --batch 64 --steps 400 on a GPU)")
    else:
        train(
            epochs=args.epochs, batch_size=args.batch, steps_per_epoch=args.steps,
            lr=args.lr, line_len=args.line_len, checkpoint_dir=args.checkpoint_dir,
        )
