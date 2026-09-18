"""Training loop for MrzCRNN, with a severity curriculum.

Curriculum: early steps train mostly on lightly-degraded (easy) synthetic
lines so the model first learns the charset and basic shapes, then the
degradation severity range widens over the course of training so it also
learns to be robust to the print-scan noise, blur and skew it will see on
real captures. This is a training loop, not run as part of the module vendor
self-test suite (no GPU in this environment -- see BACKEND_BRIEF.md §1.3);
its self-test below runs a handful of CPU steps on a tiny synthetic dataset
just to prove the loop is wired correctly end to end.

Real training: `python -m app.ocr.mrz.train --epochs 30 --batch 64 --steps 400`
on a GPU box, per the brief. That produces the checkpoint infer.py loads.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.nn import CTCLoss
from torch.utils.data import DataLoader

from .data import SyntheticMrzLineDataset, WidthBucketedSampler, collate_batch
from .model import BLANK_IDX, MrzCRNN


def severity_for_step(step: int, total_steps: int, *, start: tuple[float, float] = (0.0, 0.3), end: tuple[float, float] = (0.1, 0.9)) -> tuple[float, float]:
    """Linear curriculum: severity range widens from `start` to `end` over training."""
    t = min(1.0, step / max(total_steps, 1))
    lo = start[0] + (end[0] - start[0]) * t
    hi = start[1] + (end[1] - start[1]) * t
    return (lo, hi)


def train(
    *,
    epochs: int = 30,
    batch_size: int = 64,
    steps_per_epoch: int = 400,
    lr: float = 3e-4,
    checkpoint_dir: str | Path = "models/mrz_crnn",
    device: str | None = None,
) -> MrzCRNN:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MrzCRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ctc_loss = CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    total_steps = epochs * steps_per_epoch
    global_step = 0

    for epoch in range(epochs):
        severity_range = severity_for_step(global_step, total_steps)
        dataset = SyntheticMrzLineDataset(size=steps_per_epoch * batch_size, base_seed=epoch, severity_range=severity_range)
        sampler = WidthBucketedSampler(dataset, batch_size=batch_size)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_batch)

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        for batch in loader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            log_probs = model(images)  # (B, T, C)
            input_lengths = torch.full(
                (images.shape[0],), log_probs.shape[1], dtype=torch.long, device=device
            )
            loss = ctc_loss(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        elapsed = time.time() - start_time
        print(
            f"epoch {epoch + 1}/{epochs}  loss={epoch_loss / max(n_batches, 1):.4f}  "
            f"severity={severity_range}  {elapsed:.1f}s"
        )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "model.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=400, help="steps (batches) per epoch")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="models/mrz_crnn")
    parser.add_argument("--self-test", action="store_true", help="run a tiny CPU smoke test instead of real training")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.self_test:
        print("train.py self-test: 2 tiny epochs on synthetic data, CPU, no real convergence expected")
        model = train(epochs=2, batch_size=4, steps_per_epoch=3, checkpoint_dir="_selftest_ckpt")
        assert any(p.requires_grad for p in model.parameters())
        import shutil

        shutil.rmtree("_selftest_ckpt", ignore_errors=True)
        print("train.py self-test OK (loop runs end to end; real training needs --epochs 30 --batch 64 --steps 400 on a GPU)")
    else:
        train(epochs=args.epochs, batch_size=args.batch, steps_per_epoch=args.steps, lr=args.lr, checkpoint_dir=args.checkpoint_dir)
