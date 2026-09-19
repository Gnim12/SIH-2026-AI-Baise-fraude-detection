"""Training dataset: synthetic MRZ line crops + fixed-slot collate.

Each sample is one MRZ *line* (not a full 2/3-line MRZ), since the CRNN in
model.py reads one line at a time. Images are generated on the fly by
synth.py so there's no fixed-size dataset to ship -- an effectively unlimited
stream of fresh, print-scan-degraded synthetic lines with known ground truth.
`MrzDiskDataset` below reads a pre-generated set instead (scripts/gen_mrz_dataset.py)
so training doesn't pay synth.py's generation cost every epoch.

Every crop is normalised to the canonical (32, CANONICAL_WIDTH) canvas in
collate_batch, so batches are always the same shape. That makes
WidthBucketedSampler a no-op for padding purposes (all samples are already the
same width); it is retained only because train() still constructs it, and can
be deleted without effect.
"""
from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from . import synth
from .canonical import CANONICAL_HEIGHT, CANONICAL_WIDTH, normalize_line
from .spec import MRZ_CHARSET

CHAR_TO_IDX = {c: i for i, c in enumerate(MRZ_CHARSET)}
TARGET_HEIGHT = CANONICAL_HEIGHT


@dataclass
class MrzSample:
    image: np.ndarray  # (H, W) uint8
    text: str


class SyntheticMrzLineDataset(Dataset[MrzSample]):
    """Infinite-ish synthetic dataset: generates a fresh degraded MRZ line
    on every __getitem__, seeded deterministically from (base_seed, index)
    so a given index always reproduces the same sample within a run.
    """

    def __init__(
        self,
        size: int = 10_000,
        base_seed: int = 0,
        severity_range: tuple[float, float] = (0.1, 0.8),
        confusable_bias: float = 0.0,
        jitter_frac: float = 0.03,
    ):
        self.size = size
        self.base_seed = base_seed
        self.severity_range = severity_range
        self.confusable_bias = confusable_bias
        self.jitter_frac = jitter_frac

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> MrzSample:
        rng = random.Random(self.base_seed * 1_000_003 + index)
        record = synth.generate_record(rng, confusable_bias=self.confusable_bias)
        line_idx = rng.randrange(len(record.lines))
        line_text = record.lines[line_idx]

        severity = rng.uniform(*self.severity_range)
        clean = synth.render_mrz_lines([line_text], char_h=TARGET_HEIGHT)
        degraded = synth.degrade(clean, rng, severity=severity)
        degraded = horizontal_jitter(degraded, rng, max_frac=self.jitter_frac)
        return MrzSample(image=degraded, text=line_text)


def horizontal_jitter(image: np.ndarray, rng: random.Random, *, max_frac: float = 0.03) -> np.ndarray:
    """Shift the crop left/right by up to `max_frac` of its width, filling the
    exposed edge with background (255).

    Simulates detect.py handing back a band whose horizontal crop boundary is
    off by a few pixels -- a fixed-slot head has no CTC-style alignment
    tolerance built in, so slots need to see this drift during training to
    stay robust to it. Chosen over resizing/stretching because a real
    misdetected crop shifts content, it doesn't rescale it; a pure translation
    with background fill is the closest match to that failure mode without
    also fabricating a blur/scale artifact degrade() already covers.
    """
    w = image.shape[1]
    max_px = max(1, int(round(w * max_frac)))
    delta = rng.randint(-max_px, max_px)
    if delta == 0:
        return image
    shifted = np.full_like(image, 255)
    if delta > 0:
        shifted[:, delta:] = image[:, : w - delta]
    else:
        shifted[:, : w + delta] = image[:, -delta:]
    return shifted


def encode_text(text: str, line_len: int) -> torch.Tensor:
    if len(text) != line_len:
        raise ValueError(f"label length {len(text)} does not match line_len {line_len}: {text!r}")
    return torch.tensor([CHAR_TO_IDX[c] for c in text], dtype=torch.long)


def collate_batch(samples: list[MrzSample]) -> dict[str, torch.Tensor]:
    """Normalise every crop to the canonical (32, CANONICAL_WIDTH) canvas --
    the same `normalize_line` infer.py applies at inference, so train and
    serve see identical geometry -- and stack fixed-length per-slot integer
    targets, one row of `line_len` per sample.
    """
    line_len = len(samples[0].text)
    batch_images = torch.zeros(len(samples), 1, CANONICAL_HEIGHT, CANONICAL_WIDTH)
    for i, sample in enumerate(samples):
        batch_images[i, 0] = torch.from_numpy(normalize_line(sample.image)).float() / 255.0

    targets = torch.stack([encode_text(s.text, line_len) for s in samples])
    return {"images": batch_images, "targets": targets}


class WidthBucketedSampler(Sampler[list[int]]):
    """Groups sample indices into batches of similar rendered width.

    Widths aren't known without rendering, which is expensive to do twice, so
    this samples a pool of candidate indices, sorts by a cheap width proxy
    (text length, since degrade()'s resample only perturbs width by a bounded
    factor), and yields fixed-size batches from within that sorted pool --
    reducing padding waste without a separate preprocessing pass.
    """

    def __init__(self, dataset: SyntheticMrzLineDataset, batch_size: int, pool_factor: int = 8, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pool_size = batch_size * pool_factor
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for pool_start in range(0, len(indices), self.pool_size):
            pool = indices[pool_start : pool_start + self.pool_size]
            # Text length is a free width proxy: it's known before rendering
            # and rendered width is (line_length * char_w) up to the
            # bounded degrade() resample factor.
            pool.sort(key=lambda i: len(self.dataset[i].text))
            for batch_start in range(0, len(pool), self.batch_size):
                batch = pool[batch_start : batch_start + self.batch_size]
                if batch:
                    yield batch

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class MrzDiskDataset(Dataset[MrzSample]):
    """Reads a pre-generated dataset written by scripts/gen_mrz_dataset.py:
    one manifest.jsonl of {filename, line_index, label, severity, seed} rows
    beside the PNG crops, all under one split directory. Avoids paying
    synth.py's generation cost on every epoch once a fixed set exists.
    """

    def __init__(self, split_dir: str | Path, *, jitter_frac: float = 0.03, seed: int = 0):
        self.split_dir = Path(split_dir)
        manifest_path = self.split_dir / "manifest.jsonl"
        with manifest_path.open("r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        self.jitter_frac = jitter_frac
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> MrzSample:
        row = self.rows[index]
        image_path = self.split_dir / row["filename"]
        image = np.array(Image.open(image_path).convert("L"))
        if self.jitter_frac > 0.0:
            rng = random.Random(self.seed * 1_000_003 + index)
            image = horizontal_jitter(image, rng, max_frac=self.jitter_frac)
        return MrzSample(image=image, text=row["label"])


if __name__ == "__main__":
    dataset = SyntheticMrzLineDataset(size=64, base_seed=99)
    sample = dataset[0]
    print(f"data.py: single sample image shape {sample.image.shape}, text={sample.text!r}")
    assert sample.image.ndim == 2

    batch_samples = [dataset[i] for i in range(6)]
    widths = [s.image.shape[1] for s in batch_samples]
    print(f"data.py: raw widths before collate: {widths}")

    batch = collate_batch(batch_samples)
    print(f"data.py: collated batch images {tuple(batch['images'].shape)}, "
          f"targets shape={tuple(batch['targets'].shape)}")
    assert batch["images"].shape[0] == 6
    assert batch["images"].shape[2:] == (CANONICAL_HEIGHT, CANONICAL_WIDTH)
    assert batch["targets"].shape == (6, 44)

    # horizontal_jitter must stay within its stated bound.
    jitter_rng = random.Random(3)
    img = np.zeros((32, 700), dtype=np.uint8)
    for _ in range(50):
        out = horizontal_jitter(img, jitter_rng, max_frac=0.03)
        assert out.shape == img.shape
    print("data.py: horizontal_jitter shape-preserving OK")

    # encode_text must raise, not pad, on a length mismatch.
    try:
        encode_text("TOOSHORT", 44)
        raise AssertionError("encode_text should have raised on length mismatch")
    except ValueError:
        print("data.py: encode_text correctly raises on line_len mismatch")

    sampler = WidthBucketedSampler(SyntheticMrzLineDataset(size=40, base_seed=1), batch_size=8)
    batches = list(iter(sampler))
    print(f"data.py: width-bucketed sampler produced {len(batches)} batches "
          f"of sizes {[len(b) for b in batches]}")
    assert sum(len(b) for b in batches) == 40

    print("data.py self-test OK")
