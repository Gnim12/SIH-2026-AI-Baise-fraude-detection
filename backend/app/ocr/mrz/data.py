"""Training dataset: synthetic MRZ line crops + width-bucketed collate.

Each sample is one MRZ *line* (not a full 2/3-line MRZ), since the CRNN in
model.py reads one line at a time. Images are generated on the fly by
synth.py so there's no fixed-size dataset to ship -- an effectively unlimited
stream of fresh, print-scan-degraded synthetic lines with known ground truth.

Width-bucketed collate: line images vary in rendered width (MRZ length is
fixed at 30/36/44 chars, but degrade()'s resample step changes pixel width
slightly), so naive batching would pad every sample to the batch max, wasting
compute on batches that mix very different widths. Bucketing groups
similar-width samples together before padding.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from . import synth
from .model import BLANK_IDX
from .spec import MRZ_CHARSET

CHAR_TO_IDX = {c: i for i, c in enumerate(MRZ_CHARSET)}
TARGET_HEIGHT = 32


@dataclass
class MrzSample:
    image: np.ndarray  # (H, W) uint8
    text: str


class SyntheticMrzLineDataset(Dataset):
    """Infinite-ish synthetic dataset: generates a fresh degraded MRZ line
    on every __getitem__, seeded deterministically from (base_seed, index)
    so a given index always reproduces the same sample within a run.
    """

    def __init__(self, size: int = 10_000, base_seed: int = 0, severity_range: tuple[float, float] = (0.1, 0.8)):
        self.size = size
        self.base_seed = base_seed
        self.severity_range = severity_range

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> MrzSample:
        rng = random.Random(self.base_seed * 1_000_003 + index)
        record = synth.generate_record(rng)
        line_idx = rng.randrange(len(record.lines))
        line_text = record.lines[line_idx]

        severity = rng.uniform(*self.severity_range)
        clean = synth.render_mrz_lines([line_text], char_h=TARGET_HEIGHT)
        degraded = synth.degrade(clean, rng, severity=severity)
        return MrzSample(image=degraded, text=line_text)


def encode_text(text: str) -> torch.Tensor:
    return torch.tensor([CHAR_TO_IDX[c] for c in text], dtype=torch.long)


def _resize_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    if image.shape[0] == target_h:
        return image
    from PIL import Image

    scale = target_h / image.shape[0]
    new_w = max(1, int(round(image.shape[1] * scale)))
    pil = Image.fromarray(image).resize((new_w, target_h), Image.BILINEAR)
    return np.array(pil)


def collate_batch(samples: list[MrzSample]) -> dict[str, torch.Tensor]:
    """Pad a (width-bucketed) list of samples to the batch's max width."""
    resized = [_resize_to_height(s.image, TARGET_HEIGHT) for s in samples]
    max_w = max(img.shape[1] for img in resized)

    batch_images = torch.zeros(len(samples), 1, TARGET_HEIGHT, max_w)
    for i, img in enumerate(resized):
        w = img.shape[1]
        tensor = torch.from_numpy(img).float() / 255.0
        batch_images[i, 0, :, :w] = tensor

    targets = [encode_text(s.text) for s in samples]
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_flat = torch.cat(targets)
    input_widths = torch.tensor([img.shape[1] for img in resized], dtype=torch.long)

    return {
        "images": batch_images,
        "targets": targets_flat,
        "target_lengths": target_lengths,
        "input_widths": input_widths,
    }


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

    def __iter__(self):
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
          f"target_lengths={batch['target_lengths'].tolist()}")
    assert batch["images"].shape[0] == 6
    assert batch["images"].shape[2] == TARGET_HEIGHT
    assert batch["images"].shape[3] == max(widths)
    assert batch["targets"].numel() == sum(batch["target_lengths"].tolist())

    sampler = WidthBucketedSampler(SyntheticMrzLineDataset(size=40, base_seed=1), batch_size=8)
    batches = list(iter(sampler))
    print(f"data.py: width-bucketed sampler produced {len(batches)} batches "
          f"of sizes {[len(b) for b in batches]}")
    assert sum(len(b) for b in batches) == 40

    print("data.py self-test OK")
