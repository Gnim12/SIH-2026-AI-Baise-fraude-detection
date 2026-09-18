"""CRNN + CTC recognizer for MRZ line images.

Architecture: a small convolutional stack downsamples height to 1 while
preserving width (so the model stays alignment-agnostic along the text axis),
then a bidirectional LSTM reads the resulting feature sequence, and a linear
head classifies each timestep over the MRZ charset plus one CTC blank symbol.

Not trained here -- GPU training is out of scope for this environment (see
BACKEND_BRIEF.md §1.3). `infer.py` runs in a clearly-labelled stub mode until
real weights exist (train.py, ~15k steps).
"""
from __future__ import annotations

import torch
from torch import nn

from .spec import MRZ_CHARSET

BLANK_IDX = len(MRZ_CHARSET)  # CTC blank appended after the real charset
NUM_CLASSES = len(MRZ_CHARSET) + 1


class _ConvBackbone(nn.Module):
    """Downsamples height to 1 in four stages while keeping width mostly intact."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),  # /2 height, /2 width

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),  # /4 height, /4 width

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # /8 height, /4 width

            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # /16 height, /4 width

            nn.Conv2d(512, 512, 2, stride=(2, 1), padding=(0, 1)), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            # height collapses to 1 for a typical 32px-tall input crop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MrzCRNN(nn.Module):
    """CRNN + CTC MRZ line recognizer.

    Input: (batch, 1, H, W) grayscale line crops, H expected to be 32px after
    letterboxing/resizing upstream (see data.py). Output: (batch, T, NUM_CLASSES)
    per-timestep log-probabilities over the MRZ charset + CTC blank.
    """

    def __init__(self, hidden_size: int = 256, rnn_layers: int = 2) -> None:
        super().__init__()
        self.backbone = _ConvBackbone()
        self.rnn = nn.LSTM(
            input_size=512, hidden_size=hidden_size, num_layers=rnn_layers,
            bidirectional=True, batch_first=True,
        )
        self.head = nn.Linear(hidden_size * 2, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)  # (B, C, 1, W')
        if features.shape[2] != 1:
            features = features.mean(dim=2, keepdim=True)
        features = features.squeeze(2).permute(0, 2, 1)  # (B, W', C)
        rnn_out, _ = self.rnn(features)  # (B, W', 2*hidden)
        logits = self.head(rnn_out)  # (B, W', NUM_CLASSES)
        return torch.log_softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = MrzCRNN()
    n_params = model.count_parameters()
    print(f"model.py: MrzCRNN has {n_params:,} trainable parameters "
          f"(brief targets ~8.2M for the trained checkpoint's exact config)")

    batch = torch.randn(4, 1, 32, 44 * 16)  # 4 line crops, OCR-B-ish char cell width
    with torch.no_grad():
        out = model(batch)
    print(f"model.py: forward pass output shape {tuple(out.shape)} "
          f"(batch, timesteps, {NUM_CLASSES} classes)")

    assert out.shape[0] == 4
    assert out.shape[2] == NUM_CLASSES
    # log_softmax rows must sum (in prob space) to ~1
    probs_sum = out.exp().sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-4)

    print("model.py self-test OK")
