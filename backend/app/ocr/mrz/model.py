"""Fixed-slot CRNN recognizer for MRZ line images.

Architecture: a small convolutional stack downsamples height to 1 while
preserving width, a bidirectional LSTM reads the resulting feature sequence,
then the time axis is pooled from its native length down to exactly
`line_len` slots and a linear head classifies each slot over the MRZ
charset. There is no CTC blank: the MRZ is a fixed-length, evenly-spaced
monospace grid defined by ICAO 9303 (44/36/30 chars for TD3/TD2/TD1), so the
alignment problem CTC exists to solve is already solved by the spec. Forcing
CTC onto a known-alignment domain only adds a variable-length-to-fixed-slot
collapse step that then has to be reinvented downstream (see decode.py).

Not trained here -- GPU training is out of scope for this environment (see
BACKEND_BRIEF.md §1.3). `infer.py` runs in a clearly-labelled stub mode until
real weights exist (train.py).
"""
from __future__ import annotations

import torch
from torch import nn

from .spec import MRZ_CHARSET

NUM_CLASSES = len(MRZ_CHARSET)  # no CTC blank in a fixed-slot formulation


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
        result: torch.Tensor = self.net(x)
        return result


class MrzCRNN(nn.Module):
    """Fixed-slot CRNN MRZ line recognizer.

    Input: (batch, 1, H, W) grayscale line crops, H expected to be 32px after
    letterboxing/resizing upstream (see data.py). Output: (batch, line_len,
    NUM_CLASSES) per-slot log-probabilities over the MRZ charset -- exactly
    the shape decode.decode_mrz expects, no collapse step needed.

    `line_len` defaults to 44 (TD3). Pass 36 or 30 for TD2 / TD1; the
    backbone and RNN are unchanged, only the adaptive pool target differs.
    """

    def __init__(self, hidden_size: int = 256, rnn_layers: int = 2, line_len: int = 44) -> None:
        super().__init__()
        self.line_len = line_len
        self.backbone = _ConvBackbone()
        self.rnn = nn.LSTM(
            input_size=512, hidden_size=hidden_size, num_layers=rnn_layers,
            bidirectional=True, batch_first=True,
        )
        self.time_pool = nn.AdaptiveAvgPool1d(line_len)
        self.head = nn.Linear(hidden_size * 2, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)  # (B, C, 1, W')
        if features.shape[2] != 1:
            features = features.mean(dim=2, keepdim=True)
        features = features.squeeze(2).permute(0, 2, 1)  # (B, W', C)
        rnn_out, _ = self.rnn(features)  # (B, W', 2*hidden)
        pooled = self.time_pool(rnn_out.permute(0, 2, 1))  # (B, 2*hidden, line_len)
        pooled = pooled.permute(0, 2, 1)  # (B, line_len, 2*hidden)
        logits = self.head(pooled)  # (B, line_len, NUM_CLASSES)
        return torch.log_softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = MrzCRNN()
    n_params = model.count_parameters()
    print(f"model.py: MrzCRNN has {n_params:,} trainable parameters")

    batch = torch.randn(4, 1, 32, 44 * 16)  # 4 line crops, OCR-B-ish char cell width
    with torch.no_grad():
        out = model(batch)
    print(f"model.py: forward pass output shape {tuple(out.shape)} "
          f"(batch, line_len, {NUM_CLASSES} classes)")

    assert out.shape == (4, 44, NUM_CLASSES)
    # log_softmax rows must sum (in prob space) to ~1
    probs_sum = out.exp().sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-4)

    for line_len in (36, 30):
        m = MrzCRNN(line_len=line_len)
        with torch.no_grad():
            o = m(torch.randn(2, 1, 32, line_len * 16))
        assert o.shape == (2, line_len, NUM_CLASSES), o.shape
        print(f"model.py: line_len={line_len} -> output shape {tuple(o.shape)} OK")

    print("model.py self-test OK")
