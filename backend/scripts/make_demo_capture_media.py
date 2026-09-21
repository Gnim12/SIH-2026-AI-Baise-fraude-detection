"""DEMO-ONLY: a synthetic OVD tilt-sweep video and a synthetic live-face image, so an upload-only screening
has all three required artefacts (document still + sweep + face).

    python scripts/make_demo_capture_media.py --out scratch/demo_docs

Both are drawn/warped from scratch: no real person, no real document.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def make_sweep(doc_path: Path, out: Path, seconds: float = 5.0, fps: int = 20) -> int:
    doc = cv2.imread(str(doc_path))
    if doc is None:
        raise SystemExit(f"cannot read {doc_path}")
    h, w = doc.shape[:2]
    W, H = 960, 720
    scale = 0.62 * W / w
    doc = cv2.resize(doc, (int(w * scale), int(h * scale)))
    h, w = doc.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    n = int(seconds * fps)
    for i in range(n):
        t = i / (n - 1)
        tilt = math.sin(t * 2 * math.pi) * 0.22  # left-right tilt, ~+-25 degrees of apparent foreshortening
        dx, dy = W / 2 - w / 2, H / 2 - h / 2
        sq = tilt * h * 0.5
        dst = np.float32([[dx, dy + sq], [dx + w, dy - sq], [dx + w, dy + h + sq], [dx, dy + h - sq]])
        frame = np.full((H, W, 3), (58, 52, 46), np.uint8)
        warped = cv2.warpPerspective(doc, cv2.getPerspectiveTransform(src, dst), (W, H),
                                     borderMode=cv2.BORDER_TRANSPARENT, dst=frame)
        # moving specular band, the way a hologram catches light as the angle changes
        band = np.zeros((H, W), np.float32)
        cx = int(W * (0.2 + 0.6 * t))
        cv2.line(band, (cx - 60, 0), (cx + 60, H), 1.0, 46)
        band = cv2.GaussianBlur(band, (0, 0), 24)
        mask = (cv2.warpPerspective(np.full((h, w), 255, np.uint8), cv2.getPerspectiveTransform(src, dst), (W, H)) > 0)
        warped = warped.astype(np.float32) + (band[..., None] * 70 * mask[..., None])
        writer.write(np.clip(warped, 0, 255).astype(np.uint8))
    writer.release()
    return n


def make_face(out: Path) -> None:
    S = 640
    img = np.full((S, S, 3), (214, 208, 200), np.uint8)
    cv2.ellipse(img, (S // 2, S + 40), (230, 200), 0, 180, 360, (70, 82, 120), -1)         # shoulders
    cv2.rectangle(img, (S // 2 - 45, 400), (S // 2 + 45, 500), (150, 170, 205), -1)         # neck
    cv2.ellipse(img, (S // 2, 290), (135, 170), 0, 0, 360, (160, 185, 222), -1)             # face
    cv2.ellipse(img, (S // 2, 200), (145, 120), 0, 180, 360, (40, 48, 70), -1)              # hair
    for x in (-58, 58):
        cv2.ellipse(img, (S // 2 + x, 270), (26, 14), 0, 0, 360, (250, 250, 250), -1)
        cv2.circle(img, (S // 2 + x, 270), 10, (60, 45, 35), -1)
        cv2.line(img, (S // 2 + x - 30, 238), (S // 2 + x + 30, 232), (40, 48, 70), 6)
    cv2.line(img, (S // 2, 285), (S // 2 - 8, 345), (130, 155, 195), 4)
    cv2.ellipse(img, (S // 2, 385), (44, 18), 0, 10, 170, (90, 100, 170), 5)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    cv2.imwrite(str(out), img)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("scratch/demo_docs"))
    ap.add_argument("--doc", default="demo_clean_1_okonkwo.png")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    frames = make_sweep(a.out / a.doc, a.out / "demo_sweep.mp4")
    make_face(a.out / "demo_face.jpg")
    print(f"wrote demo_sweep.mp4 ({frames} frames) and demo_face.jpg in {a.out}")
