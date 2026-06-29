"""Regenerate the README cosine demo GIF (`examples/demo_cosine.gif`).

A seed patch sweeps across the image; the right panel is the cosine-similarity heatmap that tracks
it — now with the shared **[-1, 1] colorbar** that `cosine` mode renders since v0.2.5. Mirrors what
the live demo shows (`ll.visualize(..., method="cosine")`), reproduced as an animation so it can be
regenerated headlessly.

Run:  python examples/make_demo_gif.py
"""

import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # prefer the repo over any install

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

from featlens import FeatureExtractor, methods

HERE = Path(__file__).parent
IMAGE = HERE / "images" / "cat.jpg"
OUT = HERE / "demo_cosine.gif"

MODEL = "dinov2_vitb14"
IMG_SIZE = 448          # 448 / 14 = 32x32 feature grid -> a smooth heatmap
DISP = 384              # panel size in pixels
COLORMAP = "turbo"

# Seed keypoints (normalized x, y) visiting semantically distinct regions of the cat.
KEYPOINTS = [(0.62, 0.38), (0.40, 0.40), (0.50, 0.62), (0.33, 0.22),
             (0.15, 0.82), (0.80, 0.30), (0.62, 0.38)]
STEPS_BETWEEN = 4       # interpolated frames between keypoints
HOLD = 3                # frames held at each keypoint (a "click" pause)


def _seed_path():
    path = []
    for a, b in zip(KEYPOINTS, KEYPOINTS[1:]):
        for h in range(HOLD):
            path.append(a)
        for s in range(1, STEPS_BETWEEN + 1):
            t = s / (STEPS_BETWEEN + 1)
            path.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    path.extend([KEYPOINTS[-1]] * HOLD)
    return path


def _denorm_source(ex, pil):
    t = ex.transform(pil)
    img = torch.from_numpy(ex.denormalize(t)).permute(2, 0, 1).unsqueeze(0)
    img = F.interpolate(img, size=(DISP, DISP), mode="bilinear", align_corners=False)
    return img[0].permute(1, 2, 0).numpy()


def _heatmap(fb, seed):
    sim = methods.cosine_similarity_map(fb, seed)             # [h, w] in [-1, 1]
    rgb = methods.apply_colormap((sim + 1.0) / 2.0, COLORMAP)  # [h, w, 3]
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float()
    t = F.interpolate(t, size=(DISP, DISP), mode="bilinear", align_corners=False)
    return t[0].permute(1, 2, 0).numpy()


def _frame(src, fb, seed):
    sx, sy = seed[0] * DISP, seed[1] * DISP
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    axes[0].imshow(np.clip(src, 0, 1))
    axes[0].set_title("source (click to move seed)", fontsize=11)
    axes[1].imshow(np.clip(_heatmap(fb, seed), 0, 1))
    axes[1].set_title("cosine similarity", fontsize=11)
    for ax in axes:
        ax.scatter([sx], [sy], s=200, marker="*", c="white", edgecolors="black", linewidths=1.3)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    methods.cosine_colorbar(fig, [axes[1]], COLORMAP)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    ex = FeatureExtractor(MODEL, layers=[-1], img_size=IMG_SIZE)
    pil = Image.open(IMAGE).convert("RGB")
    feats = ex(ex.transform(pil).unsqueeze(0)).float().cpu()   # [1, 1, D, h, w]
    fb = rearrange(feats[0, 0], "d h w -> h w d")
    src = _denorm_source(ex, pil)

    frames = [_frame(src, fb, seed) for seed in _seed_path()]

    # Shrink for the README: downscale ~0.8x and quantize to a shared adaptive palette.
    w, h = frames[0].size
    target_w = 640
    if w > target_w:
        size = (target_w, round(h * target_w / w))
        frames = [f.resize(size, Image.LANCZOS) for f in frames]
    frames = [f.convert("P", palette=Image.ADAPTIVE, colors=192) for f in frames]

    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=180, loop=0,
                   optimize=True)
    kb = OUT.stat().st_size / 1024
    print(f"Wrote {OUT} ({len(frames)} frames, {kb:.0f} KB)")


if __name__ == "__main__":
    main()
