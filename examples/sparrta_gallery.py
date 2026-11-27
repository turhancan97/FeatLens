"""The 13 backbones of the SpaRRTa benchmark on its Bridge and City scenes.

SpaRRTa (Kargin et al., IJCV 2026) tests whether vision foundation models encode spatial relations
between objects. This script shows the last-layer patch features of the 13 frozen backbones it
evaluates, as robust-PCA RGB maps, on two of its scenes. For each backbone one PCA basis is fitted
jointly on both scenes, so colors carry over between the Bridge and City figures of the same model.

8 backbones load by name from the FeatLens registry; VGGT, SPA, CroCo, CroCo v2 and MaskFeat use the
loaders in examples/backbones/ (VGGT and CroCo need their repositories cloned):

    VGGT_REPO=/path/to/vggt CROCO_REPO=/path/to/croco python examples/sparrta_gallery.py

Writes examples/sparrta_gallery_bridge.png and examples/sparrta_gallery_city.png.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from PIL import Image

from featlens import FeatureExtractor
from featlens.pca import fit_pca_stats, get_pca_map

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "backbones"))
import load_croco, load_maskfeat, load_spa, load_vggt  # noqa: E402

SCENES = {"bridge": HERE / "images" / "sparrta_bridge.jpg", "city": HERE / "images" / "sparrta_city.jpg"}

# (title, how to load) in the order of SpaRRTa's results tables.
BACKBONES = [
    ("DINO B/16", "dino_vitb16"),
    ("DINOv2 B/14", "dinov2_vitb14"),
    ("DINOv2 reg. B/14", "dinov2_reg_vitb14"),
    ("DINOv2 reg. L/14", "dinov2_reg_vitl14"),
    ("DINOv3 B/16", "dinov3_vitb16"),
    ("VGGT L/14", lambda s: load_vggt.load(img_size=s)),
    ("SPA B/16", lambda s: load_spa.load(img_size=s)),
    ("CroCo B/16", lambda s: load_croco.load(img_size=s, version="v1")),
    ("CroCo v2 B/16", lambda s: load_croco.load(img_size=s, version="v2")),
    ("CLIP B/16", "clip_base_laion"),
    ("DeiT III B/16", "deit3_base"),
    ("MAE B/16", "mae_vitb16"),
    ("MaskFeat B/16", lambda s: load_maskfeat.load(img_size=s)),
]


def pca_maps(spec, img_size: int, device: str) -> dict:
    """Last-layer robust-PCA RGB map per scene, with one basis fitted on both scenes."""
    model = spec if isinstance(spec, str) else spec(img_size)
    ex = FeatureExtractor(model, layers=[-1], img_size=img_size).to(device)
    with torch.no_grad():
        feats = ex(ex.load_images(list(SCENES.values())).to(device))[:, 0].float().cpu()  # [S, D, h, w]
    maps = rearrange(feats, "s d h w -> s h w d")
    stats = fit_pca_stats(maps)  # one basis over both scenes
    out = {name: get_pca_map(maps[i], pca_stats=stats) for i, name in enumerate(SCENES)}
    del ex, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def figure(scene: str, tiles: list, out: Path) -> None:
    cols = 7
    fig, axes = plt.subplots(2, cols, figsize=(cols * 2.2, 2 * 2.45))
    panels = [("input", np.asarray(Image.open(SCENES[scene]).convert("RGB")))] + tiles
    for ax, (title, img) in zip(axes.ravel(), panels):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.6)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--out-dir", default=str(HERE))
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    per_scene = {name: [] for name in SCENES}
    for title, spec in BACKBONES:
        print(f"{title} ...", flush=True)
        for scene, rgb in pca_maps(spec, args.img_size, device).items():
            per_scene[scene].append((title, rgb))
    for scene, tiles in per_scene.items():
        figure(scene, tiles, Path(args.out_dir) / f"sparrta_gallery_{scene}.png")


if __name__ == "__main__":
    main()
