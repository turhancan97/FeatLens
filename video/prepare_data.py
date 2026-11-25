"""Precompute real DINO features for the explainer scenes (run in an env with featlens + timm).

    python video/prepare_data.py

Writes video/assets/cat_dino.npz with, for DINO ViT-B/16 at 224px, last layer:
  image      [224, 224, 3] uint8  the model input (squash-resized), for on-screen display
  coords     [196, 3]      float  each patch projected on the top-3 robust PCA directions
  rgb_min/max [3]          float  robust color range (MAD outlier filtering, as FeatLens does)
  rgb_robust [14, 14, 3]   float  FeatLens's PCA->RGB map (robust range, clipped)
  rgb_naive  [14, 14, 3]   float  same projection but plain min/max range (outliers stretch it)
  outlier    [196]         bool   patches outside the robust range on any channel
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from featlens import FeatureExtractor
from featlens.pca import get_robust_pca

ROOT = Path(__file__).resolve().parent
IMAGE = ROOT.parent / "examples" / "images" / "cat_hires.jpg"


def main():
    ex = FeatureExtractor("dino_vitb16", layers=[-1], img_size=224)
    x = ex.load_images([IMAGE])                       # [1, 3, 224, 224]
    with torch.no_grad():
        fmap = ex(x)[0, 0]                            # [D, 14, 14]
    d, h, w = fmap.shape
    feats = fmap.permute(1, 2, 0).reshape(h * w, d).float()

    reduction, rgb_min, rgb_max = get_robust_pca(feats)
    coords = feats @ reduction                        # [196, 3]

    robust = ((coords - rgb_min) / (rgb_max - rgb_min + 1e-8)).clamp(0, 1)
    cmin, cmax = coords.min(0).values, coords.max(0).values
    naive = (coords - cmin) / (cmax - cmin + 1e-8)
    outlier = ((coords < rgb_min) | (coords > rgb_max)).any(1)

    image = (ex.denormalize(x[0]) * 255).clip(0, 255).astype(np.uint8)

    out = ROOT / "assets" / "cat_dino.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez(out, image=image, coords=coords.numpy(), rgb_min=rgb_min.numpy(),
             rgb_max=rgb_max.numpy(), rgb_robust=robust.reshape(h, w, 3).numpy(),
             rgb_naive=naive.reshape(h, w, 3).numpy(), outlier=outlier.numpy())
    Image.fromarray(image).save(ROOT / "assets" / "cat_224.png")
    print(f"wrote {out}: D={d}, grid={h}x{w}, outliers={int(outlier.sum())}/{h * w}")


if __name__ == "__main__":
    main()
