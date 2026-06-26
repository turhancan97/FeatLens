"""FeatLens quick-start — generates the README gallery.

Run:  python examples/quickstart.py
Uses the bundled example images; downloads small pretrained weights via timm/torchvision.
Outputs are written next to this file (the committed gallery the README displays).
"""

from pathlib import Path

import featlens as ll

HERE = Path(__file__).parent
IMAGES = HERE / "images"
NAMES = ["astronaut", "cat", "coffee"]

# 1) Per-image feature maps: one DINO ViT-B/16 row per image, across layers (shared basis).
for name in NAMES:
    ll.visualize("dino_vitb16", IMAGES / f"{name}.jpg", layers=[2, 5, 8, 11],
                 out=HERE / f"feat_{name}.png")

# 2) Compare models at the final layer (per-tile basis).
ll.compare(["dino_vitb16", "dinov2_vitb14", "clip_large_openai"], IMAGES / "cat.jpg",
           layer=-1, out=HERE / "compare_models.png")

# 3) Full model x layer grid, overlaid on the source image.
ll.grid(["dino_vitb16", "dinov2_vitb14"], IMAGES / "cat.jpg", layers=[2, 5, 8, 11],
        out=HERE / "grid_overlay.png", overlay=True)

# 4) Bring your own model (escape hatch): any nn.Module via a feature_fn or hook target.
import torch.nn as nn
import torchvision
from featlens import FeatureExtractor, FeatureGrid
from featlens.adapters import custom_adapter

resnet = torchvision.models.resnet50(weights="DEFAULT")
trunk = nn.Sequential(*list(resnet.children())[:-2])  # -> [B, 2048, h, w]
lm = custom_adapter.load(trunk, patch_size=32, feature_fn=lambda m, x: m(x), name="resnet50")
ll.FeatureGrid([FeatureExtractor(lm)]).render(IMAGES / "cat.jpg", out_path=HERE / "resnet50.png")

print(f"Wrote gallery to {HERE}")
