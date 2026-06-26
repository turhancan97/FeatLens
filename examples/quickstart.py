"""LayerLens quick-start — three ways to look at feature maps.

Run:  python examples/quickstart.py
(uses the bundled example images; downloads small pretrained weights via timm).
"""

from pathlib import Path

import layerlens as ll

HERE = Path(__file__).parent
IMG = HERE / "images" / "cat.jpg"
OUT = HERE / "out"

# 1) One model, scrub layers (shared PCA basis -> colors comparable across the row).
ll.visualize("dino_vitb16", IMG, layers=[2, 5, 8, 11], out=OUT / "dino_layers.png")

# 2) Compare models at the final layer (per-tile basis).
ll.compare(["dino_vitb16", "dinov2_vitb14", "clip_large_openai"], IMG, layer=-1,
           out=OUT / "compare_models.png")

# 3) Full model x layer grid, overlaid on the source image.
ll.grid(["dino_vitb16", "dinov2_vitb14"], IMG, layers=[2, 5, 8, 11],
        out=OUT / "grid_overlay.png", overlay=True)

# 4) Bring your own model (escape hatch): any nn.Module via a feature_fn or a hook target.
import torch.nn as nn
import torchvision
from layerlens import FeatureExtractor
from layerlens.adapters import custom_adapter

resnet = torchvision.models.resnet50(weights="DEFAULT")
trunk = nn.Sequential(*list(resnet.children())[:-2])  # -> [B, 2048, h, w]
lm = custom_adapter.load(trunk, patch_size=32, feature_fn=lambda m, x: m(x), name="resnet50")
ll.FeatureGrid([FeatureExtractor(lm)]).render(IMG, out_path=OUT / "resnet50.png")

print(f"Wrote visualizations to {OUT}")
