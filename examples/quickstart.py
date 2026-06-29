"""FeatLens quick-start — generates the README gallery.

Run:  python examples/quickstart.py
Uses the bundled example images; downloads small pretrained weights via timm/torchvision.
Outputs are written next to this file (the committed gallery the README displays).
"""

from pathlib import Path

import featlens as ll

HERE = Path(__file__).parent
IMAGES = HERE / "images"

# 1) Per-image feature maps (the README hero rows). A *patch-8* DINO ViT-S at 768px gives a fine
#    96x96 feature grid, so thin structures (whiskers, feather barbs, individual fruit) stay crisp.
#    CPU-friendly (~30s/image); raise img_size for an even finer grid if you have the compute.
HERO_NAMES = ["peacock", "cat_hires", "market"]
for name in HERO_NAMES:
    ll.visualize("timm:vit_small_patch8_224.dino", IMAGES / f"{name}.jpg", layers=[2, 5, 8, 11],
                 img_size=768, out=HERE / f"feat_{name}.png")

# 2) Compare models at the final layer (per-tile basis).
ll.compare(["dino_vitb16", "dinov2_vitb14", "clip_large_openai"], IMAGES / "cat.jpg",
           layer=-1, out=HERE / "compare_models.png")

# 3) Full model x layer grid, overlaid on the source image.
ll.grid(["dino_vitb16", "dinov2_vitb14"], IMAGES / "cat.jpg", layers=[2, 5, 8, 11],
        out=HERE / "grid_overlay.png", overlay=True)

# 3b) v0.2 methods on one DINOv2 row across layers: cosine / k-means / foreground.
ll.visualize("dinov2_vitb14", IMAGES / "cat.jpg", layers=[2, 5, 8, 11],
             method="cosine", seed=(0.5, 0.45), out=HERE / "method_cosine.png")
ll.visualize("dinov2_vitb14", IMAGES / "cat.jpg", layers=[2, 5, 8, 11],
             method="kmeans", k=6, out=HERE / "method_kmeans.png")
ll.visualize("dinov2_vitb14", IMAGES / "cat.jpg", layers=[2, 5, 8, 11],
             method="foreground", out=HERE / "method_foreground.png")

# 3c) Cross-image correspondence: seed a patch in cat, find it in coffee.
ll.correspond("dinov2_vitb14", IMAGES / "cat.jpg", IMAGES / "coffee.jpg",
              layer=-1, seed=(0.5, 0.45), topk=3, out=HERE / "correspond.png")

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
