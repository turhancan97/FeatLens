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

# 2) Compare models at the final layer (per-tile PCA basis). This is `compare(...)` laid out as a
#    horizontal row (the function stacks models vertically); cat_hires at 448px for a denser grid.
def _compare_models():
    import matplotlib.pyplot as plt
    from featlens import FeatureExtractor, methods
    models = [("dino_vitb16", "DINO"), ("dinov2_vitb14", "DINOv2"), ("clip_large_openai", "CLIP")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, (spec, label) in zip(axes.ravel(), models):
        ex = FeatureExtractor(spec, layers=[-1], img_size=448)
        feats = ex.forward(ex.load_images([IMAGES / "cat_hires.jpg"]))  # [1, 1, D, h, w]
        ax.imshow(methods.colorize(feats[0, 0].permute(1, 2, 0), "pca"), interpolation="nearest")
        ax.set_title(label, fontsize=14); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=1.2)
    fig.savefig(HERE / "compare_models.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

_compare_models()

# 2b) Same scene, six ViT-B/16 backbones: last-layer PCA maps at 1024px (a 64x64 grid). Same
#     architecture and patch size throughout, so the differences are purely the training objective.
def _compare_b16_market():
    import matplotlib.pyplot as plt
    from featlens import FeatureExtractor, methods
    models = [("dino_vitb16", "DINO"), ("dinov3_vitb16", "DINOv3"), ("mae_vitb16", "MAE"),
              ("siglip_vitb16", "SigLIP"), ("supervised_vitb16", "Supervised (AugReg)"),
              ("perception_encoder_vitb16", "Perception Encoder")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.4))
    for ax, (spec, label) in zip(axes.ravel(), models):
        ex = FeatureExtractor(spec, layers=[-1], img_size=1024)
        feats = ex.forward(ex.load_images([IMAGES / "market.jpg"]))  # [1, 1, D, h, w]
        ax.imshow(methods.colorize(feats[0, 0].permute(1, 2, 0), "pca"), interpolation="nearest")
        ax.set_title(label, fontsize=14); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=1.2)
    fig.savefig(HERE / "compare_b16_market.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

_compare_b16_market()

# 3) Full model x layer grid, overlaid on the source image. Again use the higher-resolution cat
#    and a larger input size so the overlaid tiles have more spatial detail.
ll.grid(["dino_vitb16", "dinov2_vitb14"], IMAGES / "cat_hires.jpg", layers=[2, 5, 8, 11],
        img_size=448, out=HERE / "grid_overlay.png", overlay=True)

# 3b) v0.2 methods on one DINOv2 row across layers: cosine / k-means / foreground / saliency.
#     Use the higher-resolution cat and a larger input so the patch-grid methods read more crisply.
ll.visualize("dinov2_vitb14", IMAGES / "cat_hires.jpg", layers=[2, 5, 8, 11],
             method="cosine", seed=(0.5, 0.45), img_size=448, out=HERE / "method_cosine.png")
ll.visualize("dinov2_vitb14", IMAGES / "cat_hires.jpg", layers=[2, 5, 8, 11],
             method="kmeans", k=6, img_size=448, out=HERE / "method_kmeans.png")
ll.visualize("dinov2_vitb14", IMAGES / "cat_hires.jpg", layers=[2, 5, 8, 11],
             method="foreground", img_size=448, out=HERE / "method_foreground.png")
ll.visualize("dinov2_vitb14", IMAGES / "cat_hires.jpg", layers=[2, 5, 8, 11],
             method="saliency", img_size=448, out=HERE / "method_saliency.png")

# 3c-att) Attention-rollout (timm ViT): where is the [CLS] token looking? (overlaid on the cat)
ll.attention("dino_vitb16", IMAGES / "cat_hires.jpg", layer=-1, img_size=448, overlay=True,
             out=HERE / "attention_rollout.png")

# 3c-vid) Multi-frame video: a synthetic horizontal "pan" across market.jpg -> filmstrip + GIF,
#         plus a side-by-side GIF of the input frame next to its (per-frame) DINOv2 feature map.
def _video_filmstrip():
    import tempfile
    import numpy as np
    from PIL import Image
    src = Image.open(IMAGES / "market.jpg").convert("RGB")
    W, H = src.size
    cw, n = int(W * 0.55), 6
    crops = []
    with tempfile.TemporaryDirectory() as d:
        for i in range(n):
            x = round(i * (W - cw) / (n - 1))
            crop = src.crop((x, 0, x + cw, H))
            crop.save(Path(d) / f"frame_{i:02d}.jpg")
            crops.append(crop)
        res = ll.video("dinov2_vitb14", d, layers=[-1], n_frames=n, method="pca", img_size=448,
                       out=HERE / "video_filmstrip.png", return_data=True)
    # The 2D path runs each frame independently, so input frame i lines up 1:1 with feature step i.
    feat = np.asarray(res["frames_rgb"])[-1]            # [n, disp, disp, 3] in [0, 1]
    disp = feat.shape[1]
    gap = np.ones((disp, 6, 3), np.float32)
    combo = []
    for i in range(feat.shape[0]):
        im = np.asarray(crops[i].resize((disp, disp), Image.LANCZOS), np.float32) / 255.0
        pair = np.concatenate([im, gap, feat[i]], axis=1)
        combo.append(Image.fromarray((np.clip(pair, 0, 1) * 255).astype(np.uint8)))
    combo[0].save(HERE / "video_filmstrip_compare.gif", save_all=True, append_images=combo[1:],
                  duration=250, loop=0, optimize=True)

_video_filmstrip()

# 3c) Cross-image correspondence: seed the real cat's eye, find the matching part in a
#     watercolor cat — DINOv2 features match the same semantic part across photo and illustration.
ll.correspond("dinov2_vitb14", IMAGES / "cat_hires.jpg", IMAGES / "cat_cartoon.jpg",
              layer=-1, seed=(0.40, 0.40), topk=3, out=HERE / "correspond.png")

# 4) Bring your own model (escape hatch): any nn.Module via a feature_fn or hook target.
import torch.nn as nn
import torchvision
from featlens import FeatureExtractor, methods
from featlens.adapters import custom_adapter

resnet = torchvision.models.resnet50(weights="DEFAULT")
trunk = nn.Sequential(*list(resnet.children())[:-2])  # -> [B, 2048, h, w]
lm = custom_adapter.load(trunk, patch_size=32, feature_fn=lambda m, x: m(x), name="resnet50")


def _resnet_gallery():
    import matplotlib.pyplot as plt
    from PIL import Image

    # Horizontal layout: images across the columns, source on the top row and the ResNet-50
    # layer -1 feature map below it.
    names = [("cat_hires", "Cat"), ("peacock", "Peacock"), ("market", "Market")]
    ex = FeatureExtractor(lm, img_size=768)
    fig, axes = plt.subplots(2, len(names), figsize=(13.2, 9.2))

    for c, (stem, label) in enumerate(names):
        img_path = IMAGES / f"{stem}.jpg"
        with Image.open(img_path).convert("RGB") as pil:
            src = ex.denormalize(ex.transform(pil))
        feats = ex.forward(ex.load_images([img_path]))  # [1, 1, D, h, w]
        rgb = methods.colorize(feats[0, 0].permute(1, 2, 0), "pca")

        axes[0, c].imshow(src)
        axes[0, c].set_title(label, fontsize=12)
        axes[1, c].imshow(rgb, interpolation="nearest")
        for r in range(2):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    axes[0, 0].set_ylabel("source", fontsize=12)
    axes[1, 0].set_ylabel("ResNet-50 layer -1", fontsize=12)
    fig.tight_layout(pad=1.0)
    fig.savefig(HERE / "resnet50.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


_resnet_gallery()

print(f"Wrote gallery to {HERE}")
