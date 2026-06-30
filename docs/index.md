# FeatLens

🤗 **Try the [live demo](https://huggingface.co/spaces/turhancan97/FeatLens-demo)** — no install.

**See what any vision model encodes.** FeatLens renders **feature maps** for **any**
vision model — DINO, DINOv2/v3, CLIP, SigLIP, MAE, DeiT, V-JEPA, CNNs, … — loaded from **any**
source (timm, HuggingFace `transformers`, `torch.hub`, an external repo, or a model you built
yourself), and from **any layer**, as a clean **model × layer** grid. Color the features by robust
**PCA**, **cosine-similarity** to a seed patch, **k-means** segmentation, a **foreground** mask, or
**saliency** ([methods](methods.md)) — match patches **across two images**, roll up a ViT's
**attention**, batch a folder, or sweep a **video** clip.

<p align="center">
  <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/feat_peacock.png" alt="DINO feature maps across layers" width="100%">
</p>

Most "DINO PCA" scripts are welded to one model. FeatLens separates **representation access**
(a small adapter layer over the model zoo) from **visualization** (PCA / cosine / k-means /
foreground / saliency / attention-rollout), so you can point it at a new model in seconds and
compare models/layers side by side.

```bash
pip install "featlens[timm]"
```

```python
import featlens as fl
fl.grid(["dino_vitb16", "dinov2_vitb14"], "img.jpg", layers=[2, 5, 8, 11], overlay=True, out="grid.png")
```

[Get started →](installation.md){ .md-button .md-button--primary }
[API reference →](api.md){ .md-button }

## Gallery

DINO ViT-S/8 @ 768px feature maps across layers 2 / 5 / 8 / 11 — a *patch-8* backbone at high
resolution gives a fine **96×96** grid, so thin structures stay crisp:

| Image (original size) | Source | Feature maps |
|---|---|---|
| `peacock.jpg` · 1600×1280 | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/images/peacock.jpg" width="100"> | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/feat_peacock.png" width="420"> |
| `cat_hires.jpg` · 1600×1200 | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/images/cat_hires.jpg" width="100"> | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/feat_cat_hires.png" width="420"> |
| `market.jpg` · 1600×1063 | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/images/market.jpg" width="100"> | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/feat_market.png" width="420"> |

**`grid(...)` — model × layer, overlaid on the image:**

<p align="center"><img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/grid_overlay.png" alt="model x layer grid overlay" width="100%"></p>

**`compare(...)` — models at the final layer** &nbsp;|&nbsp; **a ResNet-50 (CNN escape hatch):**

<p align="center">
  <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/compare_models.png" alt="compare models" height="300">
  &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/resnet50.png" alt="resnet50 feature map" height="300">
</p>

**Same scene, six ViT-B/16 backbones** — `market.jpg` at 1024px, **last-layer** features (a 64×64
grid), PCA→RGB per model. Architecture and patch size are fixed, so the differences are purely the
*training objective*: DINOv3 and DINO carve out smooth semantic regions, MAE stays low-frequency,
while SigLIP, supervised, and Perception Encoder encode much higher-frequency detail.

<p align="center"><img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/compare_b16_market.png" alt="six ViT-B/16 backbones compared on one image" width="100%"></p>

**`attention(...)` — attention-rollout for a timm ViT** (DINO ViT-B/16, overlaid) &nbsp;|&nbsp;
**`video(...)` — per-frame feature maps as a filmstrip:**

<p align="center">
  <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/attention_rollout.png" alt="attention rollout" height="240"><br>
  <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/video_filmstrip.png" alt="video filmstrip" width="100%">
</p>

A **temporal** model feeds the whole clip at once and splits the spatiotemporal tokens into
per-time-step grids — here **V-JEPA 2.1** (ViT-B/16) on a real cockatoo clip, last layer, with one
shared PCA basis across frames so the moving bird stays color-coherent ([`examples/vjepa_video.py`](https://github.com/turhancan97/FeatLens/blob/main/examples/vjepa_video.py)):

<p align="center"><img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/vjepa_video.png" alt="V-JEPA temporal filmstrip" width="100%"></p>
