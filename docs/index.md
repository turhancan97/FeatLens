# FeatLens

🤗 **Try the [live demo](https://huggingface.co/spaces/turhancan97/FeatLens-demo)** — no install.

**See what any vision model encodes.** FeatLens renders **feature maps** for **any**
vision model — DINO, DINOv2/v3, CLIP, SigLIP, MAE, DeiT, V-JEPA, CNNs, … — loaded from **any**
source (timm, HuggingFace `transformers`, `torch.hub`, an external repo, or a model you built
yourself), and from **any layer**, as a clean **model × layer** grid. Color the features by robust
**PCA**, **cosine-similarity** to a seed patch, **k-means** segmentation, or a **foreground** mask
([methods](methods.md)) — and match patches **across two images**.

<p align="center">
  <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/feat_cat.png" alt="DINO feature maps across layers" width="100%">
</p>

Most "DINO PCA" scripts are welded to one model. FeatLens separates **representation access**
(a small adapter layer over the model zoo) from **visualization** (PCA / cosine / k-means /
foreground), so you can point it at a new model in seconds and compare models/layers side by side.

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

DINO ViT-B/16 feature maps across layers 2 / 5 / 8 / 11, on the bundled example images:

| Image (original size) | Source | Feature maps |
|---|---|---|
| `astronaut.jpg` · 512×512 | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/images/astronaut.jpg" width="100"> | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/feat_astronaut.png" width="420"> |
| `cat.jpg` · 451×300 | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/images/cat.jpg" width="100"> | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/feat_cat.png" width="420"> |
| `coffee.jpg` · 600×400 | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/images/coffee.jpg" width="100"> | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/feat_coffee.png" width="420"> |

**`grid(...)` — model × layer, overlaid on the image:**

<p align="center"><img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/grid_overlay.png" alt="model x layer grid overlay" width="100%"></p>

**`compare(...)` — models at the final layer** &nbsp;|&nbsp; **a ResNet-50 (CNN escape hatch):**

<p align="center">
  <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/compare_models.png" alt="compare models" height="300">
  &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/resnet50.png" alt="resnet50 feature map" height="300">
</p>
