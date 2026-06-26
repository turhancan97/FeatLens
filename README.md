# FeatLens

**See what any vision model encodes.** FeatLens renders PCA-to-RGB **feature maps** for
**any** vision model — DINO, DINOv2/v3, CLIP, SigLIP, MAE, DeiT, V-JEPA, CNNs, … — loaded from
**any** source (timm, HuggingFace `transformers`, `torch.hub`, an external repo, or a model you
built yourself), and from **any layer**, as a clean **model × layer** grid.

<p align="center">
  <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/feat_cat.png" alt="DINO feature maps across layers" width="100%">
</p>

Most "DINO PCA" scripts are welded to one model. FeatLens separates **representation access**
(a small adapter layer over the model zoo) from **visualization** (robust PCA → RGB), so you can
point it at a new model in seconds and compare models/layers side by side.

## Gallery

All produced by `examples/quickstart.py` on the three bundled images. Sizes below are the
**originals**; each image is resized to `img_size` (default 224) before the model.

**`visualize(...)` — DINO ViT-B/16 feature maps across layers 2 / 5 / 8 / 11:**

| Image (original size) | Source | Feature maps |
|---|---|---|
| `astronaut.jpg` · 512×512 | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/images/astronaut.jpg" width="110"> | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/feat_astronaut.png" width="430"> |
| `cat.jpg` · 451×300 | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/images/cat.jpg" width="110"> | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/feat_cat.png" width="430"> |
| `coffee.jpg` · 600×400 | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/images/coffee.jpg" width="110"> | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/feat_coffee.png" width="430"> |

**`grid(...)` — model × layer, overlaid on the image** (DINO vs DINOv2 across layers 2/5/8/11):

<p align="center"><img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/grid_overlay.png" alt="model x layer grid overlay" width="100%"></p>

**`compare(...)` — models at the final layer** &nbsp;|&nbsp; **`custom_adapter` — a ResNet-50 (CNN escape hatch)**

<p align="center">
  <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/compare_models.png" alt="compare models at last layer" height="320">
  &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/resnet50.png" alt="resnet50 feature map" height="320">
</p>

## Install

```bash
pip install -e ".[timm]"          # timm backend (DINO, CLIP, SigLIP, DeiT, ...)
# extras: [hf] transformers · [clip] open_clip · [all]
```

Install PyTorch for your platform first (https://pytorch.org).

## Quick start (Python)

```python
import featlens as ll

# One model, scrub layers (shared PCA basis -> colors comparable across the row)
ll.visualize("dinov2_vitb14", "img.jpg", layers=[2, 5, 8, 11], out="row.png")

# Compare models at the final layer (per-tile basis)
ll.compare(["dino_vitb16", "mae_vitb16", "clip_large_openai"], "img.jpg", layer=-1, out="cmp.png")

# Full model x layer grid, overlaid on the image
ll.grid(["dino_vitb16", "dinov2_vitb14"], "img.jpg", layers=[2, 5, 8, 11], overlay=True, out="grid.png")
```

## Quick start (CLI)

```bash
featlens --models dino_vitb16 clip_large_openai --layers 2 5 8 11 \
    --images examples/images/cat.jpg --mode grid --out out/grid.png
featlens --config configs/example.yaml --images examples/images/cat.jpg --out out/grid.png
```

## Image size & resizing

Images are resized to a square **`img_size` × `img_size`** before the model (default **224**).
`img_size` must be divisible by the model's patch size (multiples of 16 for patch-16 models,
14 for patch-14). Larger sizes give a finer feature grid at more compute:

```python
ll.visualize("dinov2_vitb14", "img.jpg", layers=[2, 5, 8, 11], img_size=448)   # 32x32 grid
```

For **non-square images**, choose how aspect ratio is handled with `resize_mode`:

| `resize_mode` | behavior |
|---------------|----------|
| `squash` (default) | resize straight to `img_size²` — may distort |
| `crop` | resize shortest side to `img_size`, center-crop — aspect preserved |
| `pad` | resize longest side to `img_size`, pad to square — keeps the whole image |

```python
ll.grid([...], "wide.jpg", resize_mode="crop")          # Python
```

```bash
featlens --models dino_vitb16 --images wide.jpg --resize-mode pad --img-size 448 --out g.png
```

(`FeatureGrid(interpolation_size=…)` is separate — it only upscales the rendered tiles, not the
model input.)

## Model sources

| Source | How to pass it | Needs |
|--------|----------------|-------|
| **timm** | friendly name (`dinov2_vitb14`) or raw id (`vit_base_patch16_224`) | `[timm]` |
| **HuggingFace** | `hf:facebook/dinov2-base` | `[hf]` |
| **torch.hub (V-JEPA)** | `vjepa2_vitl16` | network for weights |
| **External repo** (VGGT/SPA/…) | `external_adapter.load(repo_dir, builder, hook_target=…)` | the cloned repo |
| **Your own model** | `custom_adapter.load(model, feature_fn=…)` | — |

Friendly names (see `featlens/registry.py`) cover DINO, DINOv2/v3, CLIP, SigLIP, MAE, DeiT,
Perception Encoder and V-JEPA; any other timm id works directly.

## Layers

`layers=[2, 5, 8, 11]` selects **transformer block indices** (0-based, **negatives allowed**,
`-1` = last). The same convention holds across backends — for HuggingFace models FeatLens maps
block `i` to `hidden_states[i+1]` (skipping the embedding output) for you.

## Bring your own model

Anything that isn't built in works through the escape hatch — give a feature function or a hook
target. CNNs work for free (their conv map is already spatial):

```python
import torch.nn as nn, torchvision
from featlens import FeatureExtractor, FeatureGrid
from featlens.adapters import custom_adapter

resnet = torchvision.models.resnet50(weights="DEFAULT")
trunk = nn.Sequential(*list(resnet.children())[:-2])           # -> [B, 2048, h, w]
lm = custom_adapter.load(trunk, patch_size=32, feature_fn=lambda m, x: m(x), name="resnet50")
FeatureGrid([FeatureExtractor(lm)]).render("img.jpg", out_path="resnet50.png")
```

For a model in its own repo, `external_adapter.load(repo_dir, builder, hook_target="blocks")`
puts the repo on `sys.path`, builds the model, and hooks its blocks.

## How it works

1. **Adapters** resolve a spec → a `LoadedModel` and drive extraction in one of three modes:
   forward **hooks** on per-block modules (ViTs/CNNs/V-JEPA), HF **`output_hidden_states`**, or a
   user **callable**.
2. `tokens_to_grid` normalizes whatever a layer emits (`[B,N,D]` tokens with optional
   CLS/register prefixes, or `[B,D,h,w]` maps) into a dense `[B,D,h,w]` grid.
3. **Robust PCA** (median-absolute-deviation outlier filtering) projects features to RGB;
   `FeatureGrid` lays out the model × layer tiles with a per-tile or shared-per-model basis.

The extraction core adapts the `FrozenBackbone` pattern; the PCA is adapted from the SpaRRTa
feature-map script.

## Releasing

Releases publish to [PyPI](https://pypi.org/project/featlens/) automatically via
`.github/workflows/publish.yml` (PyPI **Trusted Publishing** — no API token stored in the repo).

One-time setup on PyPI: add a *trusted publisher* for the project (Account → Publishing) with
owner `turhancan97`, repository `FeatLens`, workflow `publish.yml`, environment `pypi`. PyPI
supports a *pending* publisher so the very first release can also go through Actions.

Then cut a release by pushing a tag:

```bash
# bump the version in pyproject.toml first, then:
git tag v0.1.0 && git push origin v0.1.0
```

The workflow builds the sdist + wheel, runs `twine check`, and uploads to PyPI.

## License

[MIT](LICENSE).
