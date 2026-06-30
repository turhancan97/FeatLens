<p align="center">
  <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/assets/logo_with_text.png" alt="FeatLens" width="440">
</p>

<p align="center">
  <a href="https://github.com/turhancan97/FeatLens/actions/workflows/test.yml"><img src="https://github.com/turhancan97/FeatLens/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://pypi.org/project/featlens/"><img src="https://img.shields.io/pypi/v/featlens" alt="PyPI"></a>
  <a href="https://turhancan97.github.io/FeatLens/"><img src="https://img.shields.io/badge/docs-online-blue" alt="Docs"></a>
  <a href="https://huggingface.co/spaces/turhancan97/FeatLens-demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Spaces-yellow" alt="Hugging Face Space"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

📖 **Documentation:** https://turhancan97.github.io/FeatLens/ &nbsp;·&nbsp; 🤗 **Live demo:** https://huggingface.co/spaces/turhancan97/FeatLens-demo

**See what any vision model encodes.** FeatLens renders **feature maps** for
**any** vision model — DINO, DINOv2/v3, CLIP, SigLIP, MAE, DeiT, V-JEPA, CNNs, … — loaded from
**any** source (timm, HuggingFace `transformers`, `torch.hub`, an external repo, or a model you
built yourself), and from **any layer**, as a clean **model × layer** grid. Color the features by
robust **PCA**, **cosine-similarity** to a seed patch, **k-means** segmentation, or a **foreground**
mask — and match patches **across two images**.

<p align="center">
  <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/feat_peacock.png" alt="DINO feature maps across layers" width="100%">
</p>

Most "DINO PCA" scripts are welded to one model. FeatLens separates **representation access**
(a small adapter layer over the model zoo) from **visualization** (PCA / cosine / k-means /
foreground), so you can point it at a new model in seconds and compare models/layers side by side.

## <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/assets/icon-48.png" height="22" alt=""> Gallery

All produced by `examples/quickstart.py`. The per-image rows below use **DINO ViT-S/8 at 768px**
— a small *patch-8* backbone at high resolution gives a fine **96×96** feature grid, so thin
structures (whiskers, feather barbs, individual fruit) stay crisp. The model × layer, `compare`
and the compare / method figures further down use DINO ViT-B/16 at 448px for a denser feature grid.

**`visualize(...)` — DINO ViT-S/8 @ 768px, feature maps across layers 2 / 5 / 8 / 11:**

| Image (original size) | Source | Feature maps |
|---|---|---|
| `peacock.jpg` · 1600×1280 | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/images/peacock.jpg" width="110"> | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/feat_peacock.png" width="430"> |
| `cat_hires.jpg` · 1600×1200 | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/images/cat_hires.jpg" width="110"> | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/feat_cat_hires.png" width="430"> |
| `market.jpg` · 1600×1063 | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/images/market.jpg" width="110"> | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/feat_market.png" width="430"> |

**`grid(...)` — model × layer, overlaid on `cat_hires.jpg` at 448px** (DINO vs DINOv2 across layers 2/5/8/11):

<p align="center"><img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/grid_overlay.png" alt="model x layer grid overlay" width="100%"></p>

**`compare(...)` — models at the final layer on `cat_hires.jpg` at 448px:**

<p align="center"><img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/compare_models.png" alt="compare models at last layer" width="100%"></p>

**`custom_adapter` — a ResNet-50 (CNN escape hatch) across three images at 768px, each with its `layer -1` feature map:**

<p align="center"><img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/resnet50.png" alt="resnet50 feature map" width="100%"></p>

**Same scene, six ViT-B/16 backbones** — `market.jpg` at 1024px, **last-layer** features (a 64×64
grid), PCA→RGB per model. Architecture and patch size are held fixed, so the differences are purely
the *training objective*: DINOv3 and DINO carve the scene into smooth semantic regions, MAE stays
low-frequency, while SigLIP, supervised, and Perception Encoder encode much higher-frequency detail.

<p align="center"><img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/compare_b16_market.png" alt="six ViT-B/16 backbones compared on one image" width="100%"></p>

**Beyond PCA** — the same DINOv2 row on `cat_hires.jpg` at 448px, recolored by **cosine-similarity** to a seed patch,
**k-means** segmentation, a **foreground** mask, and **saliency** (activation magnitude) — across
layers 2 / 5 / 8 / 11:

| Method | Across layers |
|---|---|
| `cosine` (seed on the cat) | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/method_cosine.png" width="520"> |
| `kmeans` (k=6) | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/method_kmeans.png" width="520"> |
| `foreground` | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/method_foreground.png" width="520"> |
| `saliency` (activation magnitude) | <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/method_saliency.png" width="520"> |

**`correspond(...)` — seed a patch in image A, find the matches in image B.** Here the seed is on
a real cat's eye; DINOv2 features match the *same semantic part* on a watercolor cat, across the
photo→illustration domain gap:

<p align="center"><img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/correspond.png" alt="cross-image correspondence" height="300"></p>

**`attention(...)` — attention-rollout for a timm ViT.** Composing DINO ViT-B/16's self-attention
(Abnar–Zuidema) shows the `[CLS]` token concentrating on the cat — overlaid, with a `[0, 1]` scale:

<p align="center"><img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/attention_rollout.png" alt="attention rollout" height="300"></p>

**`video(...)` — per-frame feature maps over a clip, as a filmstrip** (+ an animated GIF). Here a
synthetic pan across `market.jpg`, with DINOv2 last-layer PCA features tracking the scene:

<p align="center"><img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/video_filmstrip.png" alt="video filmstrip" width="100%"></p>

Played back as the **input clip beside its feature map** (left: source frames, right: DINOv2 PCA):

<p align="center"><img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/video_filmstrip_compare.gif" alt="input clip next to DINOv2 feature map" width="60%"></p>

For a **temporal** model the whole clip is fed *once* and the spatiotemporal tokens are split back
into per-time-step grids. Here **V-JEPA 2.1** (ViT-B/16) on a real cockatoo clip, last layer, one
**shared PCA basis across frames** so the colors stay consistent and the bird (centre) reads as it
moves against the fixed perch — `python examples/vjepa_video.py`:

<p align="center"><img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/vjepa_video.png" alt="V-JEPA temporal filmstrip" width="100%"></p>

Played back as the **input clip beside its feature map** (left: source frames, right: V-JEPA PCA):

<p align="center"><img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/vjepa_video_compare.gif" alt="input video next to V-JEPA feature map" width="60%"></p>

## <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/assets/icon-48.png" height="22" alt=""> Install

```bash
pip install -e ".[timm]"          # timm backend (DINO, CLIP, SigLIP, DeiT, ...)
# extras: [hf] transformers · [clip] open_clip · [video] read .mp4 clips · [all]
```

Install PyTorch for your platform first (https://pytorch.org).

## <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/assets/icon-48.png" height="22" alt=""> Quick start (Python)

```python
import featlens as ll

# One model, scrub layers (shared PCA basis -> colors comparable across the row)
ll.visualize("dinov2_vitb14", "img.jpg", layers=[2, 5, 8, 11], out="row.png")

# Compare models at the final layer (per-tile basis)
ll.compare(["dino_vitb16", "mae_vitb16", "clip_large_openai"], "img.jpg", layer=-1, out="cmp.png")

# Full model x layer grid, overlaid on the image
ll.grid(["dino_vitb16", "dinov2_vitb14"], "img.jpg", layers=[2, 5, 8, 11], overlay=True, out="grid.png")

# Batch a whole folder -> one figure per image (+ a contact-sheet montage)
ll.batch("dino_vitb16", "photos/", "out/", layers=[2, 5, 8, 11], montage="sheet.png")

# Multi-frame video -> a filmstrip (frames x layers) + an animated GIF
ll.video("dinov2_vitb14", "clip.mp4", layers=[5, 11], n_frames=16, out="strip.png")  # needs featlens[video]
ll.video("vjepa2_1_vitb16", "clip.mp4", n_frames=16, out="strip.png")  # temporal: one clip, per-step grids

# Attention-rollout: where is the [CLS] token looking? (timm ViTs)
ll.attention("dino_vitb16", "img.jpg", layer=-1, overlay=True, out="attn.png")
```

## <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/assets/icon-48.png" height="22" alt=""> Quick start (CLI)

```bash
featlens --models dino_vitb16 clip_large_openai --layers 2 5 8 11 \
    --images examples/images/cat.jpg --mode grid --out out/grid.png
featlens --config configs/example.yaml --images examples/images/cat.jpg --out out/grid.png

# Batch: point --images at a folder (or glob) and --out-dir at an output folder
featlens --models dino_vitb16 --layers 2 5 8 11 --images photos/ --out-dir out/

# Video (filmstrip + GIF) and attention-rollout
featlens --mode video --models dinov2_vitb14 --images clip.mp4 --n-frames 16 --out strip.png
featlens --mode attention --models dino_vitb16 --images cat.jpg --overlay --out attn.png
```

## <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/assets/icon-48.png" height="22" alt=""> Image size & resizing

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

## <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/assets/icon-48.png" height="22" alt=""> Model sources

| Source | How to pass it | Needs |
|--------|----------------|-------|
| **timm** | friendly name (`dinov2_vitb14`) or raw id (`vit_base_patch16_224`) | `[timm]` |
| **HuggingFace** | `hf:facebook/dinov2-base` | `[hf]` |
| **torch.hub (V-JEPA)** | `vjepa2_vitl16` | network for weights |
| **External repo** (VGGT/SPA/…) | `external_adapter.load(repo_dir, builder, hook_target=…)` | the cloned repo |
| **Your own model** | `custom_adapter.load(model, feature_fn=…)` | — |

Friendly names (see `featlens/registry.py`) cover DINO, DINOv2/v3, CLIP, SigLIP, MAE, DeiT,
EVA-02, BEiT, SAM, Perception Encoder and V-JEPA; any other timm id works directly.

## <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/assets/icon-48.png" height="22" alt=""> Layers

`layers=[2, 5, 8, 11]` selects **transformer block indices** (0-based, **negatives allowed**,
`-1` = last). The same convention holds across backends — for HuggingFace models FeatLens maps
block `i` to `hidden_states[i+1]` (skipping the embedding output) for you.

## <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/assets/icon-48.png" height="22" alt=""> Visualization methods

Every method consumes the same dense feature stack, so it works on `grid` / `visualize` /
`compare` and across any layer:

<p align="center">
  <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/examples/demo_cosine.gif" alt="cosine similarity follows the clicked seed patch" width="70%"><br>
  <em><code>cosine</code> mode: the heatmap (right, with its [-1, 1] colorbar) tracks the seed patch (white star) — click anywhere on the <a href="https://huggingface.co/spaces/turhancan97/FeatLens-demo">🤗 live demo</a>.</em>
</p>

| `method` | shows | extra args |
|----------|-------|------------|
| `pca` (default) | robust PCA → RGB | `basis`, `remove_first_component` |
| `cosine` | cosine similarity to a **seed** patch (with a [-1, 1] colorbar) | `seed=(x, y)`, `colormap` |
| `kmeans` | unsupervised k-means segmentation (with a cluster legend) | `k` |
| `foreground` | fg/bg mask (first PCA component) | — |
| `saliency` | per-patch activation magnitude (with a [0, 1] colorbar) | `colormap` |

```python
fl.visualize("dino_vitb16", "img.jpg", layers=[2, 5, 8, 11], method="cosine", seed=(0.5, 0.5))
fl.compare(["dino_vitb16", "dinov2_vitb14"], "img.jpg", layer=-1, method="kmeans", k=8)

# Cross-image correspondence: multiple seeds, and mutual-NN to drop spurious matches
fl.correspond("dino_vitb16", "a.jpg", "b.jpg", seed=[(0.4, 0.5), (0.6, 0.3)], mutual=True, out="corr.png")

# Get the arrays back (no PNG round-trip): RGB tiles + the underlying scalar field
res = fl.visualize("dino_vitb16", "img.jpg", layers=[2, 5, 8, 11], method="cosine", return_data=True)
res["tiles"]    # [R, C, H, W, 3] rendered RGB     res["scalars"]  # [R, C, h, w] cosine sim in [-1, 1]
```

`seed` is **normalized** image coords `(x, y) ∈ [0, 1]` (resolution/model independent). Pass
`cache=True` to memoize extraction on disk (`$FEATLENS_CACHE_DIR`, else `~/.cache/featlens`) so
re-renders are instant. Try it in the browser on the
[**🤗 live demo**](https://huggingface.co/spaces/turhancan97/FeatLens-demo) (or run [`demo/`](demo/)
locally) — in `cosine` mode, click the image to move the seed. See the
[docs](https://turhancan97.github.io/FeatLens/methods/).

## <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/assets/icon-48.png" height="22" alt=""> Bring your own model

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

## <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/assets/icon-48.png" height="22" alt=""> How it works

1. **Adapters** resolve a spec → a `LoadedModel` and drive extraction in one of three modes:
   forward **hooks** on per-block modules (ViTs/CNNs/V-JEPA), HF **`output_hidden_states`**, or a
   user **callable**.
2. `tokens_to_grid` normalizes whatever a layer emits (`[B,N,D]` tokens with optional
   CLS/register prefixes, or `[B,D,h,w]` maps) into a dense `[B,D,h,w]` grid.
3. **Robust PCA** (median-absolute-deviation outlier filtering) projects features to RGB;
   `FeatureGrid` lays out the model × layer tiles with a per-tile or shared-per-model basis.

The extraction core adapts the `FrozenBackbone` pattern; the PCA is adapted from the SpaRRTa
feature-map script.

## <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/assets/icon-48.png" height="22" alt=""> Contributing

**Contributions are welcome — this is an open-source project and we're happy to accept and support them.** Whether it's a new model adapter, a visualization method, a bug fix, docs, or just a question, please jump in (see [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide):

- 🐛 **Found a bug or have an idea?** [Open an issue](https://github.com/turhancan97/FeatLens/issues) — bug reports, feature requests, and questions are all welcome.
- 🔧 **Want to send a change?** [Fork the repo](https://github.com/turhancan97/FeatLens/fork), create a branch, and [open a pull request](https://github.com/turhancan97/FeatLens/pulls). Small, focused PRs are easiest to review.
- ✅ **Before you push:** run `pytest -q` and, for docs changes, `mkdocs build --strict`. New behavior should come with a test; new models should be verified to load and forward.
- 💬 **Not sure where to start?** Open an issue describing what you'd like to do and we'll help you scope it.

By contributing you agree that your contributions are licensed under the project's [MIT License](LICENSE).

## <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/assets/icon-48.png" height="22" alt=""> Releasing

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

## <img src="https://raw.githubusercontent.com/turhancan97/FeatLens/main/assets/icon-48.png" height="22" alt=""> License

[MIT](LICENSE).
