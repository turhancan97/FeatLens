# Model sources

A *spec* is what you pass for a model. It can be a friendly name, a raw timm id, a V-JEPA name,
or an explicit `backend:ident`.

| Source | How to pass it | Needs |
|--------|----------------|-------|
| **timm** | friendly name (`dinov2_vitb14`) or raw id (`vit_base_patch16_224`) | `[timm]` |
| **HuggingFace** | `hf:facebook/dinov2-base` | `[hf]` |
| **torch.hub (V-JEPA)** | `vjepa2_vitl16` | network for weights |
| **External repo** (VGGT/CroCo/…) | `external_adapter.load(repo_dir, builder, hook_target=…)` | the cloned repo |
| **Your own model** | `custom_adapter.load(model, feature_fn=…)` | — |

Friendly names (see [`featlens/registry.py`](https://github.com/turhancan97/FeatLens/blob/main/featlens/registry.py))
cover DINO, DINOv2/v3 (including DINOv2 with registers), CLIP, SigLIP, MAE, DeiT, EVA-02, BEiT, SAM,
Perception Encoder and V-JEPA;
any other timm id works directly.

## Bring your own model

Anything that isn't built in works through the **escape hatch** — give a feature function or a
hook target. CNNs work for free (their conv map is already spatial):

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

## Backbones from the SpaRRTa benchmark

[SpaRRTa](https://github.com/gmum/SpaRRTa) (Kargin et al., *International Journal of Computer Vision*,
2026, [doi:10.1007/s11263-026-03112-7](https://doi.org/10.1007/s11263-026-03112-7)) evaluates whether
vision foundation models encode spatial relations between objects. FeatLens can load all 13 backbones
it evaluates: 8 by name, and 5 through small loaders in
[`examples/backbones/`](https://github.com/turhancan97/FeatLens/tree/main/examples/backbones) that use
the adapters above, so FeatLens itself gains no new dependencies.

| Backbone | How to load it |
|---|---|
| DINO B/16 | `dino_vitb16` |
| DINOv2 B/14 | `dinov2_vitb14` |
| DINOv2 with registers B/14, L/14 | `dinov2_reg_vitb14`, `dinov2_reg_vitl14` (also `dinov2_reg_vits14`) |
| DINOv3 B/16 | `dinov3_vitb16` |
| CLIP B/16 (LAION-2B) | `clip_base_laion` |
| DeiT III B/16 | `deit3_base` |
| MAE B/16 | `mae_vitb16` |
| VGGT L/14 | `examples/backbones/load_vggt.py` (needs a clone of [facebookresearch/vggt](https://github.com/facebookresearch/vggt)) |
| SPA B/16 | `examples/backbones/load_spa.py` (weights from Hugging Face, no SPA code needed) |
| CroCo B/16, CroCo v2 B/16 | `examples/backbones/load_croco.py --version v1\|v2` (needs a clone of [naver/croco](https://github.com/naver/croco)) |
| MaskFeat B/16 | `examples/backbones/load_maskfeat.py` (OpenMMLab weights, no OpenMMLab install) |

Each loader returns a `LoadedModel`, so it works anywhere a model spec does:

```python
import sys; sys.path.insert(0, "examples/backbones")
import featlens as ll
import load_vggt                     # export VGGT_REPO=/path/to/vggt

lm = load_vggt.load()                # VGGT's per-image ViT-L/14 encoder, 24 blocks
ll.visualize(("vggt_l14", lm), "img.jpg", layers=[5, 11, 17, -1], out="vggt.png")
```

How each one is wired:

- **VGGT:** runs only the per-image DINOv2 encoder (`aggregator.patch_embed`, ViT-L/14 with
  registers) before VGGT's multi-view blocks, over its 24 blocks.
- **CroCo / CroCo v2:** the model takes an image *pair*, so the loader wraps the encoder alone
  (`_encode_image`) over its 12 `enc_blocks`; there is no CLS token.
- **SPA / MaskFeat:** both are standard ViT-B/16 encoders, so their released weights are loaded into
  FeatLens's own timm ViT-B/16 (MaskFeat's OpenMMLab key names are renamed to timm's) and read like
  any registry model. A strict key check makes sure every encoder weight is used.

All five are read at the same point as every timm model in FeatLens, the normalized representation
inside each block (`blocks[i].norm2`), so their maps are directly comparable with the registry
backbones. The [SpaRRTa blog post](https://sparrta.gmum.net/blog/one-sparrta-scene-through-the-eyes-of-13-vision-models/)
used SpaRRTa's own extraction (raw block outputs, and block 11 of 12 for MAE), so its maps differ in
places, most visibly for MAE and VGGT.

[`examples/sparrta_gallery.py`](https://github.com/turhancan97/FeatLens/blob/main/examples/sparrta_gallery.py)
renders all 13 on two SpaRRTa scenes (see the [Gallery](index.md#the-13-sparrta-backbones)).

## How it works

1. **Adapters** resolve a spec to a `LoadedModel` and drive extraction in one of three modes:
   forward **hooks** on per-block modules (ViTs/CNNs/V-JEPA), HF **`output_hidden_states`**, or a
   user **callable**.
2. `tokens_to_grid` normalizes whatever a layer emits (`[B,N,D]` tokens with optional
   CLS/register prefixes, or `[B,D,h,w]` maps) into a dense `[B,D,h,w]` grid.
3. **Robust PCA** (median-absolute-deviation outlier filtering) projects features to RGB;
   `FeatureGrid` lays out the model × layer tiles with a per-tile or shared-per-model basis.
