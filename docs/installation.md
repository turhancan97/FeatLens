# Installation

FeatLens is on PyPI:

```bash
pip install "featlens[timm]"          # timm backend (DINO, CLIP, SigLIP, DeiT, ...)
```

Model backends are optional extras — install only what you need:

| Extra | Enables |
|-------|---------|
| `featlens[timm]` | timm models (DINO, DINOv2/v3, CLIP, SigLIP, DeiT, …) |
| `featlens[hf]` | HuggingFace `transformers` models |
| `featlens[clip]` | `open_clip` |
| `featlens[all]` | all of the above |

```bash
pip install "featlens[all]"
```

!!! note "PyTorch"
    Install PyTorch for your platform/CUDA first — see [pytorch.org](https://pytorch.org).
    FeatLens depends on `torch` + `torchvision` but does not pin a CUDA build.

## From source

```bash
git clone https://github.com/turhancan97/FeatLens
cd FeatLens
pip install -e ".[timm]"
```
