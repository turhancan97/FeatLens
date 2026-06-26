"""timm backend: DINO/DINOv2/v3, CLIP, SigLIP, DeiT, MAE, supervised ViTs, ...

Loads via ``timm.create_model`` and hooks ``blocks[i].norm2`` (the normalized
representation), matching the proven ``FrozenBackbone`` timm path. The correct per-model
normalization is read straight from timm's data config — no guessing.
"""

from __future__ import annotations

import torch.nn as nn

from .base import LoadedModel, infer_patch_size, real_index
from ..preprocess import IMAGENET_MEAN, IMAGENET_STD


def load(identifier: str, img_size: int = 224, pretrained: bool = True) -> LoadedModel:
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "The timm backend requires timm. Install it with `pip install \"featlens[timm]\"`."
        ) from exc

    try:
        model = timm.create_model(identifier, pretrained=pretrained, img_size=img_size)
    except TypeError:
        # Some models don't accept img_size= (e.g. non-ViT); fall back to default.
        model = timm.create_model(identifier, pretrained=pretrained)
    model.eval()

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError(
            f"timm model '{identifier}' has no `.blocks` (not a standard ViT). "
            "Use the custom backend with an explicit hook target for this model."
        )
    num_blocks = len(blocks)
    embed_dim = int(getattr(model, "embed_dim", getattr(model, "num_features")))
    patch_size = infer_patch_size(model)
    if patch_size is None:
        raise ValueError(f"Could not infer patch size for timm model '{identifier}'.")

    try:
        cfg = timm.data.resolve_model_data_config(model)
        mean, std = tuple(cfg["mean"]), tuple(cfg["std"])
    except Exception:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    def hook_module(m: nn.Module, idx: int) -> nn.Module:
        return m.blocks[real_index(idx, num_blocks)].norm2

    return LoadedModel(
        model=model,
        num_blocks=num_blocks,
        embed_dim=embed_dim,
        patch_size=patch_size,
        mean=mean,
        std=std,
        mode="hook",
        hook_module_fn=hook_module,
        name=identifier,
    )
