"""Custom backend — the escape hatch for any model the built-in backends don't cover.

Two ways to use it:

1. **Hook target** — pass an already-built ``model`` plus ``hook_target``: either a callable
   ``(model, block_idx) -> module`` or the name of a ``ModuleList`` attribute (e.g. ``"blocks"``)
   whose elements are hooked.
2. **feature_fn** — pass ``feature_fn(model, images) -> [B, N, D]`` (or ``[B, D, h, w]``) and
   LayerLens skips hooks entirely. Best for exotic models / single-layer extraction.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import torch.nn as nn

from .base import LoadedModel, infer_patch_size, real_index
from ..preprocess import IMAGENET_MEAN, IMAGENET_STD


def load(
    model: nn.Module,
    *,
    patch_size: Optional[int] = None,
    hook_target: Optional[Union[str, Callable[[nn.Module, int], nn.Module]]] = None,
    feature_fn: Optional[Callable] = None,
    num_blocks: Optional[int] = None,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    name: str = "custom",
) -> LoadedModel:
    if (hook_target is None) == (feature_fn is None):
        raise ValueError("Provide exactly one of `hook_target` or `feature_fn`.")

    if patch_size is None:
        patch_size = infer_patch_size(model)
        if patch_size is None:
            raise ValueError(
                "Could not infer patch_size for the custom model; pass patch_size=... "
                "(the stride/downsample factor between input pixels and the feature grid)."
            )

    if feature_fn is not None:
        return LoadedModel(
            model=model, num_blocks=num_blocks or 1, embed_dim=0, patch_size=int(patch_size),
            mean=mean, std=std, mode="callable", feature_fn=feature_fn, name=name,
        )

    if callable(hook_target):
        hook_fn = hook_target
        nblocks = num_blocks or 0
    else:
        module_list = getattr(model, hook_target, None)
        if module_list is None or not hasattr(module_list, "__getitem__"):
            raise ValueError(
                f"hook_target '{hook_target}' is not a ModuleList attribute on the model."
            )
        nblocks = len(module_list)

        def hook_fn(m: nn.Module, idx: int) -> nn.Module:
            return getattr(m, hook_target)[real_index(idx, nblocks)]

    return LoadedModel(
        model=model, num_blocks=nblocks or 1, embed_dim=0, patch_size=int(patch_size),
        mean=mean, std=std, mode="hook", hook_module_fn=hook_fn, name=name,
    )
