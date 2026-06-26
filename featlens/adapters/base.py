"""Adapter base types.

Every backend (timm / hf / torchhub / external / custom) resolves a model spec into a
``LoadedModel`` descriptor. The extractor then drives all of them uniformly via one of
three modes:

- ``"hook"``          : register forward hooks on per-block modules (ViTs, CNNs, V-JEPA).
- ``"hidden_states"`` : run once with ``output_hidden_states=True`` and read the tuple (HF).
- ``"callable"``      : call a user ``feature_fn(model, images)`` (the escape hatch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch.nn as nn


@dataclass
class LoadedModel:
    model: nn.Module
    num_blocks: int
    embed_dim: int
    patch_size: int
    mean: Sequence[float]
    std: Sequence[float]
    mode: str  # "hook" | "hidden_states" | "callable"
    # mode="hook": maps (model, block_index) -> the submodule to hook.
    hook_module_fn: Optional[Callable[[nn.Module, int], nn.Module]] = None
    # mode="callable": feature_fn(model, images) -> [B, N, D] or [B, D, h, w].
    feature_fn: Optional[Callable] = None
    # mode="hidden_states": +1 offset so block index i -> hidden_states[i+1] (skip embeddings).
    hidden_states_offset: int = 1
    uses_temporal: bool = False  # V-JEPA 2.1 expects a time axis
    name: str = "model"
    extra: dict = field(default_factory=dict)


def infer_patch_size(model: nn.Module) -> Optional[int]:
    """Infer a square ViT patch size from a model's patch_embed, if present."""
    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is None:
        return None
    patch_size = getattr(patch_embed, "patch_size", None)
    if patch_size is not None:
        if isinstance(patch_size, (tuple, list)):
            if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
                raise ValueError(f"Only square patch sizes are supported, got {patch_size}.")
            return int(patch_size[0])
        return int(patch_size)
    proj = getattr(patch_embed, "proj", None)
    kernel_size = getattr(proj, "kernel_size", None)
    if kernel_size is not None:
        if isinstance(kernel_size, (tuple, list)):
            if len(kernel_size) != 2 or kernel_size[0] != kernel_size[1]:
                raise ValueError(f"Only square patch sizes are supported, got {kernel_size}.")
            return int(kernel_size[0])
        return int(kernel_size)
    return None


def real_index(idx: int, num_blocks: int) -> int:
    """Resolve a possibly-negative block index against the block count."""
    r = idx if idx >= 0 else num_blocks + idx
    if not (0 <= r < num_blocks):
        raise ValueError(f"Layer index {idx} out of range for a model with {num_blocks} blocks.")
    return r
