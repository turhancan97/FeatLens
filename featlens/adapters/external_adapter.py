"""External-repo backend — models that live in their own (non-pip) codebase, e.g. VGGT / SPA.

This is a thin convenience over the custom backend: it puts a repo on ``sys.path`` (so its
imports resolve), calls a user ``builder()`` that returns the model, and then wires it up like
any custom model via a ``hook_target`` or ``feature_fn``.

Example::

    from featlens.adapters import external_adapter
    lm = external_adapter.load(
        repo_dir=os.environ["VGGT_REPO"],
        builder=lambda: build_my_vggt(),
        hook_target="aggregator.blocks",   # or feature_fn=...
        patch_size=14,
    )
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Optional, Sequence, Union

import torch.nn as nn

from .base import LoadedModel
from ..preprocess import IMAGENET_MEAN, IMAGENET_STD
from . import custom_adapter


def load(
    repo_dir: str,
    builder: Callable[[], nn.Module],
    *,
    patch_size: Optional[int] = None,
    hook_target: Optional[Union[str, Callable]] = None,
    feature_fn: Optional[Callable] = None,
    num_blocks: Optional[int] = None,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    name: str = "external",
) -> LoadedModel:
    repo_dir = os.path.expanduser(str(repo_dir or "")).strip()
    if not repo_dir or not os.path.isdir(repo_dir):
        raise FileNotFoundError(
            f"External repo dir not found: {repo_dir!r}. Pass the path (e.g. $VGGT_REPO) "
            "to the cloned model repository."
        )
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    model = builder()
    if not isinstance(model, nn.Module):
        raise TypeError(f"builder() must return an nn.Module, got {type(model).__name__}.")

    return custom_adapter.load(
        model, patch_size=patch_size, hook_target=hook_target, feature_fn=feature_fn,
        num_blocks=num_blocks, mean=mean, std=std, name=name,
    )
