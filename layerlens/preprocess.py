"""Image preprocessing: per-model normalization and transforms.

Unlike a training pipeline that fixes one normalization, LayerLens builds the transform
*per model* using the model's own expected mean/std (timm reports these exactly), so
SigLIP / CLIP / ImageNet checkpoints are all fed correctly without any runtime remap.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
SIGLIP_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
SIGLIP_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)

_NAMED = {
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "clip": (CLIP_MEAN, CLIP_STD),
    "siglip": (SIGLIP_MEAN, SIGLIP_STD),
}


def resolve_mean_std(norm) -> Tuple[Sequence[float], Sequence[float]]:
    """Resolve a normalization spec to (mean, std).

    ``norm`` may be a named string ("imagenet"/"clip"/"siglip"), a (mean, std) pair,
    or None (defaults to imagenet).
    """
    if norm is None:
        return IMAGENET_MEAN, IMAGENET_STD
    if isinstance(norm, str):
        if norm not in _NAMED:
            raise ValueError(f"Unknown normalization '{norm}'. Use one of {list(_NAMED)}.")
        return _NAMED[norm]
    if isinstance(norm, (tuple, list)) and len(norm) == 2:
        return [float(x) for x in norm[0]], [float(x) for x in norm[1]]
    raise ValueError(f"Unsupported normalization spec: {norm!r}")


def build_transform(img_size: int, mean: Sequence[float], std: Sequence[float]):
    from torchvision import transforms as T

    return T.Compose(
        [
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def denormalize(image: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Undo normalization on a ``[3, H, W]`` tensor -> ``[H, W, 3]`` float array in [0, 1]."""
    mean_t = torch.tensor(list(mean)).view(3, 1, 1)
    std_t = torch.tensor(list(std)).view(3, 1, 1)
    img = (image.detach().cpu() * std_t + mean_t).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()
