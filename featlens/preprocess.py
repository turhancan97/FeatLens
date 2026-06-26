"""Image preprocessing: per-model normalization and transforms.

Unlike a training pipeline that fixes one normalization, FeatLens builds the transform
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


class _ResizeLongestPad:
    """Resize so the longest side == ``size`` (aspect preserved), then center-pad to a square."""

    def __init__(self, size: int, fill: int = 0):
        self.size = int(size)
        self.fill = fill

    def __call__(self, img):
        from torchvision.transforms import functional as TF
        from torchvision.transforms import InterpolationMode

        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        img = TF.resize(img, [nh, nw], interpolation=InterpolationMode.BICUBIC)
        pad_l = (self.size - nw) // 2
        pad_t = (self.size - nh) // 2
        pad_r = self.size - nw - pad_l
        pad_b = self.size - nh - pad_t
        return TF.pad(img, [pad_l, pad_t, pad_r, pad_b], fill=self.fill)


def build_transform(img_size: int, mean: Sequence[float], std: Sequence[float],
                    resize_mode: str = "squash"):
    """Build a preprocessing transform.

    ``resize_mode``:
      - ``"squash"`` (default): resize to ``img_size x img_size`` (may distort aspect ratio).
      - ``"crop"``  : resize the shortest side to ``img_size``, then center-crop (aspect preserved).
      - ``"pad"``   : resize the longest side to ``img_size``, then pad to a square (whole image kept).
    All modes yield a square ``img_size x img_size`` tensor (so the feature grid stays square).
    """
    from torchvision import transforms as T

    bicubic = T.InterpolationMode.BICUBIC
    if resize_mode == "squash":
        resize = [T.Resize((img_size, img_size), interpolation=bicubic)]
    elif resize_mode == "crop":
        resize = [T.Resize(img_size, interpolation=bicubic), T.CenterCrop(img_size)]
    elif resize_mode == "pad":
        resize = [_ResizeLongestPad(img_size)]
    else:
        raise ValueError(f"Unknown resize_mode '{resize_mode}'. Use 'squash', 'crop', or 'pad'.")

    return T.Compose(resize + [T.ToTensor(), T.Normalize(mean=list(mean), std=list(std))])


def denormalize(image: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Undo normalization on a ``[3, H, W]`` tensor -> ``[H, W, 3]`` float array in [0, 1]."""
    mean_t = torch.tensor(list(mean)).view(3, 1, 1)
    std_t = torch.tensor(list(std)).view(3, 1, 1)
    img = (image.detach().cpu() * std_t + mean_t).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()
