"""Attention-rollout — "where is the [CLS] token looking?".

Captures the per-block self-attention of a **timm ViT** by hooking ``blocks[i].attn.attn_drop``
(its *input* is the post-softmax attention ``[B, heads, N, N]``; we disable ``fused_attn`` so that
path runs), then composes them with Abnar & Zuidema rollout — ``Â = ½A + ½I``, row-normalize,
cumulative matrix product — and reads the CLS→patch row as a ``[h, w]`` heatmap.

Bounded to timm ViT-family (hook-mode, non-temporal) models; raises a clear error otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from . import methods
from .adapters import real_index
from .extractor import FeatureExtractor

PathLike = Union[str, Path]


def _rollout(attn_layers, head: str, discard_ratio: float) -> np.ndarray:
    """Compose per-layer attention ``[1, heads, N, N]`` into a single ``[N, N]`` rollout matrix."""
    reduce = {"mean": lambda a: a.mean(0), "max": lambda a: a.max(0).values,
              "min": lambda a: a.min(0).values}
    if head not in reduce:
        raise ValueError(f"head must be one of {list(reduce)} (got {head!r}).")
    result = None
    for attn in attn_layers:
        A = reduce[head](attn[0].float())  # [N, N]
        if discard_ratio > 0:
            flat = A.flatten()
            n_zero = int(flat.numel() * discard_ratio)
            if n_zero > 0:
                _, idx = flat.topk(n_zero, largest=False)
                flat[idx] = 0.0
                A = flat.reshape(A.shape)
        N = A.shape[-1]
        A = A + torch.eye(N)
        A = A / A.sum(dim=-1, keepdim=True)
        result = A if result is None else A @ result
    return result.cpu().numpy()


def attention(
    model: Union[str, FeatureExtractor],
    image: PathLike,
    *,
    layer: int = -1,
    img_size: int = 224,
    head: str = "mean",
    discard_ratio: float = 0.0,
    colormap: str = "turbo",
    overlay: bool = False,
    overlay_alpha: float = 0.5,
    out: Optional[PathLike] = None,
    return_data: bool = False,
    pretrained: bool = True,
    device: Optional[str] = None,
    resize_mode: str = "squash",
):
    """Attention-rollout heatmap for a timm ViT, up to ``layer``.

    Returns the saved path (or the figure); with ``return_data=True`` a dict with the ``rollout``
    ``[h, w]`` map in ``[0, 1]``.
    """
    from PIL import Image

    ex = model if isinstance(model, FeatureExtractor) else FeatureExtractor(
        model, layers=[layer], img_size=img_size, pretrained=pretrained, resize_mode=resize_mode)
    if ex.lm.mode != "hook" or ex.lm.uses_temporal:
        raise NotImplementedError(
            "attention rollout supports timm ViT (hook-mode, non-temporal) models only.")
    blocks = getattr(ex.model, "blocks", None)
    if blocks is None:
        raise NotImplementedError("attention rollout needs a model with `.blocks` (a ViT).")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ex.model.to(dev)
    real = real_index(layer, len(blocks))

    captured, saved, handles = {}, {}, []

    def _mk(i):
        def hook(_m, inp, _out):
            captured[i] = inp[0].detach().cpu()
        return hook

    for i in range(real + 1):
        attn_mod = getattr(blocks[i], "attn", None)
        drop = getattr(attn_mod, "attn_drop", None)
        if attn_mod is None or drop is None:
            for h in handles:
                h.remove()
            raise NotImplementedError(
                "attention rollout needs `blocks[i].attn.attn_drop` (standard timm ViT Attention).")
        if hasattr(attn_mod, "fused_attn"):
            saved[i] = attn_mod.fused_attn
            attn_mod.fused_attn = False
        handles.append(drop.register_forward_hook(_mk(i)))

    try:
        pil = Image.open(image).convert("RGB")
        t = ex.transform(pil).unsqueeze(0).to(dev)
        with torch.no_grad():
            ex.model(t)
    finally:
        for h in handles:
            h.remove()
        for i, v in saved.items():
            blocks[i].attn.fused_attn = v

    roll = _rollout([captured[i] for i in range(real + 1)], head, discard_ratio)  # [N, N]
    h_feat = w_feat = img_size // ex.patch_size
    N = roll.shape[-1]
    num_prefix = N - h_feat * w_feat
    # CLS row over patch columns; if there's no prefix/CLS, fall back to mean attention received.
    vec = roll[0, num_prefix:] if num_prefix > 0 else roll.mean(axis=0)
    heat = vec.reshape(h_feat, w_feat)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    composed = _compose(ex, pil, heat, colormap, overlay, overlay_alpha, layer, out)
    if not return_data:
        return composed
    return {"rollout": heat, "layer": layer,
            "path": composed if out else None, "fig": None if out else composed}


def _compose(ex, pil, heat, colormap, overlay, overlay_alpha, layer, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = 384
    src = torch.from_numpy(ex.denormalize(ex.transform(pil))).permute(2, 0, 1).unsqueeze(0)
    src = F.interpolate(src, size=(size, size), mode="bilinear", align_corners=False)[0]
    src = src.permute(1, 2, 0).numpy()

    heat_rgb = methods.apply_colormap(heat, colormap)
    heat_up = F.interpolate(
        torch.from_numpy(np.ascontiguousarray(heat_rgb)).permute(2, 0, 1).unsqueeze(0).float(),
        size=(size, size), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
    right = (1 - overlay_alpha) * src + overlay_alpha * heat_up if overlay else heat_up

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(np.clip(src, 0, 1)); axes[0].set_title("source", fontsize=11)
    axes[1].imshow(np.clip(right, 0, 1)); axes[1].set_title(f"attention rollout (→ layer {layer})",
                                                            fontsize=11)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    methods.scalar_colorbar(fig, [axes[1]], colormap, vmin=0.0, vmax=1.0,
                            label="attention (rollout)", ticks=[0.0, 0.5, 1.0])
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return str(out)
    return fig
