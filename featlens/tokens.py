"""Universal token/feature-map normalizer.

Turns whatever a layer emits — a token sequence ``[B, N, D]`` (optionally with
CLS / register prefix tokens) or a spatial map ``[B, D, h, w]`` — into a clean dense
grid ``[B, D, H_feat, W_feat]``. This is the single choke-point that makes the rest of
the framework model-agnostic (adapted from ``FrozenBackbone._extract_dense_maps_from_tokens``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def tokens_to_grid(
    token_tensor: torch.Tensor,
    batch: int,
    h_feat: int,
    w_feat: int,
    layer_idx: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Normalize a layer output to ``([B, D, h, w], cls_or_None)``.

    Accepts rank-3 token sequences ``[B, N, D]`` or rank-4 maps ``[B, D, h, w]``
    (CNNs / conv stages take this path for free). Prefix tokens (CLS, registers,
    distillation) are detected via ``N - h*w`` and stripped; the first prefix token
    is returned as the CLS token when present. The embed dim ``D`` is inferred.
    """
    if token_tensor.dim() == 4:
        if token_tensor.shape[0] != batch:
            raise ValueError(
                f"Layer {layer_idx}: expected batch {batch}, got {token_tensor.shape[0]}."
            )
        token_tensor = token_tensor.flatten(2).transpose(1, 2).contiguous()
    elif token_tensor.dim() != 3:
        raise ValueError(
            f"Layer {layer_idx}: expected a rank-3 [B,N,D] or rank-4 [B,D,h,w] tensor, "
            f"got shape {list(token_tensor.shape)}."
        )

    num_patches = h_feat * w_feat
    seq_len = int(token_tensor.shape[1])
    embed_dim = int(token_tensor.shape[2])
    num_prefix = seq_len - num_patches
    if num_prefix < 0:
        raise ValueError(
            f"Layer {layer_idx}: sequence length {seq_len} is smaller than the expected "
            f"patch count {num_patches} ({h_feat}x{w_feat}). Check patch_size / image size."
        )

    cls_token: Optional[torch.Tensor] = None
    if num_prefix > 0:
        cls_token = token_tensor[:, 0, :].detach()
        token_tensor = token_tensor[:, num_prefix:, :]

    grid = token_tensor.permute(0, 2, 1).reshape(batch, embed_dim, h_feat, w_feat)
    return grid, cls_token


def tokens_to_spatiotemporal(
    token_tensor: torch.Tensor,
    batch: int,
    n_temporal: int,
    h_feat: int,
    w_feat: int,
    layer_idx: int = 0,
) -> torch.Tensor:
    """Split a spatiotemporal token sequence into per-time-step grids.

    A temporal model (V-JEPA) emits ``N = n_temporal * h * w (+ prefix)`` tokens for a clip.
    ``tokens_to_grid`` would wrongly strip the extra time steps as "prefix"; this splits them
    instead, returning ``[n_temporal, B, D, h, w]`` (one dense grid per temporal token).
    """
    if token_tensor.dim() != 3:
        raise ValueError(
            f"Layer {layer_idx}: spatiotemporal reshape expects [B, N, D], got "
            f"{list(token_tensor.shape)}."
        )
    seq_len = int(token_tensor.shape[1])
    embed_dim = int(token_tensor.shape[2])
    per_step = h_feat * w_feat
    expected = n_temporal * per_step
    num_prefix = seq_len - expected
    if num_prefix < 0:
        raise ValueError(
            f"Layer {layer_idx}: sequence length {seq_len} < expected {expected} "
            f"(n_temporal={n_temporal}, {h_feat}x{w_feat}). Check num_frames / tubelet_size."
        )
    if num_prefix > 0:
        token_tensor = token_tensor[:, num_prefix:, :]
    # [B, T*h*w, D] -> [T, B, D, h, w]
    grids = token_tensor.reshape(batch, n_temporal, h_feat, w_feat, embed_dim)
    return grids.permute(1, 0, 4, 2, 3).contiguous()
