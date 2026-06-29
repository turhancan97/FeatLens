"""torch.hub backend (V-JEPA 2 / 2.1).

Builds the encoder architecture from the official hub repo, then loads weights itself
with a robust checkpoint parser (digs through ``target_encoder``/``ema_encoder``/``encoder``
nesting and strips ``module./backbone./encoder.`` prefixes). Hooks each transformer block.
Adapted from ``FrozenBackbone``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from .base import LoadedModel, real_index

VJEPA_ENTRYPOINTS = {
    "vjepa2_vitl16": "vjepa2_vit_large",
    "vjepa2_1_vitb16": "vjepa2_1_vit_base_384",
    "vjepa2_1_vitl16": "vjepa2_1_vit_large_384",
}
_OFFICIAL_CKPT = {
    "vjepa2_vit_large": "vitl.pt",
    "vjepa2_1_vit_base_384": "vjepa2_1_vitb_dist_vitG_384.pt",
    "vjepa2_1_vit_large_384": "vjepa2_1_vitl_dist_vitG_384.pt",
}
_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
_V21_TYPES = {"vjepa2_1_vitb16", "vjepa2_1_vitl16"}
_DEFAULT_HUB_REF = "204698b45b3712590f06245fbfba32d3be539812"


def _is_state_dict(payload) -> bool:
    return isinstance(payload, dict) and any(
        isinstance(v, (torch.Tensor, nn.Parameter)) for v in payload.values()
    )


def _clean_state_dict(state_dict: Dict[str, object]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    prefixes = ("module.", "backbone.", "encoder.", "target_encoder.", "ema_encoder.")
    for key, val in state_dict.items():
        if not isinstance(val, (torch.Tensor, nn.Parameter)):
            continue
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if new_key.startswith(p):
                    new_key = new_key[len(p):]
                    changed = True
        cleaned[new_key] = val.detach().cpu()
    return cleaned


def _extract_encoder_state(payload) -> Dict[str, torch.Tensor]:
    if _is_state_dict(payload):
        cleaned = _clean_state_dict(payload)
        if cleaned:
            return cleaned
    if isinstance(payload, dict):
        for key in ("target_encoder", "ema_encoder", "encoder", "state_dict", "model"):
            nested = payload.get(key)
            if _is_state_dict(nested):
                cleaned = _clean_state_dict(nested)
                if cleaned:
                    return cleaned
            if isinstance(nested, dict) and _is_state_dict(nested.get("state_dict")):
                cleaned = _clean_state_dict(nested["state_dict"])
                if cleaned:
                    return cleaned
    raise ValueError(
        "Unknown V-JEPA checkpoint schema. Expected a raw encoder state_dict or a "
        "checkpoint containing target_encoder / ema_encoder / encoder."
    )


def load(
    identifier: str,
    img_size: int = 256,
    pretrained: bool = True,
    hub_ref: str = _DEFAULT_HUB_REF,
    ckpt_path: str = "",
    num_frames: int = 64,
    tubelet_size: int = 2,
    patch_size: int = 16,
) -> LoadedModel:
    # identifier may be a friendly name or a raw entrypoint.
    friendly = identifier
    entrypoint = VJEPA_ENTRYPOINTS.get(identifier, identifier)
    hub_repo = f"facebookresearch/vjepa2:{(hub_ref or 'main').strip()}"
    try:
        loaded = torch.hub.load(
            hub_repo, entrypoint, pretrained=False,
            patch_size=patch_size, num_frames=num_frames, tubelet_size=tubelet_size,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load V-JEPA via torch.hub (repo={hub_repo!r}, entry={entrypoint!r})."
        ) from exc
    encoder = loaded[0] if isinstance(loaded, (tuple, list)) else loaded

    if ckpt_path:
        payload = torch.load(Path(ckpt_path).expanduser(), map_location="cpu")
        encoder.load_state_dict(_extract_encoder_state(payload), strict=False)
    elif pretrained:
        url = f"{_BASE_URL}/{_OFFICIAL_CKPT[entrypoint]}"
        payload = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        encoder.load_state_dict(_extract_encoder_state(payload), strict=False)
    encoder.eval()

    embed_dim = int(getattr(encoder, "embed_dim", getattr(encoder, "num_features", 0)))
    num_blocks = len(getattr(encoder, "blocks", []))
    if embed_dim <= 0 or num_blocks <= 0:
        raise ValueError(f"Could not infer embed_dim/depth for V-JEPA '{identifier}'.")

    def hook_module(m: nn.Module, idx: int) -> nn.Module:
        return m.blocks[real_index(idx, num_blocks)]

    # V-JEPA hooks may emit tuples; the extractor's hook handles that.
    return LoadedModel(
        model=encoder,
        num_blocks=num_blocks,
        embed_dim=embed_dim,
        patch_size=patch_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        mode="hook",
        hook_module_fn=hook_module,
        uses_temporal=friendly in _V21_TYPES,
        name=identifier,
        extra={"num_frames": num_frames, "tubelet_size": tubelet_size},
    )
