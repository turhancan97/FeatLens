"""Friendly model-name registry and model-spec resolution.

A *spec* is what the user passes for a model. It can be:

- a **friendly name** in ``BACKBONE_REGISTRY`` (e.g. ``"dinov2_vitb14"``),
- a **raw timm id** (e.g. ``"vit_base_patch16_224"``) — treated as timm by default,
- a **V-JEPA name** in ``VJEPA_ENTRYPOINTS`` — routed to torch.hub,
- or an **explicit ``backend:ident``** prefix to force a backend:
  ``"timm:..."``, ``"hf:facebook/dinov2-base"``, ``"hub:facebookresearch/...,entry"``.

Registry seeded from ``adaptive_multi_layer_dense_fusion``'s ``FrozenBackbone``.
"""

from __future__ import annotations

from typing import Tuple

# Friendly name -> timm identifier.
BACKBONE_REGISTRY = {
    "dinov3_vitl16": "vit_large_patch16_dinov3.lvd1689m",
    "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
    "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dino_vitb16": "vit_base_patch16_224.dino",
    "dino_vits16": "vit_small_patch16_224.dino",
    "mae_vitl16": "vit_large_patch16_224.mae",
    "mae_vitb16": "vit_base_patch16_224.mae",
    "supervised_vitl16": "vit_large_patch16_224.augreg_in21k_ft_in1k",
    "supervised_vitb16": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "deit3_small": "deit3_small_patch16_224.fb_in1k",
    "deit3_base": "deit3_base_patch16_224.fb_in1k",
    "deit3_large": "deit3_large_patch16_224.fb_in1k",
    "clip_large_openai": "vit_large_patch14_clip_224.openai",
    "clip_large_laion": "vit_large_patch14_clip_224.laion400m_e32",
    "siglip_vitl16": "vit_large_patch16_siglip_256.v2_webli",
    "siglip_vitb16": "vit_base_patch16_siglip_256.v2_webli",
    "perception_encoder_vitl14": "vit_pe_spatial_large_patch14_448.fb",
    "perception_encoder_vitb16": "vit_pe_spatial_base_patch16_512.fb",
    "perception_encoder_vits16": "vit_pe_spatial_small_patch16_512.fb",
    "eva02_small_patch14": "eva02_small_patch14_224.mim_in22k",
    "eva02_base_patch14": "eva02_base_patch14_224.mim_in22k",
    "eva02_large_patch14": "eva02_large_patch14_224.mim_in22k",
    "samvit_base": "samvit_base_patch16.sa1b",
    "beit_base_patch16": "beit_base_patch16_224.in22k_ft_in22k_in1k",
    # V-JEPA names (routed to torch.hub, not timm).
    "vjepa2_vitl16": "vjepa2_vit_large",
    "vjepa2_1_vitb16": "vjepa2_1_vit_base_384",
    "vjepa2_1_vitl16": "vjepa2_1_vit_large_384",
}

VJEPA_ENTRYPOINTS = {
    "vjepa2_vitl16": "vjepa2_vit_large",
    "vjepa2_1_vitb16": "vjepa2_1_vit_base_384",
    "vjepa2_1_vitl16": "vjepa2_1_vit_large_384",
}

_BACKENDS = {"timm", "hf", "hub", "external", "custom"}


def resolve_spec(spec: str) -> Tuple[str, str]:
    """Resolve a model spec to ``(backend, identifier)``.

    Explicit ``backend:ident`` wins. Otherwise V-JEPA names route to ``hub``,
    registry names map to their timm id, and anything else is treated as a raw timm id.
    """
    if ":" in spec:
        head, rest = spec.split(":", 1)
        if head in _BACKENDS:
            return head, rest
    if spec in VJEPA_ENTRYPOINTS:
        return "hub", spec
    if spec in BACKBONE_REGISTRY:
        return "timm", BACKBONE_REGISTRY[spec]
    return "timm", spec
