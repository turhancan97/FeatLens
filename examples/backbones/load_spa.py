"""SPA (Zhu et al., 2025): feature maps from the SPA ViT-B/16 image encoder.

SPA pretrains a ViT with 3D-aware volumetric rendering. Its image encoder is a standard timm
ViT-B/16, so this loader puts the released weights (Hugging Face ``HaoyiZhu/SPA``,
``spa-b.safetensors``) into FeatLens's own timm ViT-B/16. Features are then read exactly like any
registry model, and no SPA code or 3D dependencies are needed. Inputs use ImageNet mean/std, as SPA.

    python examples/backbones/load_spa.py --image my.jpg
"""

from __future__ import annotations

from common import IMAGENET_MEAN, IMAGENET_STD, check_state_dict, cli, resample_pos_embed

from featlens.adapters import timm_adapter

PREFIX = "model.img_backbone."
# Parts of SPA's checkpoint that belong to its rendering decoder, not the image encoder
# (same filter as SPA's own load_pretrained).
SKIP = ("decoder", "head", "upsample", "mask", "readout")


def load(img_size: int = 224, size: str = "b"):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    raw = load_file(hf_hub_download(repo_id="HaoyiZhu/SPA", filename=f"spa-{size}.safetensors"))
    state = {k[len(PREFIX):]: v for k, v in raw.items()
             if k.startswith(PREFIX) and not any(s in k for s in SKIP)}

    arch = {"b": "vit_base_patch16_224", "l": "vit_large_patch16_224"}[size]
    lm = timm_adapter.load(arch, img_size=img_size, pretrained=False)
    resample_pos_embed(state, lm.model)
    check_state_dict(lm.model.load_state_dict(state, strict=False), allowed_missing=("head.",))
    lm.mean, lm.std, lm.name = IMAGENET_MEAN, IMAGENET_STD, f"spa_{size}16"
    return lm


if __name__ == "__main__":
    cli(load, "Feature maps from the SPA ViT image encoder.",
        extra_args=lambda p: p.add_argument("--size", choices=["b", "l"], default="b"))
