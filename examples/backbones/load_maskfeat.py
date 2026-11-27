"""MaskFeat (Wei et al., 2022): feature maps from the OpenMMLab MaskFeat ViT-B/16.

MaskFeat's released weights come from OpenMMLab (mmselfsup), which normally needs ``mmcls``/``mmcv``.
The backbone is a standard ViT-B/16, so this loader renames the OpenMMLab keys to timm's and loads them
into FeatLens's own timm ViT-B/16, with no OpenMMLab install. A strict key check guarantees that every
encoder weight is mapped. Inputs use ImageNet mean/std.

    python examples/backbones/load_maskfeat.py --image my.jpg
"""

from __future__ import annotations

import pickle
import re
import types

import torch

from common import IMAGENET_MEAN, IMAGENET_STD, check_state_dict, cli, download, resample_pos_embed

from featlens.adapters import timm_adapter

CHECKPOINT = ("https://download.openmmlab.com/mmselfsup/1.x/maskfeat/"
              "maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k/"
              "maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221101-6dfc8bf3.pth")

# OpenMMLab ViT key -> timm ViT key.
RENAMES = [
    (r"^patch_embed\.projection\.", "patch_embed.proj."),
    (r"^layers\.(\d+)\.ln1\.", r"blocks.\1.norm1."),
    (r"^layers\.(\d+)\.ln2\.", r"blocks.\1.norm2."),
    (r"^layers\.(\d+)\.attn\.", r"blocks.\1.attn."),
    (r"^layers\.(\d+)\.ffn\.layers\.0\.0\.", r"blocks.\1.mlp.fc1."),
    (r"^layers\.(\d+)\.ffn\.layers\.1\.", r"blocks.\1.mlp.fc2."),
    (r"^ln1\.", "norm."),
]


def to_timm(state: dict) -> dict:
    out = {}
    for key, value in state.items():
        if not key.startswith("backbone.") or key == "backbone.mask_token":
            continue  # target generator / neck / mask token are pretraining-only
        key = key[len("backbone."):]
        for pattern, repl in RENAMES:
            key = re.sub(pattern, repl, key)
        out[key] = value
    return out


class _Placeholder:
    """Stands in for OpenMMLab metadata classes so the checkpoint unpickles without mmengine."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type(name, (_Placeholder,), {})  # e.g. mmengine config/meta objects


_PICKLE = types.SimpleNamespace(Unpickler=_Unpickler, load=pickle.load, __name__="pickle")


def load(img_size: int = 224):
    # OpenMMLab checkpoints also pickle mmengine metadata objects, hence weights_only=False with an
    # unpickler that replaces those classes by placeholders (only load checkpoints you trust this way).
    ckpt = torch.load(download(CHECKPOINT, "maskfeat_vitb16.pth"), map_location="cpu",
                      weights_only=False, pickle_module=_PICKLE)
    state = to_timm(ckpt["state_dict"])

    lm = timm_adapter.load("vit_base_patch16_224", img_size=img_size, pretrained=False)
    resample_pos_embed(state, lm.model)
    check_state_dict(lm.model.load_state_dict(state, strict=False), allowed_missing=("head.",))
    lm.mean, lm.std, lm.name = IMAGENET_MEAN, IMAGENET_STD, "maskfeat_b16"
    return lm


if __name__ == "__main__":
    cli(load, "Feature maps from the MaskFeat ViT-B/16 (OpenMMLab weights, no OpenMMLab install).")
