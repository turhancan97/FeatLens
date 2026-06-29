"""Attention-rollout (CPU, random weights — no downloads)."""

from pathlib import Path

import numpy as np
import pytest

import featlens as ll

IMG = "examples/images/cat.jpg"


def test_attention_rollout_shape_and_range(tmp_path):
    res = ll.attention("vit_tiny_patch16_224", IMG, layer=-1, img_size=224,
                       pretrained=False, device="cpu", out=tmp_path / "att.png",
                       return_data=True)
    assert Path(res["path"]).exists()
    roll = res["rollout"]
    assert roll.shape == (14, 14)                       # 224 / 16
    assert roll.min() >= 0.0 and roll.max() <= 1.0
    assert roll.min() < 0.01 and roll.max() > 0.99      # normalized to ~[0, 1]


def test_attention_overlay_writes(tmp_path):
    out = ll.attention("vit_tiny_patch16_224", IMG, layer=4, img_size=224, overlay=True,
                       pretrained=False, device="cpu", out=tmp_path / "ov.png")
    assert Path(out).exists()


def test_attention_rejects_non_vit():
    # A callable/custom model is not a hook-mode ViT -> clear NotImplementedError.
    import torch.nn as nn
    from featlens import FeatureExtractor
    from featlens.adapters import custom_adapter

    lm = custom_adapter.load(nn.Conv2d(3, 16, 16, 16), patch_size=16,
                             feature_fn=lambda m, x: m(x))
    with pytest.raises(NotImplementedError):
        ll.attention(FeatureExtractor(lm), IMG, out=None)
