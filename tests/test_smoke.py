"""Lightweight smoke tests (CPU, random weights — no downloads)."""

import torch

from featlens import FeatureExtractor, FeatureGrid
from featlens.adapters import custom_adapter
from featlens.tokens import tokens_to_grid


def test_tokens_to_grid_strips_prefix():
    # [B, 1+N, D] with a CLS prefix -> [B, D, h, w], cls returned.
    B, h, w, D = 2, 4, 4, 8
    tokens = torch.randn(B, 1 + h * w, D)
    grid, cls = tokens_to_grid(tokens, B, h, w)
    assert grid.shape == (B, D, h, w)
    assert cls is not None and cls.shape == (B, D)


def test_tokens_to_grid_rank4_cnn_path():
    B, D, h, w = 1, 16, 7, 7
    grid, cls = tokens_to_grid(torch.randn(B, D, h, w), B, h, w)
    assert grid.shape == (B, D, h, w) and cls is None


def test_timm_extractor_multilayer():
    ex = FeatureExtractor("vit_tiny_patch16_224", layers=[2, 5, -1], img_size=224, pretrained=False)
    out = ex(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 3, ex.lm.embed_dim, 14, 14)


def test_custom_escape_hatch_callable():
    import torch.nn as nn

    model = nn.Conv2d(3, 32, kernel_size=16, stride=16)  # patchify -> [B, 32, 14, 14]
    lm = custom_adapter.load(model, patch_size=16, feature_fn=lambda m, x: m(x))
    out = FeatureExtractor(lm)(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 1, 32, 14, 14)


def test_resize_modes_yield_square():
    from PIL import Image
    from featlens.preprocess import build_transform

    img = Image.new("RGB", (640, 360))  # non-square
    for mode in ("squash", "crop", "pad"):
        t = build_transform(224, [0.5] * 3, [0.5] * 3, resize_mode=mode)(img)
        assert tuple(t.shape) == (3, 224, 224)


def test_img_size_changes_grid():
    ex = FeatureExtractor("vit_tiny_patch16_224", layers=[-1], img_size=384, pretrained=False)
    out = ex(torch.randn(1, 3, 384, 384))
    assert out.shape[-2:] == (24, 24)  # 384 / 16


def test_grid_render(tmp_path):
    out = FeatureGrid(["vit_tiny_patch16_224"], layers=[1, -1], pretrained=False).render(
        "examples/images/cat.jpg", out_path=tmp_path / "g.png"
    )
    assert (tmp_path / "g.png").exists()
