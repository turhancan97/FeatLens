"""Video: temporal reshape (synthetic) + non-temporal per-frame filmstrip/GIF (CPU, no downloads)."""

from pathlib import Path

import torch
from PIL import Image

import featlens as ll
from featlens.tokens import tokens_to_spatiotemporal

MODEL = "vit_tiny_patch16_224"


def test_tokens_to_spatiotemporal_splits_time():
    B, Tt, h, w, D = 1, 4, 5, 5, 8
    # sequence with a CLS prefix: N = 1 + T*h*w
    tokens = torch.randn(B, 1 + Tt * h * w, D)
    st = tokens_to_spatiotemporal(tokens, B, Tt, h, w)
    assert st.shape == (Tt, B, D, h, w)
    # the prefix was dropped and step t maps to the right slice
    flat = tokens[:, 1:, :].reshape(B, Tt, h, w, D)
    assert torch.allclose(st[2, 0], flat[0, 2].permute(2, 0, 1))


def _frames(d: Path, n=5):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (40, 40), (i * 40 % 255, 100, 160)).save(d / f"f{i:02d}.png")
    return d


def test_video_non_temporal_filmstrip_and_gif(tmp_path):
    d = _frames(tmp_path / "frames", n=6)
    out = tmp_path / "strip.png"
    res = ll.video(MODEL, d, layers=[2, -1], n_frames=4, img_size=224, pretrained=False,
                   out=out, return_data=True)
    assert Path(res["path"]).exists()                 # filmstrip
    assert Path(res["gif"]).exists()                  # auto GIF next to it
    assert res["frames_rgb"].shape[:2] == (2, 4)      # [layers, frames, H, W, 3]


def test_video_missing_frames_errors(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    try:
        ll.video(MODEL, empty, pretrained=False, out=tmp_path / "x.png")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for an empty frame folder")
