"""Batch montage + return_data feature export (CPU, random weights — no downloads)."""

from pathlib import Path

import numpy as np
from PIL import Image

import featlens as ll

MODEL = "vit_tiny_patch16_224"


def _make_images(d: Path, names):
    d.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        Image.new("RGB", (48, 32), (i * 40 % 255, 90, 150)).save(d / name)
    return d


def test_batch_montage_writes_sheet(tmp_path):
    d = _make_images(tmp_path / "imgs", ["a.png", "b.png", "c.png"])
    sheet = tmp_path / "sheet.png"
    written = ll.batch(MODEL, d, tmp_path / "out", layers=[-1], pretrained=False, montage=sheet)
    assert len(written) == 3
    assert sheet.exists()


def test_return_features_shape_and_optional():
    res = ll.grid([MODEL], "examples/images/cat.jpg", layers=[2, -1], pretrained=False,
                  return_data=True, include_features=True)
    feats = res["features"]                      # [R, B, L, D, h, w]
    assert feats.shape[0] == 1 and feats.shape[2] == 2   # 1 model, 2 layers
    assert feats.shape[-2:] == (14, 14)          # 224 / 16

    res2 = ll.grid([MODEL], "examples/images/cat.jpg", layers=[-1], pretrained=False,
                   return_data=True)
    assert "features" not in res2               # off by default
