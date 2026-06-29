"""return_data=True on the public renderers (CPU, random weights — no downloads)."""

import numpy as np

import featlens as ll

IMG = "examples/images/cat.jpg"
MODEL = "vit_tiny_patch16_224"


def test_grid_return_data_cosine_has_scalars(tmp_path):
    res = ll.grid([MODEL], IMG, layers=[2, -1], pretrained=False, method="cosine",
                  seed=(0.5, 0.5), out=tmp_path / "g.png", return_data=True)
    assert set(res) >= {"tiles", "scalars", "row_labels", "col_labels", "path"}
    assert res["tiles"].shape[:2] == (1, 2)          # [R, C, H, W, 3]
    assert res["tiles"].shape[-1] == 3
    assert res["scalars"].shape[:2] == (1, 2)        # cosine -> a scalar field per tile
    assert res["scalars"].min() >= -1.0001 and res["scalars"].max() <= 1.0001
    assert res["path"] == str(tmp_path / "g.png") and res["fig"] is None


def test_saliency_scalars_in_unit_range():
    res = ll.visualize(MODEL, IMG, layers=[-1], pretrained=False, method="saliency",
                       return_data=True)
    assert res["scalars"].min() >= 0.0 and res["scalars"].max() <= 1.0
    assert res["fig"] is not None and res["path"] is None  # no out= -> figure kept


def test_pca_and_kmeans_have_no_scalars():
    for method in ("pca", "kmeans"):
        res = ll.compare([MODEL], IMG, layer=-1, pretrained=False, method=method,
                         return_data=True)
        assert res["scalars"] is None


def test_return_data_off_returns_path(tmp_path):
    out = ll.grid([MODEL], IMG, layers=[-1], pretrained=False, out=tmp_path / "g.png")
    assert out == str(tmp_path / "g.png")  # unchanged default behavior
