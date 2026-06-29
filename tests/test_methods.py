"""v0.2 visualization methods — cosine / k-means / foreground / correspondence (CPU, random weights)."""

import numpy as np
import torch

import featlens as ll
from featlens import FeatureGrid
from featlens import methods


def _fmap(h=14, w=14, d=32):
    return torch.randn(h, w, d)


def test_colorize_shapes_and_range():
    fmap = _fmap()
    for method in ("pca", "cosine", "kmeans", "foreground"):
        rgb = methods.colorize(fmap, method, seed=(0.5, 0.5), k=5)
        assert rgb.shape == (14, 14, 3)
        assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_cosine_seed_is_max_similarity():
    fmap = _fmap()
    sim = methods.cosine_similarity_map(fmap, (0.5, 0.5))
    r, c = methods.seed_to_cell((0.5, 0.5), 14, 14)
    assert np.isclose(sim[r, c], 1.0, atol=1e-4)  # a patch is most similar to itself
    assert sim.min() >= -1.0001 and sim.max() <= 1.0001


def test_kmeans_label_count():
    labels = methods.kmeans_labels(_fmap(), k=5)
    assert labels.shape == (14, 14)
    assert labels.max() < 5 and labels.min() >= 0


def test_foreground_is_binary():
    rgb = methods.foreground_mask(_fmap())
    vals = np.unique(rgb)
    assert set(vals.tolist()).issubset({0.0, 1.0})


def test_grid_methods_render(tmp_path):
    for method in ("cosine", "kmeans", "foreground"):
        out = FeatureGrid(["vit_tiny_patch16_224"], layers=[2, -1], pretrained=False,
                          method=method, seed=(0.5, 0.5), device="cpu").render(
            "examples/images/cat.jpg", out_path=tmp_path / f"{method}.png")
        assert (tmp_path / f"{method}.png").exists()


def test_correspond_render(tmp_path):
    out = ll.correspond("vit_tiny_patch16_224", "examples/images/cat.jpg",
                        "examples/images/coffee.jpg", layer=-1, seed=(0.4, 0.5), topk=3,
                        pretrained=False, device="cpu", out=tmp_path / "corr.png")
    assert (tmp_path / "corr.png").exists()


def _has_colorbar(fig):
    # A colorbar adds an extra axes whose label matches; check for our cosine label.
    return any(getattr(ax, "get_ylabel", lambda: "")() == "cosine similarity"
               or getattr(ax, "get_xlabel", lambda: "")() == "cosine similarity"
               for ax in fig.axes)


def test_scales_present_only_where_meaningful():
    import matplotlib.pyplot as plt

    # cosine -> a [-1, 1] colorbar; pca/foreground -> none.
    fig = FeatureGrid(["vit_tiny_patch16_224"], layers=[-1], pretrained=False,
                      method="cosine", device="cpu").render("examples/images/cat.jpg")
    assert _has_colorbar(fig)
    plt.close(fig)

    fig = FeatureGrid(["vit_tiny_patch16_224"], layers=[-1], pretrained=False,
                      method="pca", device="cpu").render("examples/images/cat.jpg")
    assert not _has_colorbar(fig)
    plt.close(fig)

    # kmeans -> a legend with one entry per cluster.
    fig = FeatureGrid(["vit_tiny_patch16_224"], layers=[-1], pretrained=False,
                      method="kmeans", k=4, device="cpu").render("examples/images/cat.jpg")
    assert fig.legends and len(fig.legends[0].get_texts()) == 4
    plt.close(fig)
