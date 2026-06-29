"""Batch / directory mode (CPU, random weights — no downloads)."""

from pathlib import Path

import pytest
from PIL import Image

import featlens as ll
from featlens.batch import _expand_images


def _make_images(d: Path, names):
    d.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        Image.new("RGB", (48, 32), (i * 30 % 255, 80, 160)).save(d / name)
    return d


def test_expand_dir_sorts_and_filters(tmp_path):
    d = _make_images(tmp_path / "imgs", ["b.png", "a.jpg", "c.PNG"])
    (d / "notes.txt").write_text("not an image")
    out = _expand_images(d)
    assert [p.name for p in out] == ["a.jpg", "b.png", "c.PNG"]  # sorted, .txt dropped


def test_expand_glob_and_list(tmp_path):
    d = _make_images(tmp_path / "imgs", ["x.png", "y.png"])
    assert len(_expand_images(str(d / "*.png"))) == 2
    assert len(_expand_images([d / "x.png", d / "y.png"])) == 2


def test_batch_writes_one_figure_per_image(tmp_path):
    d = _make_images(tmp_path / "imgs", ["a.png", "b.png", "c.png"])
    out_dir = tmp_path / "out"
    written = ll.batch("vit_tiny_patch16_224", d, out_dir, layers=[-1], pretrained=False)
    assert len(written) == 3
    assert {Path(p).name for p in written} == {"a.png", "b.png", "c.png"}
    assert all(Path(p).exists() for p in written)


def test_batch_loads_model_once(tmp_path, monkeypatch):
    d = _make_images(tmp_path / "imgs", ["a.png", "b.png", "c.png"])
    import sys
    batch_mod = sys.modules["featlens.batch"]  # the exported `batch` fn shadows the attribute

    calls = {"n": 0}
    real = batch_mod.FeatureGrid

    def counting_grid(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(batch_mod, "FeatureGrid", counting_grid)
    ll.batch("vit_tiny_patch16_224", d, tmp_path / "out", layers=[-1], pretrained=False)
    assert calls["n"] == 1  # one grid for all three images


def test_batch_compare_mode(tmp_path):
    d = _make_images(tmp_path / "imgs", ["a.png", "b.png"])
    written = ll.batch(["vit_tiny_patch16_224"], d, tmp_path / "out",
                       mode="compare", layer=-1, pretrained=False)
    assert len(written) == 2 and all(Path(p).exists() for p in written)


def test_batch_rejects_correspond(tmp_path):
    d = _make_images(tmp_path / "imgs", ["a.png"])
    with pytest.raises(ValueError):
        ll.batch("vit_tiny_patch16_224", d, tmp_path / "out", mode="correspond", pretrained=False)


def test_batch_empty_dir_errors(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        ll.batch("vit_tiny_patch16_224", empty, tmp_path / "out", pretrained=False)
