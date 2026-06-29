"""Multi-seed + mutual-NN correspondence (CPU, random weights — no downloads)."""

from pathlib import Path

import featlens as ll

IMG = "examples/images/cat.jpg"
IMG_B = "examples/images/coffee.jpg"
MODEL = "vit_tiny_patch16_224"


def test_self_match_is_mutual_and_is_the_seed():
    # A == B: the seed's top match must be the seed patch itself, and it is cycle-consistent.
    res = ll.correspond(MODEL, IMG, IMG, layer=-1, seed=(0.5, 0.5), topk=1, mutual=True,
                        pretrained=False, device="cpu", return_data=True)
    (r, c) = res["matches"][0][0]
    sx, sy = res["seeds"][0]
    h, w = res["similarity"].shape[1:]
    assert (r, c) == (int(sy * h), int(sx * w))   # match == seed cell
    assert res["mutual"][0][0] is True


def test_multi_seed_renders_and_returns_per_seed(tmp_path):
    seeds = [(0.3, 0.4), (0.7, 0.6)]
    res = ll.correspond(MODEL, IMG, IMG_B, layer=-1, seed=seeds, topk=2,
                        pretrained=False, device="cpu", out=tmp_path / "c.png",
                        return_data=True)
    assert (tmp_path / "c.png").exists()
    assert len(res["seeds"]) == 2 and len(res["matches"]) == 2
    assert res["similarity"].shape[0] == 2        # one similarity map per seed


def test_single_seed_default_unchanged(tmp_path):
    out = ll.correspond(MODEL, IMG, IMG_B, layer=-1, seed=(0.4, 0.5), topk=3,
                        pretrained=False, device="cpu", out=tmp_path / "c.png")
    assert out == str(tmp_path / "c.png")
