"""Feature cache — round-trip, hit avoids recompute, content invalidation (CPU, random weights)."""

import numpy as np
import torch

from featlens import FeatureGrid
from featlens.cache import FeatureCache, make_key


def test_cache_roundtrip(tmp_path):
    cache = FeatureCache(str(tmp_path))
    t = torch.randn(3, 32, 14, 14)
    cache.put("abc", t)
    got = cache.get("abc")
    assert got is not None and torch.equal(got, t)
    assert cache.get("missing") is None


def test_make_key_sensitivity():
    base = dict(image_bytes=b"img", model_id="m", img_size=224, resize_mode="squash", layers=[1, 2])
    k0 = make_key(**base)
    assert k0 == make_key(image_bytes=b"img", model_id="m", img_size=224,
                          resize_mode="squash", layers=[2, 1])  # layer order-insensitive
    assert k0 != make_key(**{**base, "image_bytes": b"other"})   # content invalidates
    assert k0 != make_key(**{**base, "img_size": 384})           # size invalidates
    assert k0 != make_key(**{**base, "model_id": "n"})           # model invalidates


def test_cache_evicts_over_max_bytes(tmp_path):
    t = torch.randn(64, 64)
    probe = FeatureCache(str(tmp_path))
    probe.put("probe", t)
    one = (tmp_path / "probe.pt").stat().st_size
    (tmp_path / "probe.pt").unlink()

    cap = int(one * 2.5)                      # room for ~2 entries
    cache = FeatureCache(str(tmp_path), max_bytes=cap)
    for key in ("a", "b", "c", "d"):
        cache.put(key, t)

    remaining = list(tmp_path.glob("*.pt"))
    assert sum(p.stat().st_size for p in remaining) <= cap   # stayed under the cap
    assert len(remaining) < 4                                # something was evicted
    assert (tmp_path / "d.pt").exists()                      # the newest entry survived


def test_cache_unlimited_when_max_bytes_zero(tmp_path):
    cache = FeatureCache(str(tmp_path), max_bytes=0)
    for key in ("a", "b", "c"):
        cache.put(key, torch.randn(32, 32))
    assert len(list(tmp_path.glob("*.pt"))) == 3            # nothing evicted


def test_cache_hit_avoids_recompute(tmp_path):
    img = "examples/images/cat.jpg"
    kw = dict(layers=[5, 11], pretrained=False, device="cpu", cache=True, cache_dir=str(tmp_path))

    g1 = FeatureGrid(["vit_tiny_patch16_224"], **kw)
    g1.render(img)
    assert g1.n_extractions == 1            # one forward on the cold cache

    g2 = FeatureGrid(["vit_tiny_patch16_224"], **kw)
    g2.render(img)
    assert g2.n_extractions == 0            # fully served from disk
