"""Shared helpers for the example loaders in this folder (not part of the featlens package).

Each ``load_*.py`` exposes a ``load(...)`` function that returns a :class:`featlens.adapters.LoadedModel`,
so the backbone can be passed to ``FeatureExtractor`` / ``FeatureGrid`` / ``ll.visualize`` like a
built-in one, and a small command line that renders a layer row for one image.

Checkpoints are downloaded once into ``$FEATLENS_CACHE`` (default ``~/.cache/featlens``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import torch

from featlens.preprocess import IMAGENET_MEAN, IMAGENET_STD

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "cache_dir", "download", "check_state_dict",
           "norm2_hook", "resample_pos_embed", "cli", "require_repo"]


def cache_dir() -> Path:
    path = Path(os.environ.get("FEATLENS_CACHE", "~/.cache/featlens")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url: str, filename: str) -> Path:
    """Download ``url`` into the cache once and return the local path."""
    dest = cache_dir() / filename
    if not dest.exists():
        print(f"Downloading {url} -> {dest}")
        torch.hub.download_url_to_file(url, str(dest), progress=True)
    return dest


def require_repo(env_var: str, url: str) -> str:
    """Return the path of a cloned model repository given in ``$env_var``."""
    path = os.environ.get(env_var, "")
    if not path or not os.path.isdir(path):
        raise SystemExit(f"Set ${env_var} to a clone of {url}  (git clone {url})")
    return path


def norm2_hook(blocks_attr: str = "blocks"):
    """Hook target reading ``<blocks_attr>[i].norm2``, the point FeatLens reads for every timm ViT.

    Using the same point for external models keeps their feature maps comparable with registry ones.
    """
    from featlens.adapters.base import real_index

    def hook(model, idx):
        blocks = getattr(model, blocks_attr)
        return blocks[real_index(idx, len(blocks))].norm2

    return hook


def check_state_dict(result, allowed_missing: Iterable[str] = ()) -> None:
    """Fail loudly unless a non-strict ``load_state_dict`` only missed ``allowed_missing`` prefixes."""
    allowed = tuple(allowed_missing)
    missing = [k for k in result.missing_keys if not k.startswith(allowed)]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch - missing: {missing}, unexpected: {result.unexpected_keys}")


def resample_pos_embed(state_dict: dict, model: torch.nn.Module, num_prefix_tokens: int = 1) -> None:
    """Resize an absolute position embedding in ``state_dict`` to the model's grid, in place."""
    from timm.layers import resample_abs_pos_embed

    pe = state_dict.get("pos_embed")
    if pe is not None and pe.shape != model.pos_embed.shape:
        state_dict["pos_embed"] = resample_abs_pos_embed(
            pe, new_size=model.patch_embed.grid_size, num_prefix_tokens=num_prefix_tokens)


def cli(load_fn, description: str, extra_args=None) -> None:
    """Shared command line: load the backbone, render a layer row for one image."""
    import featlens as ll

    p = argparse.ArgumentParser(description=description)
    p.add_argument("--image", default=str(Path(__file__).resolve().parents[1] / "images" / "sparrta_bridge.jpg"))
    p.add_argument("--layers", type=int, nargs="+", default=[2, 5, 8, -1])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--out", default=None, help="output PNG (default: <model>_layers.png)")
    if extra_args:
        extra_args(p)
    args = p.parse_args()
    kwargs = {k: v for k, v in vars(args).items() if k not in {"image", "layers", "img_size", "out"}}
    lm = load_fn(img_size=args.img_size, **kwargs)
    out = args.out or f"{lm.name.split()[0].lower()}_layers.png"
    # A (label, LoadedModel) pair works anywhere a model spec does.
    path = ll.visualize((lm.name, lm), args.image, layers=args.layers, img_size=args.img_size, out=out)
    print(f"{lm.name}: {lm.num_blocks} blocks, patch {lm.patch_size} -> {path}")
