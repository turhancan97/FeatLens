"""Opt-in on-disk cache for extracted features.

Extraction is the slow part; the rendered view (PCA / cosine / k-means / …) is cheap. So caching
the per-image ``[L, D, h, w]`` feature stack keyed on *what determines it* makes re-renders — and
especially the interactive demo, where the same image is re-colored on every click — instant.

The cache is **opt-in** (``cache=True``) so nothing writes to disk by surprise. The key includes
the image *content* (not its path) so editing an image invalidates its entry automatically.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Sequence

import torch


def default_cache_dir() -> Path:
    """``FEATLENS_CACHE_DIR`` env var, else ``~/.cache/featlens``."""
    env = os.environ.get("FEATLENS_CACHE_DIR")
    return Path(env) if env else Path.home() / ".cache" / "featlens"


def make_key(
    image_bytes: bytes,
    model_id: str,
    img_size: int,
    resize_mode: str,
    layers: Sequence[int],
    pretrained: bool = True,
) -> str:
    """Stable sha1 over everything that determines the extracted feature stack."""
    h = hashlib.sha1()
    h.update(image_bytes)
    h.update(b"\x00")
    h.update(str(model_id).encode())
    h.update(f"|{int(img_size)}|{resize_mode}|{int(pretrained)}|".encode())
    h.update(",".join(str(int(x)) for x in sorted(layers)).encode())
    return h.hexdigest()


class FeatureCache:
    """Tiny content-addressed cache of feature tensors saved with ``torch.save``."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.pt"

    def get(self, key: str) -> Optional[torch.Tensor]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return torch.load(p, map_location="cpu")
        except Exception:
            return None  # corrupt / partial write -> treat as a miss

    def put(self, key: str, tensor: torch.Tensor) -> None:
        tmp = self._path(key).with_suffix(".pt.tmp")
        torch.save(tensor.detach().cpu(), tmp)
        os.replace(tmp, self._path(key))  # atomic so readers never see a partial file
