"""FeatLens — model-agnostic feature-map visualization.

Load any vision model (timm / HuggingFace / torch.hub / external repo / your own) and render
PCA-to-RGB feature maps from any layer, as a model × layer grid.

Quick start::

    import featlens as ll
    ll.grid(["dino_vitb16", "clip_large_openai"], "img.jpg", layers=[2, 5, 8, 11], out="grid.png")
    ll.visualize("dinov2_vitb14", "img.jpg", layers=[2, 5, 8, 11], out="row.png")   # scrub layers
    ll.compare(["dino_vitb16", "mae_vitb16"], "img.jpg", layer=-1, out="cmp.png")    # compare models

v0.2 adds non-PCA views (``method="cosine"|"kmeans"|"foreground"``), an opt-in feature
``cache=True``, :func:`correspond` for cross-image patch matching, and :func:`batch` to render a
whole directory (one figure per image).
"""

from typing import Optional, Sequence, Union
from pathlib import Path

from .extractor import FeatureExtractor
from .grid import FeatureGrid
from .correspond import correspond
from .batch import batch
from .video import video
from .attention import attention

__version__ = "0.3.0"
__all__ = ["FeatureExtractor", "FeatureGrid", "grid", "visualize", "compare", "correspond",
           "batch", "video", "attention"]


_RENDER_KEYS = ("overlay", "overlay_alpha", "figscale", "return_data", "include_features")


def _split_kwargs(kwargs):
    render_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in _RENDER_KEYS}
    return kwargs, render_kw


def grid(models, images, layers=None, out=None, **kwargs):
    """Render a full model × layer grid (per-tile PCA basis by default)."""
    ctor_kw, render_kw = _split_kwargs(kwargs)
    return FeatureGrid(models, layers=layers, **ctor_kw).render(images, out_path=out, **render_kw)


def visualize(model, images, layers=None, out=None, **kwargs):
    """One model across layers — uses a shared PCA basis so colors are comparable across the row."""
    kwargs.setdefault("basis", "shared_per_model")
    ctor_kw, render_kw = _split_kwargs(kwargs)
    return FeatureGrid([model], layers=layers, **ctor_kw).render(images, out_path=out, **render_kw)


def compare(models, images, layer: int = -1, out=None, **kwargs):
    """Many models at a single layer (per-tile PCA basis)."""
    ctor_kw, render_kw = _split_kwargs(kwargs)
    return FeatureGrid(models, layers=[layer], **ctor_kw).render(images, out_path=out, **render_kw)
