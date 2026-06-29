"""Batch / directory mode — render one figure per image to an output directory.

The single-image path already does all the work; batch is "build the grid **once** (so model
weights load once) and render each image to its own file." Point it at a directory, a glob, or a
list of paths. The opt-in feature cache (``cache=True``) still applies, so re-runs are fast.
"""

from __future__ import annotations

import glob as _glob
from pathlib import Path
from typing import List, Optional, Sequence, Union

from .grid import FeatureGrid

PathLike = Union[str, Path]

# Image extensions we expand a directory / glob to (case-insensitive).
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
_GLOB_CHARS = ("*", "?", "[")


def _expand_images(images: Union[PathLike, Sequence[PathLike]]) -> List[Path]:
    """Resolve a directory / glob / file / list into a flat list of image paths.

    - a **directory** -> its image files, sorted by name;
    - a **glob** string (contains ``*``/``?``/``[``) -> matching paths, sorted;
    - a **file** -> itself;
    - a **list/tuple** -> each entry expanded and concatenated (so a list of dirs works).
    """
    if isinstance(images, (list, tuple)):
        out: List[Path] = []
        for item in images:
            out.extend(_expand_images(item))
        return out

    s = str(images)
    if any(ch in s for ch in _GLOB_CHARS):
        return [Path(p) for p in sorted(_glob.glob(s)) if Path(p).is_file()]

    p = Path(images)
    if p.is_dir():
        return sorted(q for q in p.iterdir()
                      if q.is_file() and q.suffix.lower() in IMAGE_EXTS)
    return [p]


def batch(
    models,
    images: Union[PathLike, Sequence[PathLike]],
    out_dir: PathLike,
    *,
    mode: str = "grid",
    layers: Optional[Sequence[int]] = None,
    layer: int = -1,
    out_suffix: str = "",
    overlay: bool = False,
    overlay_alpha: float = 0.45,
    figscale: float = 2.6,
    **kwargs,
) -> List[str]:
    """Render one figure per image into ``out_dir`` and return the written paths.

    ``models``/``images`` mirror :func:`featlens.grid`; ``images`` may be a directory, a glob, a
    single file, or a list of any of those. ``mode`` is ``"grid"`` (models × ``layers``),
    ``"visualize"`` (one model across ``layers``, shared basis), or ``"compare"`` (models at a
    single ``layer``). Other keywords (``method``, ``seed``, ``k``, ``colormap``, ``cache``,
    ``img_size``, ``device``, …) are forwarded to :class:`featlens.FeatureGrid`.

    The grid is built once, so model weights load a single time and are reused across every image.
    """
    if mode not in ("grid", "visualize", "compare"):
        raise ValueError("batch mode must be 'grid', 'visualize', or 'compare' "
                         "(correspond needs two images and isn't a per-image-folder op).")

    paths = _expand_images(images)
    if not paths:
        raise FileNotFoundError(f"No images found for {images!r}.")

    if mode == "compare":
        grid_layers = [layer]
    elif mode == "visualize":
        grid_layers = list(layers) if layers is not None else None
        kwargs.setdefault("basis", "shared_per_model")
    else:  # grid
        grid_layers = list(layers) if layers is not None else None

    model_specs = models if isinstance(models, (list, tuple)) else [models]
    grid = FeatureGrid(model_specs, layers=grid_layers, **kwargs)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    seen: set = set()
    n = len(paths)
    for i, img in enumerate(paths, start=1):
        stem = img.stem + out_suffix
        name = stem
        j = 1
        while name in seen:  # disambiguate stem collisions across source dirs
            name = f"{stem}_{j}"
            j += 1
        seen.add(name)
        out_path = out_root / f"{name}.png"
        grid.render([img], out_path=out_path, overlay=overlay,
                    overlay_alpha=overlay_alpha, figscale=figscale)
        written.append(str(out_path))
        print(f"[{i}/{n}] wrote {out_path}")

    return written
