"""Visualization methods — color a dense feature map ``[h, w, D]`` into an RGB tile.

PCA→RGB (the v0.1 default) lives in :mod:`featlens.pca`; this module adds the v0.2 family that
turns the *same* dense map into a different view:

- ``"cosine"``     — cosine similarity of every patch to a **seed** patch (a heatmap).
- ``"kmeans"``     — unsupervised k-means clustering of the patches (a segmentation map).
- ``"foreground"`` — fg/bg mask from the first robust-PCA component (reuses :func:`pca.get_robust_pca`).

These operate on the **native patch grid** (e.g. 16×16), not on an upscaled map: clustering /
similarity are computed per patch and the resulting small RGB tile is interpolated afterwards by
the caller. The contract mirrors :func:`pca.get_pca_map`: input ``[h, w, D]`` (or ``[1, h, w, D]``)
torch tensor, output ``[h, w, 3]`` numpy float array in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .pca import get_pca_map, get_robust_pca

METHODS = ("pca", "cosine", "kmeans", "foreground", "saliency")


def _get_cmap(name: str):
    """Version-safe colormap lookup (``matplotlib.colormaps`` on 3.6+, else ``cm.get_cmap``)."""
    import matplotlib

    try:
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        import matplotlib.cm as cm
        return cm.get_cmap(name)


def colorize(
    fmap: torch.Tensor,
    method: str = "pca",
    *,
    seed: Optional[Sequence[float]] = None,
    k: int = 6,
    colormap: str = "turbo",
    mark_seed: bool = True,
    pca_stats=None,
    outlier_threshold: float = 2.0,
    remove_first_component: bool = False,
) -> np.ndarray:
    """Color a dense feature map ``[h, w, D]`` to RGB ``[h, w, 3]`` with the chosen ``method``."""
    if fmap.dim() == 4:
        fmap = fmap[0]
    if fmap.dim() != 3:
        raise ValueError(f"colorize expects [h, w, D] (got shape {tuple(fmap.shape)}).")

    if method == "pca":
        return get_pca_map(fmap, pca_stats=pca_stats, outlier_threshold=outlier_threshold,
                           remove_first_component=remove_first_component)
    if method == "cosine":
        sim = cosine_similarity_map(fmap, seed if seed is not None else (0.5, 0.5))
        rgb = apply_colormap((sim + 1.0) / 2.0, colormap)
        if mark_seed:
            h, w = fmap.shape[:2]
            r, c = seed_to_cell(seed if seed is not None else (0.5, 0.5), h, w)
            rgb[r, c] = np.array([1.0, 1.0, 1.0])
        return rgb
    if method == "kmeans":
        labels = kmeans_labels(fmap, k)
        return labels_to_rgb(labels, k)
    if method == "foreground":
        return foreground_mask(fmap, outlier_threshold=outlier_threshold)
    if method == "saliency":
        return apply_colormap(saliency_map(fmap), colormap)
    raise ValueError(f"Unknown method '{method}'. Choose from {METHODS}.")


def method_scalar(fmap: torch.Tensor, method: str, *,
                  seed: Optional[Sequence[float]] = None) -> Optional[np.ndarray]:
    """The scalar field underlying a method (``[h, w]``), or ``None`` if it has none.

    ``cosine`` → similarity to the seed in ``[-1, 1]``; ``saliency`` → normalized magnitude in
    ``[0, 1]``. ``pca`` / ``kmeans`` / ``foreground`` have no single scalar field, so return ``None``.
    """
    if method == "cosine":
        return cosine_similarity_map(fmap, seed if seed is not None else (0.5, 0.5))
    if method == "saliency":
        return saliency_map(fmap)
    return None


# ---- cosine similarity ----------------------------------------------------
def seed_to_cell(seed_xy: Sequence[float], h: int, w: int) -> Tuple[int, int]:
    """Normalized image coords ``(x, y) in [0, 1]`` -> grid cell ``(row, col)``."""
    x, y = float(seed_xy[0]), float(seed_xy[1])
    r = min(max(int(y * h), 0), h - 1)
    c = min(max(int(x * w), 0), w - 1)
    return r, c


def cosine_similarity_map(fmap: torch.Tensor, seed_xy: Sequence[float]) -> np.ndarray:
    """Cosine similarity of each patch in ``[h, w, D]`` to the seed patch -> ``[h, w]`` in [-1, 1]."""
    if fmap.dim() == 4:
        fmap = fmap[0]
    h, w, _ = fmap.shape
    r, c = seed_to_cell(seed_xy, h, w)
    flat = fmap.reshape(h * w, -1).float()
    flat = torch.nn.functional.normalize(flat, dim=1)
    seed_vec = torch.nn.functional.normalize(fmap[r, c].reshape(1, -1).float(), dim=1)
    sim = (flat @ seed_vec.T).reshape(h, w)
    return sim.cpu().numpy()


# ---- k-means --------------------------------------------------------------
def kmeans_labels(fmap: torch.Tensor, k: int = 6, iters: int = 25, seed: int = 0) -> np.ndarray:
    """Cluster patches of ``[h, w, D]`` into ``k`` groups -> ``[h, w]`` int label map.

    A tiny self-contained k-means (k-means++ init + Lloyd iterations) so the package keeps no
    sklearn/scipy dependency.
    """
    if fmap.dim() == 4:
        fmap = fmap[0]
    h, w, d = fmap.shape
    x = fmap.reshape(h * w, d).float()
    n = x.shape[0]
    k = max(1, min(int(k), n))
    g = torch.Generator(device="cpu").manual_seed(seed)

    # k-means++ initialization.
    centers = x[torch.randint(0, n, (1,), generator=g)]
    while centers.shape[0] < k:
        d2 = torch.cdist(x, centers).min(dim=1).values ** 2
        probs = d2 / (d2.sum() + 1e-12)
        idx = torch.multinomial(probs, 1, generator=g)
        centers = torch.cat([centers, x[idx]], dim=0)

    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        new_labels = torch.cdist(x, centers).argmin(dim=1)
        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = x[mask].mean(dim=0)
    return labels.reshape(h, w).cpu().numpy()


def labels_to_rgb(labels: np.ndarray, k: int, colormap: str = "tab20") -> np.ndarray:
    """Map an int label map to RGB using a qualitative palette."""
    cmap = _get_cmap(colormap)
    palette = np.array([cmap((i % cmap.N) / max(cmap.N - 1, 1))[:3] for i in range(max(k, 1))])
    return palette[labels % len(palette)]


# ---- foreground mask ------------------------------------------------------
def foreground_mask(fmap: torch.Tensor, outlier_threshold: float = 2.0,
                    threshold: float = 0.2) -> np.ndarray:
    """Binary fg/bg mask from the first robust-PCA component (fg=white, bg=black).

    Reuses the exact first-component thresholding from :func:`pca.get_robust_pca`
    (``remove_first_component`` path): patches whose normalized 1st component is ``< threshold``
    are foreground.
    """
    if fmap.dim() == 4:
        fmap = fmap[0]
    h, w, d = fmap.shape
    feats = fmap.reshape(h * w, d).float()
    reduction_mat, _, _ = get_robust_pca(feats, m=outlier_threshold)
    comp = (feats @ reduction_mat)[:, 0]
    comp = (comp - comp.min()) / (comp.max() - comp.min() + 1e-8)
    fg = (comp < threshold).reshape(h, w).cpu().numpy()
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[fg] = 1.0
    return rgb


# ---- saliency (activation magnitude) -------------------------------------
def saliency_map(fmap: torch.Tensor) -> np.ndarray:
    """Per-patch feature **L2 norm** of ``[h, w, D]``, min-max normalized to ``[0, 1]`` (``[h, w]``).

    A quick "where does the model put energy" view — high where patch activations are strong.
    """
    if fmap.dim() == 4:
        fmap = fmap[0]
    h, w, d = fmap.shape
    mag = fmap.reshape(h * w, d).float().norm(dim=1).reshape(h, w)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return mag.cpu().numpy()


# ---- shared colormap helper ----------------------------------------------
def apply_colormap(values01: np.ndarray, name: str = "turbo") -> np.ndarray:
    """Map scalar values in [0, 1] (``[h, w]`` numpy/torch) to RGB ``[h, w, 3]``."""
    if isinstance(values01, torch.Tensor):
        values01 = values01.cpu().numpy()
    values01 = np.clip(values01, 0.0, 1.0)
    cmap = _get_cmap(name)
    return cmap(values01)[..., :3]


# ---- figure-level scales (colorbar / legend) ------------------------------
def scalar_colorbar(fig, axes, colormap="turbo", *, vmin, vmax, label, ticks=None):
    """Attach a shared ``[vmin, vmax]`` colorbar to ``fig``.

    ``axes`` is the axis (or list of axes) the bar should steal space from. Tiles produced by a
    scalar method map the same value range through ``colormap``, so one bar describes the figure.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=_get_cmap(colormap))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02,
                        ticks=ticks if ticks is not None else [vmin, (vmin + vmax) / 2, vmax])
    cbar.set_label(label, fontsize=10)
    return cbar


def cosine_colorbar(fig, axes, colormap: str = "turbo"):
    """Shared [-1, 1] colorbar for ``cosine`` heatmaps."""
    return scalar_colorbar(fig, axes, colormap, vmin=-1.0, vmax=1.0,
                           label="cosine similarity", ticks=[-1.0, 0.0, 1.0])


def saliency_colorbar(fig, axes, colormap: str = "turbo"):
    """Shared [0, 1] colorbar for ``saliency`` (normalized activation magnitude)."""
    return scalar_colorbar(fig, axes, colormap, vmin=0.0, vmax=1.0,
                           label="activation (norm.)", ticks=[0.0, 0.5, 1.0])


def kmeans_legend(fig, k: int, colormap: str = "tab20"):
    """Attach a legend of ``k`` cluster swatches to ``fig`` (matches :func:`labels_to_rgb`).

    k-means runs independently per tile, so the colors are a per-tile key — not comparable across
    tiles — but they still let a reader map a color to its cluster index within a tile.
    """
    from matplotlib.patches import Patch

    cmap = _get_cmap(colormap)
    k = max(1, int(k))
    handles = [Patch(facecolor=cmap((i % cmap.N) / max(cmap.N - 1, 1)), label=f"cluster {i}")
               for i in range(k)]
    return fig.legend(handles=handles, loc="lower center", ncol=min(k, 8),
                      fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
