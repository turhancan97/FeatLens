"""Robust PCA → RGB feature-map colorization.

``get_robust_pca`` / ``get_pca_map`` are adapted from
``midvision-probe/scripts/visualize_featuremap.py``. The robustness (median-absolute-
deviation outlier filtering) is what keeps the colors meaningful when a few patches are
extreme. ``fit_pca_stats`` exposes the basis so it can be **shared** across the layers of
one model (consistent colors when scrubbing layers).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

PcaStats = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (reduction_mat, rgb_min, rgb_max)


def get_robust_pca(features: torch.Tensor, m: float = 2.0,
                   remove_first_component: bool = False) -> PcaStats:
    """Fit a robust 3-component PCA basis on ``[N, C]`` features."""
    assert features.dim() == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat

    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()

    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / (mdev + 1e-8)
    try:
        rins = colors[fg_mask][s[:, 0] < m, 0]
        gins = colors[fg_mask][s[:, 1] < m, 1]
        bins = colors[fg_mask][s[:, 2] < m, 2]
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    except Exception:
        rgb_min = torch.tensor([colors.min()] * 3)
        rgb_max = torch.tensor([colors.max()] * 3)
    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def fit_pca_stats(feature_map: torch.Tensor, outlier_threshold: float = 2.0,
                  remove_first_component: bool = False) -> PcaStats:
    """Fit PCA stats on a feature map ``[..., C]`` (any leading dims)."""
    return get_robust_pca(feature_map.reshape(-1, feature_map.shape[-1]),
                          m=outlier_threshold, remove_first_component=remove_first_component)


def get_pca_map(feature_map: torch.Tensor, pca_stats: Optional[PcaStats] = None,
                outlier_threshold: float = 2.0, remove_first_component: bool = False):
    """Convert a feature map ``[h, w, C]`` (or ``[1, h, w, C]``) to an ``[h, w, 3]`` RGB array.

    Pass ``pca_stats`` (from ``fit_pca_stats``) to reuse a shared basis; otherwise it is fit
    on this map. Returns a numpy float array in [0, 1].
    """
    if feature_map.dim() == 3:
        feature_map = feature_map[None]
    if pca_stats is None:
        pca_stats = fit_pca_stats(feature_map, outlier_threshold, remove_first_component)
    reduction_mat, color_min, color_max = pca_stats
    pca_color = feature_map @ reduction_mat
    pca_color = (pca_color - color_min) / (color_max - color_min + 1e-8)
    pca_color = pca_color.clamp(0, 1)
    return pca_color.cpu().numpy().squeeze(0)
