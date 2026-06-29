"""Cross-image patch correspondence.

Pick a **seed** patch in image A and find the patches in image B whose features are most similar
to it — the classic "do these two images share parts in feature space" view. This is structurally
different from the model × layer grid (two images, matching *between* them), so it gets its own
small renderer rather than bending :class:`featlens.FeatureGrid`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from .extractor import FeatureExtractor
from .methods import apply_colormap, seed_to_cell


def _one_layer_map(ex: FeatureExtractor, pil, layer: int, device) -> torch.Tensor:
    """Extract a single layer's dense map for one image -> ``[h, w, D]``."""
    t = ex.transform(pil).unsqueeze(0).to(device)
    feats = ex(t).float().cpu()  # [1, L, D, h, w]
    return rearrange(feats[0, 0], "d h w -> h w d")


def correspond(
    model: Union[str, FeatureExtractor],
    img_a: Union[str, Path],
    img_b: Union[str, Path],
    *,
    layer: int = -1,
    seed: Sequence[float] = (0.5, 0.5),
    topk: int = 1,
    img_size: int = 224,
    resize_mode: str = "squash",
    pretrained: bool = True,
    device: Optional[str] = None,
    colormap: str = "turbo",
    interpolation_size: int = 224,
    arrows: bool = True,
    out: Optional[Union[str, Path]] = None,
):
    """Render seed-patch correspondence between two images, as three panels.

    1. **source** — image A with the seed patch marked.
    2. **target** — the original image B; the top-``topk`` matches are circled and (when
       ``arrows`` is set, the default) an arrow runs from the seed to each match.
    3. **cosine similarity** — image B's similarity heatmap with the same matches circled.

    Each match gets its own color, shared between the target and heatmap panels so the two line
    up at a glance. Returns the saved path (if ``out``) or the figure.
    """
    from PIL import Image

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ex = model if isinstance(model, FeatureExtractor) else FeatureExtractor(
        model, layers=[layer], img_size=img_size, pretrained=pretrained, resize_mode=resize_mode)
    ex.model.to(dev)

    pil_a = Image.open(img_a).convert("RGB")
    pil_b = Image.open(img_b).convert("RGB")
    fa = _one_layer_map(ex, pil_a, layer, dev)  # [h, w, D]
    fb = _one_layer_map(ex, pil_b, layer, dev)
    h, w, _ = fa.shape

    r, c = seed_to_cell(seed, h, w)
    seed_vec = F.normalize(fa[r, c].reshape(1, -1), dim=1)
    flat_b = F.normalize(fb.reshape(h * w, -1), dim=1)
    sim = (flat_b @ seed_vec.T).reshape(h, w).numpy()  # [-1, 1]

    heat = apply_colormap((sim + 1.0) / 2.0, colormap)  # [h, w, 3]
    top_cells = np.argsort(sim.reshape(-1))[::-1][:max(1, topk)]

    src_a = _denorm_source(ex, pil_a, interpolation_size)
    src_b = _denorm_source(ex, pil_b, interpolation_size)
    heat_up = _interp_rgb(heat, interpolation_size)

    return _compose_triple(
        src_a, src_b, heat_up, (r, c), [divmod(int(i), w) for i in top_cells], (h, w),
        interpolation_size, arrows, out)


# ---- rendering helpers ----------------------------------------------------
def _denorm_source(ex: FeatureExtractor, pil, size: int) -> np.ndarray:
    t = ex.transform(pil)
    img = ex.denormalize(t)  # [H, W, 3] in [0, 1]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img = F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)
    return img[0].permute(1, 2, 0).numpy()


def _interp_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float()
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0].permute(1, 2, 0).numpy()


def _compose_triple(src_a, src_b, heat_b, seed_cell, match_cells, grid_hw, size, arrows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import ConnectionPatch

    h, w = grid_hw
    sy, sx = (seed_cell[0] + 0.5) / h * size, (seed_cell[1] + 0.5) / w * size
    outline = [pe.withStroke(linewidth=3.0, foreground="black")]
    palette = plt.get_cmap("tab10")
    colors = [palette(i % 10) for i in range(len(match_cells))]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.clip(src_a, 0, 1))
    axes[0].scatter([sx], [sy], s=180, marker="*", c="white", edgecolors="black", linewidths=1.2)
    axes[0].set_title("source (seed)", fontsize=11)
    axes[1].imshow(np.clip(src_b, 0, 1))
    axes[1].set_title("target (matches)", fontsize=11)
    axes[2].imshow(np.clip(heat_b, 0, 1))
    axes[2].set_title("cosine similarity", fontsize=11)

    for rank, (my, mx) in enumerate(match_cells):
        py, px = (my + 0.5) / h * size, (mx + 0.5) / w * size
        color = colors[rank]
        best = rank == 0
        size_pt = 150 if best else 110
        lw = 2.2 if best else 1.6
        # Same colored circle on both the target photo and the heatmap, so they correspond.
        for ax in (axes[1], axes[2]):
            ax.scatter([px], [py], s=size_pt, marker="o", facecolors="none",
                       edgecolors=[color], linewidths=lw, path_effects=outline)
        if arrows:
            # Arrow from the seed (panel 0) to this match on the *original* target photo (panel 1).
            con = ConnectionPatch(
                xyA=(sx, sy), coordsA=axes[0].transData,
                xyB=(px, py), coordsB=axes[1].transData,
                arrowstyle="-|>", color=color, linewidth=2.0 if best else 1.3,
                mutation_scale=16 if best else 11, alpha=1.0 if best else 0.8,
                shrinkA=6, shrinkB=6, zorder=5)
            con.set_path_effects(outline)
            con.set_clip_on(False)
            fig.add_artist(con)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return str(out)
    return fig
