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

from . import methods
from .extractor import FeatureExtractor
from .methods import apply_colormap, seed_to_cell


def _one_layer_map(ex: FeatureExtractor, pil, layer: int, device) -> torch.Tensor:
    """Extract a single layer's dense map for one image -> ``[h, w, D]``."""
    t = ex.transform(pil).unsqueeze(0).to(device)
    feats = ex(t).float().cpu()  # [1, L, D, h, w]
    return rearrange(feats[0, 0], "d h w -> h w d")


def _as_seed_list(seed):
    """Normalize ``seed`` to a list of ``(x, y)`` pairs — accepts one pair or a list of pairs."""
    s = list(seed)
    if len(s) == 2 and all(isinstance(v, (int, float)) for v in s):
        return [(float(s[0]), float(s[1]))]
    return [(float(a), float(b)) for a, b in s]


def correspond(
    model: Union[str, FeatureExtractor],
    img_a: Union[str, Path],
    img_b: Union[str, Path],
    *,
    layer: int = -1,
    seed: Sequence = (0.5, 0.5),
    topk: int = 1,
    mutual: bool = False,
    img_size: int = 224,
    resize_mode: str = "squash",
    pretrained: bool = True,
    device: Optional[str] = None,
    colormap: str = "turbo",
    interpolation_size: int = 224,
    arrows: bool = True,
    return_data: bool = False,
    out: Optional[Union[str, Path]] = None,
):
    """Render seed-patch correspondence between two images, as three panels.

    1. **source** — image A with the seed patch(es) marked.
    2. **target** — the original image B; the top-``topk`` matches are circled and (when
       ``arrows`` is set, the default) an arrow runs from each seed to its matches.
    3. **cosine similarity** — image B's similarity heatmap with the same matches circled.

    ``seed`` may be a single ``(x, y)`` or a **list** of them (multi-seed). With ``mutual=True``,
    matches are filtered to **cycle-consistent** ones — a match patch in B whose own nearest
    neighbour back in A is the seed patch. Returns the saved path (if ``out``) or the figure; with
    ``return_data=True`` returns a dict of the similarity maps, seeds, matches and mutual flags.
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

    flat_a = F.normalize(fa.reshape(h * w, -1), dim=1)
    flat_b = F.normalize(fb.reshape(h * w, -1), dim=1)

    seeds = _as_seed_list(seed)
    groups = []  # each: (seed_cell, [match_cells], [mutual_flags], sim[h, w])
    for sxy in seeds:
        r, c = seed_to_cell(sxy, h, w)
        sidx = r * w + c
        sim = (flat_b @ flat_a[sidx:sidx + 1].T).reshape(h, w).numpy()  # [-1, 1]
        order = np.argsort(sim.reshape(-1))[::-1][:max(1, topk)]
        cells, flags = [], []
        for mi in order:
            nn_a = int((flat_a @ flat_b[int(mi)]).argmax())  # B->A nearest neighbour
            is_mutual = nn_a == sidx
            if mutual and not is_mutual:
                continue
            cells.append(divmod(int(mi), w))
            flags.append(is_mutual)
        groups.append(((r, c), cells, flags, sim))

    # Combined heatmap: max similarity across seeds (identical to the single seed's map for one seed).
    sim_stack = np.stack([g[3] for g in groups], axis=0)
    heat = apply_colormap((sim_stack.max(axis=0) + 1.0) / 2.0, colormap)  # [h, w, 3]

    src_a = _denorm_source(ex, pil_a, interpolation_size)
    src_b = _denorm_source(ex, pil_b, interpolation_size)
    heat_up = _interp_rgb(heat, interpolation_size)

    composed = _compose_triple(
        src_a, src_b, heat_up, [(g[0], g[1]) for g in groups], (h, w),
        interpolation_size, arrows, out, colormap)
    if not return_data:
        return composed
    return {
        "similarity": sim_stack,                  # [S, h, w] in [-1, 1]
        "seeds": seeds,                           # [(x, y), ...] normalized
        "matches": [g[1] for g in groups],        # [[(row, col), ...], ...] per seed
        "mutual": [g[2] for g in groups],         # [[bool, ...], ...] per seed
        "path": composed if out else None,
        "fig": None if out else composed,
    }


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


def _compose_triple(src_a, src_b, heat_b, groups, grid_hw, size, arrows, out, colormap="turbo"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import ConnectionPatch

    h, w = grid_hw
    outline = [pe.withStroke(linewidth=3.0, foreground="black")]
    palette = plt.get_cmap("tab10")
    single = len(groups) == 1  # one seed: color per match (as before); many: color per seed

    def to_px(cell):
        return (cell[1] + 0.5) / w * size, (cell[0] + 0.5) / h * size  # (x, y)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.clip(src_a, 0, 1)); axes[0].set_title("source (seed)", fontsize=11)
    axes[1].imshow(np.clip(src_b, 0, 1)); axes[1].set_title("target (matches)", fontsize=11)
    axes[2].imshow(np.clip(heat_b, 0, 1)); axes[2].set_title("cosine similarity", fontsize=11)

    for gi, (seed_cell, match_cells) in enumerate(groups):
        sx, sy = to_px(seed_cell)
        star_c = "white" if single else palette(gi % 10)
        axes[0].scatter([sx], [sy], s=180, marker="*", c=[star_c],
                        edgecolors="black", linewidths=1.2)
        for rank, cell in enumerate(match_cells):
            px, py = to_px(cell)
            color = palette(rank % 10) if single else palette(gi % 10)
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
    methods.cosine_colorbar(fig, [axes[2]], colormap)  # scale for the similarity panel
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return str(out)
    return fig
