"""FeatureGrid: render a model × layer matrix of PCA feature-map tiles.

Rows are models, columns are layers. "One model, scrub layers" is a single row;
"many models, fixed layer" is a single column. The PCA **basis policy** differs by axis:
``per_tile`` (default; required across models) vs ``shared_per_model`` (one basis per row,
so colors are comparable when scrubbing a model's layers).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from . import methods
from .cache import FeatureCache, make_key
from .extractor import FeatureExtractor
from .pca import fit_pca_stats, get_pca_map

ModelSpec = Union[str, FeatureExtractor, Tuple[str, str]]


class FeatureGrid:
    def __init__(
        self,
        models: Sequence[ModelSpec],
        layers: Optional[Sequence[int]] = None,
        img_size: int = 224,
        pretrained: bool = True,
        device: Optional[str] = None,
        basis: str = "per_tile",
        outlier_threshold: float = 2.0,
        remove_first_component: bool = False,
        interpolation_size: int = 224,
        resize_mode: str = "squash",
        method: str = "pca",
        seed: Optional[Sequence[float]] = None,
        k: int = 6,
        colormap: str = "turbo",
        cache: bool = False,
        cache_dir: Optional[str] = None,
    ):
        if basis not in ("per_tile", "shared_per_model"):
            raise ValueError("basis must be 'per_tile' or 'shared_per_model'.")
        if method not in methods.METHODS:
            raise ValueError(f"method must be one of {methods.METHODS}.")
        self.layers = list(layers) if layers is not None else [-1]
        self.img_size = img_size
        self.pretrained = pretrained
        self.resize_mode = resize_mode
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.basis = basis
        self.outlier_threshold = outlier_threshold
        self.remove_first_component = remove_first_component
        self.interp = interpolation_size
        self.method = method
        self.seed = tuple(seed) if seed is not None else (0.5, 0.5)
        self.k = k
        self.colormap = colormap
        self._cache = FeatureCache(cache_dir) if cache else None
        self.n_extractions = 0  # model forward passes actually run (for cache tests)
        self._models = [self._make_entry(m) for m in models]

    def _make_entry(self, m: ModelSpec) -> Tuple[str, FeatureExtractor]:
        if isinstance(m, FeatureExtractor):
            return self._short_label(m.name), m
        if isinstance(m, tuple):
            label, spec = m
            ex = FeatureExtractor(spec, layers=self.layers, img_size=self.img_size,
                                  pretrained=self.pretrained, resize_mode=self.resize_mode)
            return self._short_label(label), ex
        # Use the original (usually short) spec as the row label, not the resolved timm id.
        ex = FeatureExtractor(m, layers=self.layers, img_size=self.img_size,
                              pretrained=self.pretrained, resize_mode=self.resize_mode)
        return self._short_label(m), ex

    @staticmethod
    def _short_label(name: str, maxlen: int = 22) -> str:
        name = str(name)
        if ":" in name:            # drop a backend prefix like "hf:"
            name = name.split(":", 1)[1]
        if "/" in name:            # keep the last path segment (HF ids, paths)
            name = name.split("/")[-1]
        name = name.split(".")[0]  # drop a timm pretrain tag (".dino", ".openai", ...)
        if len(name) > maxlen:
            name = name[: maxlen - 1] + "…"
        return name

    # ---- core ------------------------------------------------------------
    def _extract(self, ex: FeatureExtractor, pils) -> torch.Tensor:
        tensor = torch.stack([ex.transform(p) for p in pils], dim=0).to(self.device)
        out = ex(tensor)  # [B, L, D, h, w]
        return out.float().cpu()

    def _features_for_model(self, ex: FeatureExtractor, pils, img_bytes=None) -> torch.Tensor:
        ex.model.to(self.device)
        if self._cache is None or img_bytes is None:
            self.n_extractions += len(pils)
            return self._extract(ex, pils)

        # Cached path: resolve per image, batch only the misses through one forward.
        results: List[Optional[torch.Tensor]] = [None] * len(pils)
        keys = [make_key(b, ex.name, self.img_size, self.resize_mode, ex.layers,
                         self.pretrained) for b in img_bytes]
        misses = []
        for i, key in enumerate(keys):
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                misses.append(i)
        if misses:
            self.n_extractions += len(misses)
            out = self._extract(ex, [pils[i] for i in misses])  # [M, L, D, h, w]
            for j, i in enumerate(misses):
                results[i] = out[j]
                self._cache.put(keys[i], out[j])
        return torch.stack(results, dim=0)  # [B, L, D, h, w]

    def _layer_map(self, feats: torch.Tensor, li: int) -> torch.Tensor:
        """feats: [B, L, D, h, w] -> tall feature map [(B h'), w', D] after interpolation."""
        fmap = feats[:, li]  # [B, D, h, w]
        if self.interp:
            fmap = F.interpolate(fmap, size=(self.interp, self.interp), mode="bilinear",
                                 align_corners=False)
        return rearrange(fmap, "n d h w -> (n h) w d")

    def _method_tile(self, feats: torch.Tensor, li: int) -> np.ndarray:
        """Colorize layer ``li`` per image on the native patch grid, then interpolate the RGB.

        Used for the non-PCA methods (cosine / k-means / foreground): clustering and similarity
        are computed per patch, so we color the small ``[h, w, D]`` grid and upscale the result.
        """
        fmap = feats[:, li]  # [B, D, h, w]
        rgbs = []
        for b in range(fmap.shape[0]):
            single = rearrange(fmap[b], "d h w -> h w d")  # [h, w, D] native grid
            rgb = methods.colorize(single, self.method, seed=self.seed, k=self.k,
                                   colormap=self.colormap,
                                   outlier_threshold=self.outlier_threshold)
            rgbs.append(self._interp_rgb(rgb))
        return np.concatenate(rgbs, axis=0)  # vertical stack, matching the PCA layout

    def _interp_rgb(self, rgb: np.ndarray) -> np.ndarray:
        if not self.interp:
            return rgb
        # Nearest keeps cluster/mask boundaries crisp; cosine heatmaps read better smoothed.
        nearest = self.method in ("kmeans", "foreground")
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float()
        if nearest:
            t = F.interpolate(t, size=(self.interp, self.interp), mode="nearest")
        else:
            t = F.interpolate(t, size=(self.interp, self.interp), mode="bilinear",
                              align_corners=False)
        return t[0].permute(1, 2, 0).numpy()

    @staticmethod
    def _read_bytes(p) -> bytes:
        try:
            return Path(p).read_bytes()
        except (TypeError, OSError):
            return str(p).encode()  # not a real path -> degrade to a stable string key

    def render(
        self,
        images: Union[str, Path, Sequence[Union[str, Path]]],
        out_path: Optional[Union[str, Path]] = None,
        overlay: bool = False,
        overlay_alpha: float = 0.45,
        figscale: float = 2.6,
    ):
        from PIL import Image

        if isinstance(images, (str, Path)):
            images = [images]
        pils = [Image.open(p).convert("RGB") for p in images]
        img_bytes = [self._read_bytes(p) for p in images] if self._cache is not None else None

        n_rows = len(self._models)
        n_cols = len(self.layers)
        tiles: List[List[np.ndarray]] = [[None] * n_cols for _ in range(n_rows)]

        for r, (label, ex) in enumerate(self._models):
            feats = self._features_for_model(ex, pils, img_bytes)  # [B, L, D, h, w]
            shared = None
            if self.method == "pca" and self.basis == "shared_per_model":
                allfeat = rearrange(feats, "n l d h w -> (n l h w) d")
                shared = fit_pca_stats(allfeat, self.outlier_threshold,
                                       self.remove_first_component)
            src = self._overlay_source(ex, pils) if overlay else None
            for c in range(n_cols):
                if self.method == "pca":
                    fmap = self._layer_map(feats, c)  # [(B h'), w', D]
                    rgb = get_pca_map(fmap, pca_stats=shared,
                                      outlier_threshold=self.outlier_threshold,
                                      remove_first_component=self.remove_first_component)
                else:
                    rgb = self._method_tile(feats, c)  # native colorize + interpolate, B-stacked
                if overlay and src is not None:
                    rgb = (1 - overlay_alpha) * src + overlay_alpha * rgb
                tiles[r][c] = np.clip(rgb, 0, 1)

        return self._compose(tiles, [lbl for lbl, _ in self._models],
                             [str(l) for l in self.layers], out_path, figscale)

    def _overlay_source(self, ex: FeatureExtractor, pils) -> np.ndarray:
        """Vertically-stacked denormalized source images at the tile resolution."""
        size = self.interp or self.img_size
        imgs = []
        for p in pils:
            t = ex.transform(p)
            img = ex.denormalize(t)  # [H, W, 3] in [0,1]
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            img = F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)
            imgs.append(img[0].permute(1, 2, 0).numpy())
        return np.concatenate(imgs, axis=0)

    def _compose(self, tiles, row_labels, col_labels, out_path, figscale):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_rows, n_cols = len(tiles), len(tiles[0])
        tile_h = tiles[0][0].shape[0] / tiles[0][0].shape[1]
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(figscale * n_cols, figscale * tile_h * n_rows),
            squeeze=False,
        )
        for r in range(n_rows):
            for c in range(n_cols):
                ax = axes[r][c]
                ax.imshow(tiles[r][c])
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"layer {col_labels[c]}", fontsize=11)
                if c == 0:
                    ax.set_ylabel(row_labels[r], fontsize=11)
        fig.tight_layout()
        # A readable scale for the methods whose colors carry meaning.
        if self.method == "cosine":
            methods.cosine_colorbar(fig, axes.ravel().tolist(), self.colormap)
        elif self.method == "kmeans":
            methods.kmeans_legend(fig, self.k)
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return str(out_path)
        return fig
