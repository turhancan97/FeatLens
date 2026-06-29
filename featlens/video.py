"""Multi-frame video feature maps — a filmstrip (frames × layers) and an animated GIF.

Two paths:

- **Temporal models** (V-JEPA, ``lm.uses_temporal``): the clip is fed once and the spatiotemporal
  token sequence is split into per-time-step grids (:func:`featlens.tokens.tokens_to_spatiotemporal`).
- **Any 2D model**: each sampled frame is run independently — a per-frame feature map. This is the
  dependency-free baseline and works for every registry model.

Input is a video file (needs the ``featlens[video]`` extra) or a directory / glob / list of frames.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from . import methods
from .extractor import FeatureExtractor
from .pca import get_pca_map

PathLike = Union[str, Path]
_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif")


def _sample(seq, n):
    if n <= 0 or len(seq) <= n:
        return list(seq)
    idx = np.linspace(0, len(seq) - 1, n).round().astype(int)
    return [seq[i] for i in idx]


def _load_frames(src, n_frames):
    """Return a list of ``n_frames`` PIL frames from a video file or a folder/glob/list."""
    from PIL import Image

    p = Path(src) if isinstance(src, (str, Path)) else None
    if p is not None and p.is_file() and p.suffix.lower() in _VIDEO_EXTS:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise ImportError(
                "Reading video files needs the video extra: pip install 'featlens[video]'."
            ) from exc
        frames = [Image.fromarray(f).convert("RGB") for f in iio.imiter(p)]
        if not frames:
            raise ValueError(f"No frames decoded from {src!r}.")
        return _sample(frames, n_frames)

    from .batch import _expand_images
    paths = _sample(_expand_images(src), n_frames)
    if not paths:
        raise FileNotFoundError(f"No frames found for {src!r}.")
    return [Image.open(pp).convert("RGB") for pp in paths]


def _interp_rgb(rgb: np.ndarray, size: int, method: str) -> np.ndarray:
    nearest = method in ("kmeans", "foreground")
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float()
    mode = "nearest" if nearest else "bilinear"
    kw = {} if nearest else {"align_corners": False}
    return F.interpolate(t, size=(size, size), mode=mode, **kw)[0].permute(1, 2, 0).numpy()


def _colorize(fmap: torch.Tensor, method: str, seed, k, colormap) -> np.ndarray:
    if method == "pca":
        return get_pca_map(fmap)
    return methods.colorize(fmap, method, seed=seed, k=k, colormap=colormap)


def video(
    model,
    src: PathLike,
    *,
    layers: Optional[Sequence[int]] = None,
    n_frames: int = 16,
    method: str = "pca",
    img_size: int = 256,
    colormap: str = "turbo",
    k: int = 6,
    seed=None,
    out: Optional[PathLike] = None,
    gif: Optional[PathLike] = None,
    fps: int = 4,
    pretrained: bool = True,
    device: Optional[str] = None,
    resize_mode: str = "squash",
    interpolation_size: int = 224,
    return_data: bool = False,
):
    """Render per-frame feature maps for a clip as a filmstrip PNG and an animated GIF.

    ``src`` is a video file (needs ``featlens[video]``) or a directory / glob / list of frames.
    Returns the filmstrip path (or, with ``return_data=True``, a dict including ``frames_rgb``).
    """
    ex = FeatureExtractor(model, layers=list(layers) if layers else [-1], img_size=img_size,
                          pretrained=pretrained, resize_mode=resize_mode)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ex.model.to(dev)

    frames = _load_frames(src, n_frames)
    tens = torch.stack([ex.transform(f) for f in frames]).to(dev)  # [T, C, H, W]
    n_layers = len(ex.layers)

    if ex.lm.uses_temporal:
        tubelet = int(ex.lm.extra.get("tubelet_size", 2))
        T = max(tubelet, (tens.shape[0] // tubelet) * tubelet)
        clip = tens[:T].permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
        maps = ex.extract_clip(clip, n_temporal=T // tubelet)[0]  # [L, T', D, h, w]
        n_steps = maps.shape[1]
        get_fmap = lambda l, t: rearrange(maps[l, t], "d h w -> h w d")
    else:
        feats = ex(tens)  # [T, L, D, h, w]
        n_steps = feats.shape[0]
        get_fmap = lambda l, t: rearrange(feats[t, l], "d h w -> h w d")

    # tiles[layer][step] -> interpolated RGB
    tiles = [[None] * n_steps for _ in range(n_layers)]
    for l in range(n_layers):
        for t in range(n_steps):
            rgb = _colorize(get_fmap(l, t).cpu(), method, seed, k, colormap)
            tiles[l][t] = np.clip(_interp_rgb(rgb, interpolation_size, method), 0, 1)

    out_path = _filmstrip(tiles, [str(x) for x in ex.layers], n_steps, out) if out else None
    gif_path = gif if gif else (str(Path(out).with_suffix(".gif")) if out else None)
    if gif_path:
        _gif(tiles[-1], gif_path, fps)  # animate the last layer's frames

    if not return_data:
        return out_path or gif_path
    return {"frames_rgb": np.array(tiles), "layers": list(ex.layers),
            "path": out_path, "gif": gif_path}


def _filmstrip(tiles, layer_labels, n_steps, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(tiles)
    fig, axes = plt.subplots(n_rows, n_steps, figsize=(2.2 * n_steps, 2.3 * n_rows), squeeze=False)
    for r in range(n_rows):
        for c in range(n_steps):
            ax = axes[r][c]
            ax.imshow(tiles[r][c]); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"frame {c}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"layer {layer_labels[r]}", fontsize=10)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def _gif(frame_tiles, out, fps):
    from PIL import Image

    imgs = [Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8)) for t in frame_tiles]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    duration = max(1, int(1000 / max(1, fps)))
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=duration, loop=0, optimize=True)
    return str(out)
