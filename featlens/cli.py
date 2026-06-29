"""FeatLens command-line interface (argparse + optional YAML).

Examples::

    featlens --models dino_vitb16 clip_large_openai --layers 2 5 8 11 \\
        --images img.jpg --mode grid --overlay --out out/grid.png
    featlens --config configs/example.yaml --images img.jpg --out out/grid.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="featlens",
        description="Render PCA feature-map grids from any vision model and any layer.",
    )
    p.add_argument("--models", nargs="+", help="Model specs (friendly name, timm id, or backend:ident).")
    p.add_argument("--layers", nargs="+", type=int, help="Block indices (negatives allowed).")
    p.add_argument("--images", nargs="+", required=True, help="One or more image paths.")
    p.add_argument("--mode", choices=["grid", "visualize", "compare", "correspond"], default="grid",
                   help="grid = models x layers; visualize = one model, shared basis; "
                        "compare = many models, one layer; correspond = seed-patch matching "
                        "between --images[0] and --image-b.")
    p.add_argument("--layer", type=int, default=-1, help="Single layer for --mode compare/correspond.")
    p.add_argument("--method", choices=["pca", "cosine", "kmeans", "foreground"], default="pca",
                   help="Visualization method (default: pca).")
    p.add_argument("--seed", nargs=2, type=float, metavar=("X", "Y"), default=None,
                   help="Seed patch as normalized image coords in [0,1] for cosine/correspond.")
    p.add_argument("--k", type=int, default=6, help="Number of clusters for --method kmeans.")
    p.add_argument("--colormap", default="turbo", help="Matplotlib colormap for cosine heatmaps.")
    p.add_argument("--cache", action="store_true", help="Cache extracted features on disk.")
    p.add_argument("--image-b", default=None, help="Second image for --mode correspond.")
    p.add_argument("--topk", type=int, default=1, help="Top matches to mark for --mode correspond.")
    p.add_argument("--out", default="featlens_out.png", help="Output PNG path.")
    p.add_argument("--out-dir", default=None,
                   help="Batch mode: render one figure per input image into this directory "
                        "(--images may be a directory or glob). Not valid with --mode correspond.")
    p.add_argument("--img-size", type=int, default=224, help="Model input size (must be divisible by patch size).")
    p.add_argument("--resize-mode", choices=["squash", "crop", "pad"], default=None,
                   help="squash=force square (may distort); crop=resize shortest side + center-crop; "
                        "pad=resize longest side + pad. Default: squash.")
    p.add_argument("--basis", choices=["per_tile", "shared_per_model"], default=None,
                   help="PCA basis policy (defaults: grid/compare=per_tile, visualize=shared_per_model).")
    p.add_argument("--overlay", action="store_true", help="Blend feature map onto the source image.")
    p.add_argument("--overlay-alpha", type=float, default=0.45)
    p.add_argument("--no-pretrained", action="store_true", help="Use random weights (debug).")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto).")
    p.add_argument("--config", default=None, help="YAML with models/layers/img_size/basis defaults.")
    return p


def _load_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def main(argv: List[str] = None) -> None:
    args = _build_parser().parse_args(argv)
    cfg = _load_config(args.config) if args.config else {}

    models = args.models or cfg.get("models")
    if not models:
        raise SystemExit("No models given. Use --models ... or --config with a 'models:' list.")
    layers = args.layers if args.layers is not None else cfg.get("layers")
    img_size = args.img_size if args.img_size != 224 else cfg.get("img_size", 224)
    basis = args.basis or cfg.get("basis")
    pretrained = not args.no_pretrained

    import featlens as ll

    resize_mode = args.resize_mode or cfg.get("resize_mode") or "squash"

    if args.mode == "correspond":
        if args.out_dir:
            raise SystemExit("--out-dir is not valid with --mode correspond (it needs two images).")
        if not args.image_b:
            raise SystemExit("--mode correspond needs --image-b.")
        out = ll.correspond(
            models[0], args.images[0], args.image_b, layer=args.layer,
            seed=tuple(args.seed) if args.seed else (0.5, 0.5), topk=args.topk,
            img_size=img_size, resize_mode=resize_mode, pretrained=pretrained,
            device=args.device, colormap=args.colormap, out=args.out)
        print(f"Saved: {Path(out).resolve()}")
        return

    common = dict(img_size=img_size, pretrained=pretrained, device=args.device,
                  resize_mode=resize_mode, method=args.method, k=args.k,
                  colormap=args.colormap, cache=args.cache)
    if args.seed:
        common["seed"] = tuple(args.seed)
    if basis:
        common["basis"] = basis
    render_kw = dict(overlay=args.overlay, overlay_alpha=args.overlay_alpha)

    if args.out_dir:
        written = ll.batch(models, args.images, args.out_dir, mode=args.mode,
                           layers=layers, layer=args.layer, **common, **render_kw)
        print(f"Wrote {len(written)} figures to {Path(args.out_dir).resolve()}")
        return

    if args.mode == "visualize":
        out = ll.visualize(models[0], args.images, layers=layers, out=args.out, **common, **render_kw)
    elif args.mode == "compare":
        out = ll.compare(models, args.images, layer=args.layer, out=args.out, **common, **render_kw)
    else:
        out = ll.grid(models, args.images, layers=layers, out=args.out, **common, **render_kw)

    print(f"Saved: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
