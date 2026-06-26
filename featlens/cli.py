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
    p.add_argument("--mode", choices=["grid", "visualize", "compare"], default="grid",
                   help="grid = models x layers; visualize = one model, shared basis; "
                        "compare = many models, one layer.")
    p.add_argument("--layer", type=int, default=-1, help="Single layer for --mode compare.")
    p.add_argument("--out", default="featlens_out.png", help="Output PNG path.")
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

    common = dict(img_size=img_size, pretrained=pretrained, device=args.device)
    resize_mode = args.resize_mode or cfg.get("resize_mode")
    if resize_mode:
        common["resize_mode"] = resize_mode
    if basis:
        common["basis"] = basis
    render_kw = dict(overlay=args.overlay, overlay_alpha=args.overlay_alpha)

    if args.mode == "visualize":
        out = ll.visualize(models[0], args.images, layers=layers, out=args.out, **common, **render_kw)
    elif args.mode == "compare":
        out = ll.compare(models, args.images, layer=args.layer, out=args.out, **common, **render_kw)
    else:
        out = ll.grid(models, args.images, layers=layers, out=args.out, **common, **render_kw)

    print(f"Saved: {Path(out).resolve()}")


if __name__ == "__main__":
    main()
