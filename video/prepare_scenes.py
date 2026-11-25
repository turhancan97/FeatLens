"""Generate the real FeatLens outputs used by scenes 4–6 (run in an env with featlens + timm).

    python video/prepare_scenes.py            # all steps
    python video/prepare_scenes.py compare    # just one step

Everything is computed with FeatLens itself and saved as plain PNG tiles (no matplotlib
figure chrome) so it sits cleanly on the video's dark background:

  methods/<pca|cosine|kmeans|foreground|saliency>.png   DINOv2 ViT-B/14, cat, 448px, last layer
  correspond/{photo,watercolor}.png + match.json        DINOv2 cross-image match (seed = eye)
  compare/<spec>.png                                    six ViT-B/16 backbones on market.jpg
  visualize/layer_<n>.png                               ll.visualize("dinov2_vitb14", ...) row
  vjepa/frame_<i>.png                                   frames of examples/vjepa_video_compare.gif
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import featlens as ll
from featlens import FeatureExtractor, methods

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets"
EX = ROOT.parent / "examples"
IMAGES = EX / "images"
TILE = 448

# k-means cluster colors from the video palette (cluster ids are arbitrary, so only the
# colors change, not the segmentation).
CLUSTER_RGB = np.array([(80, 122, 255), (255, 127, 80), (52, 199, 89), (217, 127, 174),
                        (234, 179, 77), (138, 107, 221), (85, 163, 152), (255, 204, 0)]) / 255


def save(rgb, path, size=TILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    im.resize((size, size), Image.NEAREST).save(path)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def last_layer(spec, image, img_size=TILE):
    ex = FeatureExtractor(spec, layers=[-1], img_size=img_size).to(DEVICE)
    with torch.no_grad():
        f = ex(ex.load_images([image]).to(DEVICE))[0, 0]   # [D, h, w]
    return f.permute(1, 2, 0).float().cpu()                # [h, w, D]


def method_tiles():
    fmap = last_layer("dinov2_vitb14", IMAGES / "cat_hires.jpg")
    seed = (0.5, 0.45)
    save(methods.colorize(fmap, "pca"), OUT / "methods" / "pca.png")
    save(methods.colorize(fmap, "cosine", seed=seed), OUT / "methods" / "cosine.png")
    save(CLUSTER_RGB[methods.kmeans_labels(fmap, 6)], OUT / "methods" / "kmeans.png")
    save(methods.colorize(fmap, "foreground"), OUT / "methods" / "foreground.png")
    save(methods.colorize(fmap, "saliency"), OUT / "methods" / "saliency.png")
    (OUT / "methods" / "seed.json").write_text(json.dumps({"seed": seed, "grid": fmap.shape[0]}))


def correspondence():
    a, b = IMAGES / "cat_hires.jpg", IMAGES / "cat_cartoon.jpg"
    seed = (0.40, 0.40)
    res = ll.correspond("dinov2_vitb14", a, b, layer=-1, seed=seed, topk=1, img_size=TILE,
                        return_data=True)
    h, w = res["similarity"].shape[-2:]
    (row, col), = res["matches"][0]
    d = OUT / "correspond"
    d.mkdir(parents=True, exist_ok=True)
    for src, name in [(a, "photo"), (b, "watercolor")]:        # squash, as FeatLens does
        Image.open(src).convert("RGB").resize((TILE, TILE), Image.LANCZOS).save(d / f"{name}.png")
    (d / "match.json").write_text(json.dumps({
        "seed": seed, "match": [(col + 0.5) / w, (row + 0.5) / h], "grid": [h, w],
        "similarity": float(res["similarity"][0, row, col])}))
    print(f"correspond: seed {seed} -> match cell ({row}, {col}) of {h}x{w}")


def compare_backbones():
    models = [("dino_vitb16", "DINO"), ("dinov3_vitb16", "DINOv3"), ("mae_vitb16", "MAE"),
              ("siglip_vitb16", "SigLIP"), ("supervised_vitb16", "Supervised"),
              ("perception_encoder_vitb16", "Perception Encoder")]
    for spec, _ in models:
        # 1024px (a 64x64 grid), as in the README gallery's compare_b16_market.png
        save(methods.colorize(last_layer(spec, IMAGES / "market.jpg", 1024), "pca"),
             OUT / "compare" / f"{spec}.png")
    (OUT / "compare" / "models.json").write_text(json.dumps(models))


def visualize_row():
    res = ll.visualize("dinov2_vitb14", IMAGES / "cat_hires.jpg", layers=[2, 5, 8, 11],
                       img_size=TILE, return_data=True)
    tiles = np.asarray(res["tiles"])                   # [rows, cols, H, W, 3]
    for j, layer in enumerate([2, 5, 8, 11]):
        save(tiles[0, j], OUT / "visualize" / f"layer_{layer}.png", size=TILE)


def vjepa_frames():
    gif = Image.open(EX / "vjepa_video_compare.gif")
    d = OUT / "vjepa"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(gif.n_frames):
        gif.seek(i)
        gif.convert("RGB").save(d / f"frame_{i}.png")


if __name__ == "__main__":
    import sys
    steps = {"methods": method_tiles, "correspond": correspondence, "visualize": visualize_row,
             "vjepa": vjepa_frames, "compare": compare_backbones}
    for name in (sys.argv[1:] or steps):
        steps[name]()
    print("done")
