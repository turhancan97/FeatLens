# Usage

## Python

```python
import featlens as fl

# One model, scrub layers (shared PCA basis -> colors comparable across the row)
fl.visualize("dinov2_vitb14", "img.jpg", layers=[2, 5, 8, 11], out="row.png")

# Compare models at the final layer (per-tile basis)
fl.compare(["dino_vitb16", "mae_vitb16", "clip_large_openai"], "img.jpg", layer=-1, out="cmp.png")

# Full model x layer grid, overlaid on the image
fl.grid(["dino_vitb16", "dinov2_vitb14"], "img.jpg", layers=[2, 5, 8, 11], overlay=True, out="grid.png")
```

The three entry points are thin wrappers over [`FeatureGrid`](api.md): `visualize` is one row
(shared basis), `compare` is one column (per-tile basis), and `grid` is the full matrix.

Beyond the default PCA→RGB, pass `method="cosine" | "kmeans" | "foreground"` for other views, and
`cache=True` to memoize extraction — see [Visualization methods](methods.md).

```python
fl.visualize("dino_vitb16", "img.jpg", layers=[2, 5, 8, 11], method="cosine", seed=(0.5, 0.5))
```

## CLI

```bash
featlens --models dino_vitb16 clip_large_openai --layers 2 5 8 11 \
    --images examples/images/cat.jpg --mode grid --out out/grid.png

featlens --config configs/example.yaml --images examples/images/cat.jpg --out out/grid.png
```

## Image size & resizing

Images are resized to a square **`img_size` × `img_size`** before the model (default **224**).
`img_size` must be divisible by the model's patch size (multiples of 16 for patch-16 models, 14
for patch-14). Larger sizes give a finer feature grid at more compute:

```python
fl.visualize("dinov2_vitb14", "img.jpg", layers=[2, 5, 8, 11], img_size=448)   # 32x32 grid
```

For **non-square images**, choose how aspect ratio is handled with `resize_mode`:

| `resize_mode` | behavior |
|---------------|----------|
| `squash` (default) | resize straight to `img_size²` — may distort |
| `crop` | resize shortest side to `img_size`, center-crop — aspect preserved |
| `pad` | resize longest side to `img_size`, pad to square — keeps the whole image |

```python
fl.grid([...], "wide.jpg", resize_mode="crop")
```

!!! tip
    `FeatureGrid(interpolation_size=…)` is separate — it only upscales the rendered tiles, not the
    model input.

## Layers

`layers=[2, 5, 8, 11]` selects **transformer block indices** (0-based, **negatives allowed**,
`-1` = last). The convention is uniform across backends — for HuggingFace models FeatLens maps
block `i` to `hidden_states[i+1]` (skipping the embedding output) for you.
