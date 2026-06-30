# Visualization methods

Every method consumes the **same** dense feature stack — `FeatureExtractor.forward() → [B, L, D, h, w]`
— and turns it into an RGB tile a different way. Pick one with `method=` on `grid` / `visualize` /
`compare` (or `--method` on the CLI). Method works across layers too, so you can scrub a cosine
heatmap down the network just like the PCA grid.

| `method` | what it shows | extra args |
|----------|---------------|------------|
| `pca` (default) | robust PCA → RGB; structure & semantics by color | `basis`, `remove_first_component` |
| `cosine` | cosine similarity of every patch to a **seed** patch | `seed=(x, y)`, `colormap` |
| `kmeans` | unsupervised k-means of the patches → segmentation | `k` |
| `foreground` | fg/bg mask from the first robust-PCA component | — |
| `saliency` | per-patch activation magnitude (feature L2 norm) → "where the model fires" | `colormap` |

```python
import featlens as fl

# Cosine similarity to the patch at the image center, scrubbed across layers
fl.visualize("dino_vitb16", "img.jpg", layers=[2, 5, 8, 11], method="cosine", seed=(0.5, 0.5))

# k-means segmentation (8 clusters) at the last layer, comparing two models
fl.compare(["dino_vitb16", "dinov2_vitb14"], "img.jpg", layer=-1, method="kmeans", k=8)

# Foreground mask, and activation-magnitude saliency
fl.grid(["dinov2_vitb14"], "img.jpg", layers=[-1], method="foreground")
fl.visualize("dinov2_vitb14", "img.jpg", layers=[2, 5, 8, 11], method="saliency")
```

`seed` is **normalized image coordinates** `(x, y) ∈ [0, 1]`, so it is independent of model and
resolution. The seed patch is marked white in the heatmap.

`cosine` heatmaps carry a shared **[-1, 1] colorbar** and `saliency` a **[0, 1] colorbar** so the
scale is readable, and `kmeans` adds a **cluster-color legend**. (k-means runs per tile, so its
colors are a per-tile key, not comparable across tiles.) `pca` / `foreground` have no colorbar.

## Return the arrays — `return_data=True`

By default the renderers write a PNG (and return its path). Pass `return_data=True` to also get the
arrays back — no PNG round-trip:

```python
res = fl.visualize("dino_vitb16", "img.jpg", layers=[2, 5, 8, 11], method="cosine", return_data=True)
res["tiles"]    # [R, C, H, W, 3] rendered RGB
res["scalars"]  # [R, C, h, w] cosine similarity in [-1, 1] (None for pca / kmeans / foreground)
res["row_labels"], res["col_labels"], res["path"], res["fig"]
```

`correspond(..., return_data=True)` returns `similarity` (one `[h, w]` map per seed), `seeds`,
`matches`, and `mutual` flags.

## Batch / directory mode

Render **one figure per image** over a directory, glob, or list with `featlens.batch(...)`. The
model is built once and reused across every image, and `cache=True` still applies:

```python
fl.batch("dino_vitb16", "photos/", "out/", layers=[2, 5, 8, 11])      # a folder
fl.batch("dino_vitb16", "photos/*.jpg", "out/", method="cosine", seed=(0.5, 0.5))  # a glob
```

`mode=` selects the per-image view (`"grid"`, `"visualize"`, or `"compare"`); other keywords
(`method`, `k`, `colormap`, `cache`, `img_size`, …) are forwarded to the grid. Each output is named
after its source image (`out/<stem>.png`). On the CLI, pass `--out-dir`:

```bash
featlens --models dino_vitb16 --layers 2 5 8 11 --images photos/ --out-dir out/
```

Pass `montage="sheet.png"` to also tile the per-image outputs into one contact sheet, and
`return_data=True, include_features=True` to get the raw `[R, B, L, D, h, w]` feature stack back.

## Multi-frame video

Render per-frame feature maps over a clip as a **filmstrip** (frames × layers) and an animated
**GIF**. `src` is a video file (needs the `featlens[video]` extra) or a directory / glob / list of
frames:

```python
fl.video("dinov2_vitb14", "clip.mp4", layers=[5, 11], n_frames=16, out="strip.png")  # -> strip.png + strip.gif
fl.video("dino_vitb16", "frames/", method="cosine", seed=(0.5, 0.5), n_frames=12)
fl.video("vjepa2_1_vitb16", "clip.mp4", n_frames=16, img_size=384, out="strip.png")  # temporal V-JEPA
```

Temporal models (V-JEPA) feed the whole clip once and split the spatiotemporal tokens into
per-time-step grids; any other model runs each frame independently. With `method="pca"` (the
default) one PCA basis is shared across all frames (`share_pca=True`) so the colors stay consistent
over time and motion is readable — set `share_pca=False` for an independent per-frame basis. A
runnable end-to-end V-JEPA example is in [`examples/vjepa_video.py`](https://github.com/turhancan97/FeatLens/blob/main/examples/vjepa_video.py)
— it writes the filmstrip, the animated feature-map GIF, and a side-by-side GIF of the input clip
next to its feature map.

## Attention-rollout

For a **timm ViT**, compose the per-block self-attention (Abnar & Zuidema rollout) into a
"where is the `[CLS]` token looking" heatmap:

```python
fl.attention("dino_vitb16", "img.jpg", layer=-1, overlay=True, out="attn.png")
```

`head` picks the head reduction (`mean`/`max`/`min`), `discard_ratio` drops the weakest attentions
before rollout. Non-ViT / temporal models raise a clear error.

## Cross-image correspondence

Seed a patch in image A and find the matching patches in image B:

```python
fl.correspond("dino_vitb16", "a.jpg", "b.jpg", layer=-1, seed=(0.4, 0.5), topk=3, out="corr.png")

# Multiple seeds (each gets its own color) + mutual-NN filtering of spurious matches
fl.correspond("dino_vitb16", "a.jpg", "b.jpg", seed=[(0.4, 0.5), (0.6, 0.3)], mutual=True, out="corr.png")
```

Three panels: image A with the seed(s) marked, the original image B with the top-`topk` matches
circled and arrows from each seed, and image B's cosine-similarity heatmap. **`seed`** accepts a
single `(x, y)` or a **list**. **`mutual=True`** keeps only **cycle-consistent** matches — a match
patch in B whose own nearest neighbour back in A is the seed patch — which filters out spurious hits.

```bash
featlens --mode correspond --models dino_vitb16 --images a.jpg --image-b b.jpg \
    --seed 0.4 0.5 --topk 3 --mutual --out corr.png
```

## Caching

Extraction is the slow part; re-coloring is cheap. Pass `cache=True` to cache the extracted
features on disk, keyed on the image *content* + model + size + layers (editing an image
invalidates its entry). Re-renders — and the interactive demo, where the same image is re-colored
on every click — become instant.

```python
fl.grid(["dino_vitb16"], "img.jpg", layers=[2, 5, 8, 11], method="cosine", cache=True)
```

The cache directory is `$FEATLENS_CACHE_DIR` if set, else `~/.cache/featlens`. On the CLI use
`--cache`. It is bounded to **2 GiB** by default — once exceeded, the least-recently-used entries
are evicted on the next write. Override the limit with `$FEATLENS_CACHE_MAX_BYTES` (in bytes; set
it to `0` for no limit).

## Interactive demo

Try it in the browser on the
[**🤗 live demo**](https://huggingface.co/spaces/turhancan97/FeatLens-demo): pick a model, layer
and method, and in `cosine` mode **click the image** to move the seed patch. A second tab does
correspondence between two images.

The same app lives in [`demo/`](https://github.com/turhancan97/FeatLens/tree/main/demo) to run
locally:

```bash
pip install "featlens[timm]" gradio
python demo/app.py
```
