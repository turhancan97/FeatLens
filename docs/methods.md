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

```python
import featlens as fl

# Cosine similarity to the patch at the image center, scrubbed across layers
fl.visualize("dino_vitb16", "img.jpg", layers=[2, 5, 8, 11], method="cosine", seed=(0.5, 0.5))

# k-means segmentation (8 clusters) at the last layer, comparing two models
fl.compare(["dino_vitb16", "dinov2_vitb14"], "img.jpg", layer=-1, method="kmeans", k=8)

# Foreground mask
fl.grid(["dinov2_vitb14"], "img.jpg", layers=[-1], method="foreground")
```

`seed` is **normalized image coordinates** `(x, y) ∈ [0, 1]`, so it is independent of model and
resolution. The seed patch is marked white in the heatmap.

## Cross-image correspondence

Seed a patch in image A and find the matching patches in image B:

```python
fl.correspond("dino_vitb16", "a.jpg", "b.jpg", layer=-1, seed=(0.4, 0.5), topk=3, out="corr.png")
```

Left shows image A with the seed marked; right shows image B as a cosine-similarity heatmap with
the top-`topk` matches circled.

```bash
featlens --mode correspond --models dino_vitb16 --images a.jpg --image-b b.jpg \
    --seed 0.4 0.5 --topk 3 --out corr.png
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
