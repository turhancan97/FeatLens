# Changelog

All notable changes to **FeatLens** are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.6] - 2026-06-29

### Added
- **`saliency` method** — a per-patch L2-norm "where the model fires" heatmap (normalized to
  [0, 1]), available everywhere `method=` is (`--method saliency` on the CLI, and in the demo). Gets
  a [0, 1] colorbar like the other scalar views.
- **`return_data=True`** on `visualize` / `compare` / `grid` / `correspond` — returns a dict with the
  rendered RGB `tiles` **and the underlying scalar field** (`scalars`: cosine similarity / saliency
  magnitudes; `None` for pca / kmeans / foreground), plus labels and the `path` / `fig`. Use FeatLens
  output in a notebook or pipeline without re-reading PNGs. Existing calls are unchanged.
- **Correspondence: multi-seed + mutual-NN.** `correspond(seed=...)` now accepts a **list** of seeds
  (each gets its own color); `mutual=True` (CLI `--mutual`) keeps only **cycle-consistent** matches —
  a match in B whose own nearest neighbour back in A is the seed — filtering out spurious matches.

## [0.2.5] - 2026-06-29

### Added
- **Batch / directory mode** — new top-level `featlens.batch(models, images, out_dir, ...)` and a
  `--out-dir` CLI flag: render **one figure per image** over a directory, glob, or list. The grid is
  built once (weights load a single time, reused across images), and the opt-in feature cache still
  applies. Works for `grid` / `visualize` / `compare` modes.
- **Readable scales** for the methods whose colors carry meaning: `cosine` heatmaps now get a shared
  **[-1, 1] colorbar** (also in `correspond()`'s similarity panel), and `kmeans` gets a
  **cluster-color legend**. PCA keeps no colorbar (its RGB axes are arbitrary).
- README/docs gallery: a **backbone comparison** panel — six ViT-B/16 models (DINO, DINOv3, MAE,
  SigLIP, supervised, Perception Encoder) rendered on the same image (`market.jpg`) at 1024px,
  last-layer PCA maps on a 64×64 grid, so the differences are purely the training objective.
  Reproducible via `examples/quickstart.py`.

## [0.2.4] - 2026-06-29

### Changed
- Refreshed the README/docs showcase: the per-image hero rows are now rendered with a *patch-8*
  backbone (`vit_small_patch8_224.dino`) at 768 px — a fine 96×96 feature grid that resolves thin
  structures (whiskers, feather barbs, individual fruit). New high-res CC-licensed example images
  (peacock, cat, market) with attribution in `examples/images/CREDITS.md`.
- The `correspond()` showcase now matches two *related* images — a real cat's eye to a watercolor
  cat (photo→illustration) — instead of the unrelated cat/coffee pair.

## [0.2.3] - 2026-06-29

### Changed
- `correspond()` is now a **three-panel** view: source (seed), the **original** target image
  with the matches circled and arrows landing on them, and a separate cosine-similarity heatmap
  with the same matches circled. Each match has its own color, shared between the target and
  heatmap panels so they line up at a glance.

## [0.2.2] - 2026-06-29

### Added
- `correspond()` now draws **arrows from the seed to each top-k match** across the two panels
  (the best match is drawn boldest). Toggle with `arrows=False`.

### Changed
- Demo: the seed is now set by **clicking the image** (both tabs) instead of typing x/y
  numbers; the Correspondence tab re-runs the match automatically on click.

## [0.2.1] - 2026-06-28

### Added
- Feature cache is now bounded (default **2 GiB**, LRU eviction); override with
  `$FEATLENS_CACHE_MAX_BYTES` (`0` = unlimited).
- Demo smoke test (`tests/test_demo.py`) so CI catches a broken `demo/app.py`.

### Changed
- Demo: serialize requests with a queue; show only the controls each method uses (k for
  k-means, seed for cosine); render once per image click; dog/cat correspondence example pairs.

## [0.2.0] - 2026-06-26

Beyond PCA, now interactive — four new ways to look at the same `[B, L, D, h, w]` feature
stack, an opt-in feature cache, and a hosted demo.

### Added
- **Visualization methods** (`method=` on `grid` / `visualize` / `compare`, `--method` on the
  CLI): `cosine` (cosine similarity to a seed patch), `kmeans` (self-contained k-means
  segmentation, no sklearn/scipy), and `foreground` (fg/bg mask from the first robust-PCA
  component). `pca` remains the default and is byte-identical to v0.1.
- **Cross-image correspondence** — new top-level `featlens.correspond(...)` (and
  `--mode correspond` / `--image-b`): seed a patch in image A and find the top-`k` matching
  patches in image B, rendered side by side.
- **Feature caching** — opt-in `cache=True` (`--cache`) caches extracted features on disk,
  keyed on image *content* + model + size + layers. Cache dir is `$FEATLENS_CACHE_DIR` else
  `~/.cache/featlens`.
- **Interactive demo** in [`demo/`](demo/) — a Gradio app (click-to-seed in cosine mode, a
  correspondence tab) deployed as a [🤗 HuggingFace Space](https://huggingface.co/spaces/turhancan97/FeatLens-demo).
- New CLI flags: `--method`, `--seed X Y`, `--k`, `--colormap`, `--cache`.
- Docs: a [Visualization methods](https://turhancan97.github.io/FeatLens/methods/) page; a
  `demo` extra (`pip install "featlens[demo]"`).

### Changed
- Non-PCA methods colorize on the native patch grid then interpolate to RGB (cheaper and
  crisper than clustering full-resolution pixels). The PCA path is unchanged.

## [0.1.0] - 2026-06-26

### Added
- Initial release: model-agnostic dense feature-map visualization as a **model × layer** grid,
  colored by robust **PCA → RGB**.
- `FeatureExtractor.forward() → [B, L, D, h, w]` — the single contract the visualization layer
  consumes — over an adapter layer spanning timm, HuggingFace `transformers`, `torch.hub`,
  external repos, and custom `nn.Module`s.
- Public API (`grid` / `visualize` / `compare`), a `featlens` CLI, a friendly model registry,
  and MkDocs documentation.

[Unreleased]: https://github.com/turhancan97/FeatLens/compare/v0.2.6...HEAD
[0.2.6]: https://github.com/turhancan97/FeatLens/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/turhancan97/FeatLens/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/turhancan97/FeatLens/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/turhancan97/FeatLens/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/turhancan97/FeatLens/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/turhancan97/FeatLens/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/turhancan97/FeatLens/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/turhancan97/FeatLens/releases/tag/v0.1.0
