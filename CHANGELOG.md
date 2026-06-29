# Changelog

All notable changes to **FeatLens** are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/turhancan97/FeatLens/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/turhancan97/FeatLens/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/turhancan97/FeatLens/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/turhancan97/FeatLens/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/turhancan97/FeatLens/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/turhancan97/FeatLens/releases/tag/v0.1.0
