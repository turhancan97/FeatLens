---
title: FeatLens
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: gradio
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# FeatLens demo

Interactive feature-map visualization for any vision model. See
[github.com/turhancan97/FeatLens](https://github.com/turhancan97/FeatLens).

- **Feature views** — PCA / cosine-similarity / k-means / foreground maps from any layer. In
  *cosine* mode, click the image to move the seed patch.
- **Correspondence** — seed a patch in image A, find the matching patches in image B.

## Run locally

```bash
pip install "featlens[timm,hf]" gradio
python app.py
```

The first render of a model downloads its weights from the timm / HuggingFace hub.
