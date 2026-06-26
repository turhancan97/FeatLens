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
short_description: PCA / cosine / k-means feature maps from any vision model
---

# 🔎 FeatLens demo

Interactive **feature-map visualization** for any vision model — DINO, DINOv2/v3, CLIP, SigLIP,
MAE, DeiT, … — from **any layer**. Powered by [`featlens`](https://pypi.org/project/featlens/)
([GitHub](https://github.com/turhancan97/FeatLens) ·
[docs](https://turhancan97.github.io/FeatLens/)).

- **Feature views** — color the features by **PCA**, **cosine-similarity** to a seed patch,
  **k-means** segmentation, or a **foreground** mask. Pick a model and layer; in *cosine* mode,
  **click the image** to move the seed patch.
- **Correspondence** — seed a patch in image A and find the matching patches in image B.

Click an **example image** to get started. The default model is a small ViT-S so the first
render is quick; pick a larger backbone from the dropdown for sharper maps.

> Running on free CPU: the **first** render of a model downloads its weights (a few seconds for
> ViT-S, longer for ViT-L) and repeat renders on the same image are cached.

## Run locally

```bash
pip install "featlens[timm]" gradio
python app.py
```
