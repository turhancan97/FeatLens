# FeatLens explainer: narration script and scene outline

**Draft 2 (~3 min cut).**

- **Format:** 3Blue1Brown-style explainer in Manim Community, narrated by Kokoro-82M, voice
  **`af_heart`**. Each narration line is synthesized to its own clip and the animation for that
  line is timed to the clip's length (`voiceover.py`).
- **Length:** about 3 minutes (≈ 440 narrated words at ~150 wpm, plus pauses).
- **Running example:** DINO ViT-B/16 at 224×224 → 14×14 = **196 patches**, **768-dim** tokens,
  **12 layers**. Final "real" images come from the existing `examples/` gallery.
- **Palette roles** (see `palette.py`): tokens = blue · layers = purple · code = teal ·
  emphasis = mustard · seed patch = yellow · outliers / CLS tokens = pink ·
  PC1/PC2/PC3 = orange/green/blue · k-means clusters cycle through all eight colors.
  Background near-black `#16181D`, text off-white. On-screen text uses `Text` (no LaTeX).
- **Credit:** project only, no author name on screen.
- **Subtitles:** burned in at the bottom, phrase by phrase, timed from Kokoro's per-word
  timestamps (numbers shown as digits). The same text is written to `featlens_explainer.srt`.

Each numbered line in **Narration** is one voice-over clip; **Visuals** play while it's spoken.

---

## Scene 1 — Cold open (~15 s)

**Narration**
1. When a vision model looks at this cat, it doesn't see a cat. It sees a grid of numbers, rewritten layer after layer.
2. So how do we see what it has learned? That's what FeatLens is for.

**Visuals**
- Fade in `cat_hires.jpg`; it dissolves into a 14×14 grid of flickering blue number columns.
- The grid stacks into 12 translucent purple-edged layers receding into depth.
- Cut to the FeatLens logo + tagline *"See what any vision model encodes."* (mustard).

---

## Scene 2 — Patches, tokens, and back to a grid (~35 s)

**Narration**
1. A vision transformer cuts the image into square patches. In a typical setup, a 224-pixel image with sixteen-pixel patches gives a fourteen-by-fourteen grid: 196 patches. Other models use other sizes.
2. Each patch becomes a token, a vector of numbers: 768 in a base-sized model like this one, 384 in a small model, 1024 in a large one. Then a stack of transformer blocks, twelve here, rewrites every token in turn.
3. FeatLens can hook any of those layers and copy what it holds.
4. It drops the extra tokens that don't belong to any location, like the class token, and folds the rest back into the fourteen-by-fourteen grid.
5. And it works the same whether the model comes from timm, Hugging Face, torch hub, or your own code.

**Visuals**
- Note above the cat: "typical ViT-B/16 · 224 × 224 image"; after the count, "patch size and image
  size vary from model to model".
- Grid lines draw over the cat; one patch lifts out, flattens, and morphs into a tall blue bar `768`.
- A bar chart of token widths at true proportions: ViT-Ti 192 · ViT-S 384 · **ViT-B 768** · ViT-L 1024.
- All patches fly into a row of blue bars; a pink `CLS` token slots in front; the row passes through 12 purple blocks.
- A purple hook clips onto a block; a copy of the row peels off; the pink token is crossed out.
- The blue bars snake back into a 14×14 grid that tilts into 3D (`D = 768`).
- Source badges (timm / 🤗 / torch.hub / your model) funnel into one `[D, h, w]` box (teal).

---

## Scene 3 — The core trick: 768 dimensions → 3 colors (PCA) (~55 s) · **test scene**

**Narration**
1. Now the problem: we can't see 768 dimensions. We can see exactly three. Red, green and blue.
2. So think of every patch as a point in a 768-dimensional space, where similar patches land close together.
3. Principal component analysis finds the three directions along which these points spread out the most.
4. Project each patch onto them, call the three numbers red, green and blue, and paint the patch.
5. Patches the model sees as similar get similar colors, and the parts of the scene separate.
6. One catch: the far tails of the distribution stretch the color range, and every color fades.
7. So FeatLens sets the range from the bulk of the patches, measured around the median, and clips the tails at the edges.

**Visuals**
- A cloud of ~200 blue dots in rotating 3D axes, labeled "768-D (shown in 3)".
- Three arrows grow from the cloud's center along its spread: orange PC1, green PC2, blue PC3.
- One dot projects onto each arrow → three bars → mix into one RGB swatch.
- All dots take their colors; a 14×14 grid recolors cell by cell into the real DINO feature map of the cat.
- A PC1 number line under the cloud: far-tail patches turn pink; brackets at the plain min/max
  and the grid fades to the real min/max map (paler). The brackets snap in to the robust bounds
  (median ± 2 MAD); the pink ticks clamp to the edges; the grid returns to the vivid robust map.
  (Both maps are real: `video/assets/cat_dino.npz`.)
- End on the real `feat_cat_hires.png` row (layers 2 / 5 / 8 / 11).

---

## Scene 4 — More ways to read the features (~25 s)

**Narration**
1. FeatLens has other views too. Cosine similarity lights up every patch that looks like a seed you pick.
2. K-means groups patches into segments, a foreground mask splits object from background, and saliency shows where the activations are strongest.

**Visuals**
- 2×2 panels on the same grid: a yellow seed spreading into a heatmap · dots splitting into
  palette-colored clusters · a threshold line turning patches white/black · vectors shaded by length.
- Each panel cross-fades into its real `method_*.png` row.

---

## Scene 5 — Across images, models and time (~30 s)

**Narration**
1. The same features match across images. Seed a cat's eye in a photo, and the best match in a watercolor lands on the painted eye.
2. Six models with the same architecture, trained differently, carve up the same scene in very different ways.
3. And with a video model like V-JEPA, one shared color basis lets you watch the bird move.

**Visuals**
- `correspond.png`: a yellow seed on the photo; a yellow arc jumps to the match on the watercolor.
- `compare_b16_market.png`: tiles slide in one by one with model labels.
- `vjepa_video_compare.gif` frames play back.

---

## Scene 6 — One line of code (~20 s)

**Narration**
1. All of it is one line of Python. Pick a model, an image, and the layers.
2. FeatLens is open source and on pip, and you can try it right now in the browser demo. So don't just check what your model scores. Look at what it sees.

**Visuals**
- Teal code types on:
  ```python
  import featlens as ll
  ll.visualize("dinov2_vitb14", "cat.jpg", layers=[2, 5, 8, 11])
  ```
- The output row renders beneath it.
- `pip install featlens` · GitHub `turhancan97/FeatLens` · 🤗 demo `huggingface.co/spaces/turhancan97/FeatLens-demo`.
- End card: logo, tagline, DOI badge. No author credit.
