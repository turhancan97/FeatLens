---
title: 'FeatLens: Model-agnostic feature-map visualization for vision models'
tags:
  - Python
  - PyTorch
  - computer vision
  - representation learning
  - vision transformers
  - self-supervised learning
  - interpretability
authors:
  - name: Turhan Can Kargin
    orcid: 0000-0002-6751-4773
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 25 November 2026
bibliography: paper.bib
---

<!--
  DRAFT for JOSS. Items marked TODO need the author's input before submission.
  JOSS gates to clear first: >6 months of public history with development spread over that
  period (automated commit-distribution checks), and documented research use.
-->

# Summary

Modern vision models, such as self-supervised Vision Transformers (ViTs)
[@dosovitskiy2021vit; @caron2021dino; @oquab2024dinov2], image–text models
[@radford2021clip; @tschannen2025siglip2] and video encoders [@assran2025vjepa2], produce a
dense grid of high-dimensional feature vectors, one per image patch,
at every layer. These features drive downstream tasks like segmentation, correspondence and
retrieval, yet they are hard to inspect: a single layer of a base-sized ViT holds 196 vectors of
768 numbers each for a 224×224 image. A common way to look at them is to project each patch onto
the top three principal components and display the result as an RGB image [@oquab2024dinov2],
so that patches the model treats as similar share a color.

FeatLens is a Python package that makes this kind of inspection work for any vision model, from
any source, at any layer. A user names a model (for example `"dinov2_vitb14"`, a timm
identifier, a Hugging Face checkpoint or a `torch.hub` entry point) or passes their own
`torch.nn.Module`, picks images and layers, and gets a model × layer grid of feature maps in one
call (\autoref{fig:layers}). Besides robust PCA, features can be colored by cosine similarity to
a seed patch, k-means clustering, a foreground mask, or activation magnitude. The package also
matches patches across two images, computes attention rollout [@abnar2020rollout], processes
image folders, and renders feature maps over video, including temporal models such as
V-JEPA 2 [@assran2025vjepa2]. FeatLens is available on PyPI, documented online, and has a
browser demo.

![Feature maps of DINO ViT-S/8 [@caron2021dino] at layers 2, 5, 8 and 11, colored by robust PCA with one basis shared across layers so that colors are comparable along the row.\label{fig:layers}](feat_peacock.png)

# Statement of need

Researchers who train, select or probe vision backbones routinely want to see what a model
encodes at a given depth: whether a self-supervised model separates objects from background,
how features sharpen across layers, or how two training objectives differ on the same image.
Such views are a fixture of the literature: papers introducing or analysing vision backbones
routinely show PCA or similarity maps of patch features [@oquab2024dinov2; @simeoni2025dinov3;
@amir2021deepvit], and each project ships its own code for them. In practice this code is a short
script written for one model family. Each script hard-codes
how that model's tokens are extracted, which special tokens (class, register [@darcet2024registers]
or distillation tokens) must be removed, how the token sequence is reshaped into a grid, and how
colors are normalized. Comparing models from different libraries, or at different layers, means
rewriting this glue code each time, and small differences in normalization make the resulting
images hard to compare. There is no shared, model-agnostic framework that the community can use
as a common standard for these visualizations; FeatLens aims to be one.

FeatLens targets machine-learning and computer-vision researchers and students who need these
views quickly and consistently across many models. It turns "what does this layer of this model
encode?" into a one-line call, and it gives comparable outputs, with a shared color basis and the
same preprocessing, so that side-by-side figures, such as the backbone comparison in
\autoref{fig:compare}, reflect the models rather than the plotting code.

![Last-layer PCA feature maps of six ViT-B/16 backbones on the same image at 1024 px. Architecture and patch size are fixed, so differences come from the training objective: DINO [@caron2021dino], DINOv3 [@simeoni2025dinov3], MAE [@he2022mae], SigLIP 2 [@tschannen2025siglip2], a supervised ViT [@dosovitskiy2021vit], and Perception Encoder [@bolya2025pe].\label{fig:compare}](compare_b16_market.png){ width=80% }

# State of the field

Tools that explain vision models mostly answer a different question. Attribution libraries such
as Captum [@kokhlikyan2020captum] and pytorch-grad-cam [@gildenblat2021gradcam], built around
methods like Grad-CAM [@selvaraju2017gradcam], show which input pixels drive a particular
prediction. Optimization-based feature visualization [@olah2017feature] synthesizes images that
excite individual units. Neither shows the dense patch-level representation itself, which is what
self-supervised and foundation-model research usually examines. Closest in spirit is the code
released with work on deep ViT features as dense descriptors [@amir2021deepvit], which extracts
and visualizes DINO features but is tied to that model family. Model zoos such as timm
[@wightman2019timm] and Hugging Face Transformers [@wolf2020transformers] provide the backbones
but no visualization layer.

We chose to build a separate package rather than contribute to these projects because the core
of FeatLens is a *cross-library* abstraction: one adapter interface spanning timm, Transformers,
`torch.hub`, external repositories and user-defined modules, which none of those projects has a
reason to own. Attribution libraries are organized around gradients with respect to a model
output, whereas FeatLens needs only forward features and is deliberately gradient-free. For
loading models, FeatLens uses timm, Transformers and `torch.hub` as backends rather than
reimplementing them.

# Software design

FeatLens separates *representation access* from *visualization*, joined by a single contract:
a `FeatureExtractor` returns a tensor of shape `[B, L, D, h, w]` (batch, layers, channels, grid
height and width) for any supported model, built on PyTorch [@paszke2019pytorch].

**Adapters.** Each backend (timm, Hugging Face, `torch.hub`, external repository, custom module)
is a small adapter that loads the model and reports its patch size, normalization, number of
blocks, and how to reach a layer: through forward hooks, through `output_hidden_states`, or
through a user-supplied feature function for models such as CNNs. A registry maps 30 friendly
names to backend identifiers, and explicit `backend:identifier` prefixes cover everything else.
Adding a model family means writing one adapter, not touching the visualization code.

**Token normalization.** One function turns whatever a layer emits into a dense grid. A token
sequence of length *N* is matched against the expected `h × w` patch count; the `N − h·w` prefix
tokens (class, register, distillation) are detected and removed, and the rest is reshaped.
Convolutional feature maps pass through unchanged. For temporal models, the sequence is split
into per-time-step grids. This single choke-point is what keeps the rest of the code
model-agnostic.

**Visualization.** PCA coloring uses a robust range: after projecting onto three components,
the color limits are set from patches within a median-absolute-deviation band and the tails are
clipped, so that the long tails of the projected distribution do not compress the colors of the
bulk of the patches into a narrow, washed-out range. The PCA basis can be fit
per tile or shared across a model's layers (or across video frames), trading per-image contrast
for comparability. The other methods (cosine similarity, a dependency-free k-means, a foreground
mask derived from the first robust component, and feature-norm saliency) operate on the same
`[h, w, D]` maps, so they work for every supported model. Cross-image correspondence uses
cosine nearest neighbors with an optional mutual-nearest-neighbor check.

**Interfaces and quality.** The same functionality is exposed through high-level functions
(`visualize`, `compare`, `grid`, `correspond`, `attention`, `batch`, `video`), a command-line tool
with YAML configuration, and an optional on-disk feature cache. Functions can return the raw
tiles and scalar fields instead of a figure, so results can feed further analysis. A test suite
of 47 tests runs on CPU in continuous integration, and the documentation includes a gallery
generated by a single example script.

# Research impact statement

FeatLens grew out of the author's research on how visual foundation models represent images,
and it is now part of that research workflow. The SpaRRTa benchmark [@kargin2026sparrta]
evaluates whether visual foundation models encode spatial relations between objects, using
probes on frozen backbones. The SpaRRTa project uses FeatLens to look inside the same 13
backbones the benchmark evaluates [@kargin2026sparrtablog]: robust-PCA feature maps of every
encoder on two benchmark environments, and cosine-similarity maps that follow a single patch
through several depths of DINOv2, MAE and VGGT. These views give a qualitative counterpart to the
benchmark's probe accuracies, for example showing which backbones still separate the objects
whose spatial relation is being tested. The figures name the exact FeatLens revision used and
cite the archived release, so they can be regenerated.

As signals of community readiness, FeatLens is distributed on PyPI, archived on Zenodo
[@kargin2026featlens], documented with a gallery and API reference, provides a browser demo
and a narrated video explainer, and ships tagged releases with a changelog, contribution and
support guidelines, a code of conduct, and continuous integration.

# AI usage disclosure

<!-- TODO: the author must confirm every statement below is accurate before submission. -->

Generative AI was used in developing this software and its supporting materials. The author used
Claude Code (Anthropic) with the Claude Opus 4.8 and Claude Opus 5.5 models; 58 of the
repository's 82 commits at the time of writing carry an AI co-author trailer. AI assistance
covered code generation and refactoring, test scaffolding, documentation and examples, the
video explainer, and drafting and copy-editing this manuscript. The author framed the problem,
made the core design decisions, including the adapter/visualization split, the
`[B, L, D, h, w]` contract, and the choice of robust and shared PCA bases, and reviewed, edited
and validated all AI-assisted code, documentation and text.

# Acknowledgements

This work received no specific funding.

# References
