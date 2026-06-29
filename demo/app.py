"""FeatLens interactive demo (Gradio).

Two tabs:
- **Feature views** — pick a model, layer and method (PCA / cosine / k-means / foreground) and
  render a feature map. In ``cosine`` mode, *click the image* to move the seed patch and get a
  live similarity heatmap.
- **Correspondence** — *click* a patch in image A and find the matching patches in image B.

Run locally::

    pip install "featlens[timm]" gradio
    python demo/app.py

Deploy as a HuggingFace Space: push this folder (``app.py`` + ``requirements.txt`` + ``README.md``)
to a Gradio Space. The first render of a model downloads its weights.
"""

from __future__ import annotations

import os
import tempfile

import gradio as gr

import featlens as ll
from featlens.registry import BACKBONE_REGISTRY

MODELS = list(BACKBONE_REGISTRY.keys())
# Default to a small ViT-S backbone so the very first render is a quick download.
_PREFERRED = ["dinov2_vits14", "dino_vits16", "dinov3_vits16"]
DEFAULT_MODEL = next((m for m in _PREFERRED if m in MODELS), MODELS[0])
METHODS = ["pca", "cosine", "kmeans", "foreground"]

HERE = os.path.dirname(os.path.abspath(__file__))


def _ex(name: str) -> str:
    return os.path.join(HERE, "examples", "images", name)


EXAMPLE_IMAGES = [_ex(n) for n in ("cat.jpg", "coffee.jpg", "astronaut.jpg")]
# Correspondence works on two *different* views of the same thing: a dog pair and a cat pair.
CORRESPOND_PAIRS = [[_ex("dog1.jpg"), _ex("dog2.jpg")],
                    [_ex("cat1.jpg"), _ex("cat2.jpg")]]


def _depth_for(model: str) -> int:
    """Number of transformer blocks, inferred from the model name (no weights loaded).

    ViT-S/B have 12 blocks, ViT-L has 24, ViT-g has 40. Keeps the layer slider in range
    so picking e.g. layer 15 on a 12-block ViT-B can't raise an out-of-range error.
    """
    m = model.lower()
    if "vitg" in m or "giant" in m:
        return 40
    if "vitl" in m or "large" in m:
        return 24
    return 12  # vits / vitb / small / base


def _clamp_layer(model, layer) -> int:
    return max(0, min(int(layer), _depth_for(model) - 1))


def on_model_change(model, layer):
    """Resize the layer slider to the selected model's block count, clamping the value."""
    depth = _depth_for(model)
    return gr.update(maximum=depth - 1, value=min(int(layer), depth - 1))


def on_method_change(method):
    """Show only the controls the chosen method actually uses (k for kmeans, seed for cosine)."""
    is_kmeans = method == "kmeans"
    is_cosine = method == "cosine"
    return (
        gr.update(visible=is_kmeans),  # k slider
        gr.update(visible=is_cosine),  # seed hint
        gr.update(visible=is_cosine),  # seed readout
    )


def _tmp_png() -> str:
    return tempfile.NamedTemporaryFile(suffix=".png", delete=False).name


def _seed_caption(seed) -> str:
    x, y = seed
    return f"<sub>🎯 seed at x={x:.2f}, y={y:.2f}</sub>"


def render_view(image, model, layer, method, k, seed):
    if not image:
        return None
    out = _tmp_png()
    layer = _clamp_layer(model, layer)
    try:
        ll.visualize(
            model, image, layers=[int(layer)], out=out,
            method=method, k=int(k), seed=tuple(seed),
            img_size=224, cache=True, device=None,
        )
    except Exception as e:  # e.g. layer index out of range for a 12-block ViT-S/B
        raise gr.Error(f"{type(e).__name__}: {e}")
    return out


def on_click(image, evt: gr.SelectData):
    """Image click -> (normalized seed, caption). Shared by both tabs."""
    if image is None or evt is None:
        seed = (0.5, 0.5)
    else:
        h, w = image.shape[:2]
        x, y = evt.index  # (col, row) in pixels
        seed = (round(x / max(w, 1), 3), round(y / max(h, 1), 3))
    return seed, _seed_caption(seed)


def render_correspond(image_a, image_b, model, layer, seed, topk):
    if not image_a or not image_b:
        return None
    out = _tmp_png()
    layer = _clamp_layer(model, layer)
    try:
        ll.correspond(
            model, image_a, image_b, layer=int(layer),
            seed=tuple(seed), topk=int(topk), img_size=224, out=out,
        )
    except Exception as e:
        raise gr.Error(f"{type(e).__name__}: {e}")
    return out


with gr.Blocks(title="FeatLens") as demo:
    gr.Markdown(
        "# 🔎 FeatLens — see what any vision model encodes\n"
        "Render **feature maps** from any layer of any vision model, colored by **PCA**, "
        "**cosine-similarity** to a seed patch, **k-means** segmentation, or a **foreground** "
        "mask. In **cosine** mode, click the image to move the seed patch.\n\n"
        "[GitHub](https://github.com/turhancan97/FeatLens) · "
        "[PyPI](https://pypi.org/project/featlens/) · "
        "[Docs](https://turhancan97.github.io/FeatLens/)\n\n"
        "*The first render of a model downloads its weights (a few seconds for the default "
        "ViT-S); repeat renders on the same image are cached.*"
    )

    with gr.Tab("Feature views"):
        seed = gr.State((0.5, 0.5))  # set by clicking the image, not typed
        with gr.Row():
            with gr.Column():
                img = gr.Image(label="Image", type="filepath")
                model = gr.Dropdown(MODELS, value=DEFAULT_MODEL, label="Model")
                method = gr.Radio(METHODS, value="pca", label="Method")
                layer = gr.Slider(0, _depth_for(DEFAULT_MODEL) - 1, value=11, step=1,
                                  label="Layer (block index)")
                k = gr.Slider(2, 16, value=6, step=1, label="clusters (k)", visible=False)
                seed_hint = gr.Markdown("👆 *Click the image to set the seed patch.*",
                                        visible=False)
                seed_readout = gr.Markdown(_seed_caption((0.5, 0.5)), visible=False)
                go = gr.Button("Render", variant="primary")
            out = gr.Image(label="Feature map")

        gr.Examples(examples=[[p] for p in EXAMPLE_IMAGES], inputs=img, label="Example images")

        # A filepath Image needs a numpy copy to read click coords; load it for the select event.
        img_np = gr.Image(visible=False, type="numpy")
        img.change(lambda p: p, img, img_np)

        inputs = [img, model, layer, method, k, seed]
        method.change(on_method_change, method, [k, seed_hint, seed_readout])
        model.change(on_model_change, [model, layer], layer)
        go.click(render_view, inputs, out)
        for comp in (model, method, layer, k):
            comp.change(render_view, inputs, out)
        # Click the image to set the seed (cosine), then re-render.
        img_np.select(on_click, img_np, [seed, seed_readout]).then(render_view, inputs, out)

    with gr.Tab("Correspondence"):
        cseed = gr.State((0.4, 0.5))  # set by clicking image A
        with gr.Row():
            a = gr.Image(label="Image A — click to set the seed", type="filepath")
            b = gr.Image(label="Image B (target)", type="filepath")
        cseed_readout = gr.Markdown(_seed_caption((0.4, 0.5)) + " — click image A to move it")
        with gr.Row():
            cmodel = gr.Dropdown(MODELS, value=DEFAULT_MODEL, label="Model")
            clayer = gr.Slider(0, _depth_for(DEFAULT_MODEL) - 1, value=11, step=1, label="Layer")
            ctopk = gr.Slider(1, 10, value=3, step=1, label="top-k matches")
        cgo = gr.Button("Match", variant="primary")
        cout = gr.Image(label="Correspondence")

        # Hidden numpy copy of A so the click event can read pixel coords (same trick as above).
        a_np = gr.Image(visible=False, type="numpy")
        a.change(lambda p: p, a, a_np)

        cinputs = [a, b, cmodel, clayer, cseed, ctopk]
        cmodel.change(on_model_change, [cmodel, clayer], clayer)
        cgo.click(render_correspond, cinputs, cout)
        # Click image A to set the seed and re-run the match (once both images are loaded).
        a_np.select(on_click, a_np, [cseed, cseed_readout]).then(render_correspond, cinputs, cout)

        # NOTE: a multi-image gr.Examples renders as a table whose image cells collapse to
        # nothing, so the pairs were invisible. Load each pair into A/B with a button instead.
        gr.Markdown("**Example pairs** — load two views of the same animal, then click image A:")
        with gr.Row():
            dog_btn = gr.Button("🐕 Dog pair")
            cat_btn = gr.Button("🐈 Cat pair")
        dog_btn.click(lambda: tuple(CORRESPOND_PAIRS[0]), None, [a, b])
        cat_btn.click(lambda: tuple(CORRESPOND_PAIRS[1]), None, [a, b])


# Serialize requests: on a free-CPU Space, two people rendering a ViT-L at once can stall or OOM
# the box. The queue runs them one at a time and shows each visitor their position.
demo.queue(max_size=24)


if __name__ == "__main__":
    import os
    # HF Spaces serves the app on 0.0.0.0:7860; bind explicitly so launch() never
    # falls back to wanting a share link. Locally this is reachable at localhost:7860.
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
