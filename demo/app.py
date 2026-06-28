"""FeatLens interactive demo (Gradio).

Two tabs:
- **Feature views** — pick a model, layer and method (PCA / cosine / k-means / foreground) and
  render a feature map. In ``cosine`` mode, *click the image* to move the seed patch and get a
  live similarity heatmap.
- **Correspondence** — seed a patch in image A and find the matching patches in image B.

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
EXAMPLE_IMAGES = [os.path.join(HERE, "examples", "images", n)
                  for n in ("cat.jpg", "coffee.jpg", "astronaut.jpg")]


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


def _tmp_png() -> str:
    return tempfile.NamedTemporaryFile(suffix=".png", delete=False).name


def render_view(image, model, layer, method, k, seed_x, seed_y):
    if not image:
        return None
    out = _tmp_png()
    layer = _clamp_layer(model, layer)
    try:
        ll.visualize(
            model, image, layers=[int(layer)], out=out,
            method=method, k=int(k), seed=(seed_x, seed_y),
            img_size=224, cache=True, device=None,
        )
    except Exception as e:  # e.g. layer index out of range for a 12-block ViT-S/B
        raise gr.Error(f"{type(e).__name__}: {e}")
    return out


def on_click(image, evt: gr.SelectData):
    """Click on the image -> normalized seed coords (for cosine mode)."""
    if image is None or evt is None:
        return 0.5, 0.5
    h, w = image.shape[:2]
    x, y = evt.index  # (col, row) in pixels
    return round(x / max(w, 1), 3), round(y / max(h, 1), 3)


def render_correspond(image_a, image_b, model, layer, seed_x, seed_y, topk):
    if not image_a or not image_b:
        return None
    out = _tmp_png()
    layer = _clamp_layer(model, layer)
    try:
        ll.correspond(
            model, image_a, image_b, layer=int(layer),
            seed=(seed_x, seed_y), topk=int(topk), img_size=224, out=out,
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
        with gr.Row():
            with gr.Column():
                img = gr.Image(label="Image", type="filepath")
                model = gr.Dropdown(MODELS, value=DEFAULT_MODEL, label="Model")
                method = gr.Radio(METHODS, value="pca", label="Method")
                layer = gr.Slider(0, _depth_for(DEFAULT_MODEL) - 1, value=11, step=1,
                                  label="Layer (block index)")
                k = gr.Slider(2, 16, value=6, step=1, label="k (kmeans)")
                with gr.Row():
                    seed_x = gr.Number(0.5, label="seed x")
                    seed_y = gr.Number(0.5, label="seed y")
                go = gr.Button("Render", variant="primary")
            out = gr.Image(label="Feature map")

        gr.Examples(examples=[[p] for p in EXAMPLE_IMAGES], inputs=img, label="Example images")

        # A filepath Image needs a numpy copy to read click coords; load it for the select event.
        img_np = gr.Image(visible=False, type="numpy")
        img.change(lambda p: p, img, img_np)
        img_np.select(on_click, img_np, [seed_x, seed_y])

        inputs = [img, model, layer, method, k, seed_x, seed_y]
        model.change(on_model_change, [model, layer], layer)
        go.click(render_view, inputs, out)
        for comp in (model, method, layer, k, seed_x, seed_y):
            comp.change(render_view, inputs, out)

    with gr.Tab("Correspondence"):
        with gr.Row():
            a = gr.Image(label="Image A (seed)", type="filepath")
            b = gr.Image(label="Image B (target)", type="filepath")
        with gr.Row():
            cmodel = gr.Dropdown(MODELS, value=DEFAULT_MODEL, label="Model")
            clayer = gr.Slider(0, _depth_for(DEFAULT_MODEL) - 1, value=11, step=1, label="Layer")
            csx = gr.Number(0.4, label="seed x")
            csy = gr.Number(0.5, label="seed y")
            ctopk = gr.Slider(1, 10, value=3, step=1, label="top-k matches")
        cgo = gr.Button("Match", variant="primary")
        cout = gr.Image(label="Correspondence")
        cmodel.change(on_model_change, [cmodel, clayer], clayer)
        cgo.click(render_correspond, [a, b, cmodel, clayer, csx, csy, ctopk], cout)
        gr.Examples(examples=[[EXAMPLE_IMAGES[0], EXAMPLE_IMAGES[1]]], inputs=[a, b],
                    label="Example pair")


if __name__ == "__main__":
    import os
    # HF Spaces serves the app on 0.0.0.0:7860; bind explicitly so launch() never
    # falls back to wanting a share link. Locally this is reachable at localhost:7860.
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
