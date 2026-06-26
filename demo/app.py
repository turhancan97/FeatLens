"""FeatLens interactive demo (Gradio).

Two tabs:
- **Feature views** — pick a model, layer and method (PCA / cosine / k-means / foreground) and
  render a feature map. In ``cosine`` mode, *click the image* to move the seed patch and get a
  live similarity heatmap.
- **Correspondence** — seed a patch in image A and find the matching patches in image B.

Run locally::

    pip install -e ".[timm,hf]" gradio
    python demo/app.py

Deploy as a HuggingFace Space: push this folder (``app.py`` + ``requirements.txt`` + ``README.md``)
to a Gradio Space. The first render of a model downloads its weights.
"""

from __future__ import annotations

import tempfile

import gradio as gr

import featlens as ll
from featlens.registry import BACKBONE_REGISTRY

MODELS = list(BACKBONE_REGISTRY.keys())
DEFAULT_MODEL = "dino_vitb16" if "dino_vitb16" in MODELS else MODELS[0]
METHODS = ["pca", "cosine", "kmeans", "foreground"]


def _tmp_png() -> str:
    return tempfile.NamedTemporaryFile(suffix=".png", delete=False).name


def render_view(image, model, layer, method, k, seed_x, seed_y):
    if image is None:
        return None
    out = _tmp_png()
    ll.visualize(
        model, image, layers=[int(layer)], out=out,
        method=method, k=int(k), seed=(seed_x, seed_y),
        img_size=224, cache=True, device=None,
    )
    return out


def on_click(image, evt: gr.SelectData):
    """Click on the image -> normalized seed coords (for cosine mode)."""
    if image is None or evt is None:
        return 0.5, 0.5
    h, w = image.shape[:2]
    x, y = evt.index  # (col, row) in pixels
    return round(x / max(w, 1), 3), round(y / max(h, 1), 3)


def render_correspond(image_a, image_b, model, layer, seed_x, seed_y, topk):
    if image_a is None or image_b is None:
        return None
    out = _tmp_png()
    ll.correspond(
        model, image_a, image_b, layer=int(layer),
        seed=(seed_x, seed_y), topk=int(topk), img_size=224, out=out,
    )
    return out


with gr.Blocks(title="FeatLens") as demo:
    gr.Markdown(
        "# 🔎 FeatLens — see what any vision model encodes\n"
        "PCA / cosine-similarity / k-means / foreground feature maps from any layer. "
        "In **cosine** mode, click the image to move the seed patch."
    )

    with gr.Tab("Feature views"):
        with gr.Row():
            with gr.Column():
                img = gr.Image(label="Image", type="filepath")
                model = gr.Dropdown(MODELS, value=DEFAULT_MODEL, label="Model")
                method = gr.Radio(METHODS, value="pca", label="Method")
                layer = gr.Slider(0, 23, value=11, step=1, label="Layer (block index)")
                k = gr.Slider(2, 16, value=6, step=1, label="k (kmeans)")
                with gr.Row():
                    seed_x = gr.Number(0.5, label="seed x")
                    seed_y = gr.Number(0.5, label="seed y")
                go = gr.Button("Render", variant="primary")
            out = gr.Image(label="Feature map")

        # A filepath Image needs a numpy copy to read click coords; load it for the select event.
        img_np = gr.Image(visible=False, type="numpy")
        img.change(lambda p: p, img, img_np)
        img_np.select(on_click, img_np, [seed_x, seed_y])

        inputs = [img, model, layer, method, k, seed_x, seed_y]
        go.click(render_view, inputs, out)
        for comp in (model, method, layer, k, seed_x, seed_y):
            comp.change(render_view, inputs, out)

    with gr.Tab("Correspondence"):
        with gr.Row():
            a = gr.Image(label="Image A (seed)", type="filepath")
            b = gr.Image(label="Image B (target)", type="filepath")
        with gr.Row():
            cmodel = gr.Dropdown(MODELS, value=DEFAULT_MODEL, label="Model")
            clayer = gr.Slider(0, 23, value=11, step=1, label="Layer")
            csx = gr.Number(0.4, label="seed x")
            csy = gr.Number(0.5, label="seed y")
            ctopk = gr.Slider(1, 10, value=3, step=1, label="top-k matches")
        cgo = gr.Button("Match", variant="primary")
        cout = gr.Image(label="Correspondence")
        cgo.click(render_correspond, [a, b, cmodel, clayer, csx, csy, ctopk], cout)


if __name__ == "__main__":
    import os
    # HF Spaces serves the app on 0.0.0.0:7860; bind explicitly so launch() never
    # falls back to wanting a share link. Locally this is reachable at localhost:7860.
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
