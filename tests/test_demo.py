"""Smoke test for ``demo/app.py``.

CI installs the package but never touches ``demo/``, so a broken demo only surfaces on the
HuggingFace Space. This imports the app with ``gradio`` stubbed (the UI builds against a mock,
no real gradio install needed) and checks the layer-depth logic for every registry model.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "demo" / "app.py"

# Ground-truth transformer depth (number of blocks) per registry model: ViT-S/B = 12, ViT-L = 24.
# Independent of _depth_for's name heuristic, so it catches a misclassification or a new model.
EXPECTED_DEPTH = {
    "dinov3_vitl16": 24, "dinov3_vitb16": 12, "dinov3_vits16": 12,
    "dinov2_vitl14": 24, "dinov2_vitb14": 12, "dinov2_vits14": 12,
    "dino_vitb16": 12, "dino_vits16": 12,
    "mae_vitl16": 24, "mae_vitb16": 12,
    "supervised_vitl16": 24, "supervised_vitb16": 12,
    "deit3_small": 12, "deit3_base": 12, "deit3_large": 24,
    "clip_large_openai": 24, "clip_large_laion": 24,
    "siglip_vitl16": 24, "siglip_vitb16": 12,
    "perception_encoder_vitl14": 24, "perception_encoder_vitb16": 12,
    "perception_encoder_vits16": 12,
    "vjepa2_vitl16": 24, "vjepa2_1_vitb16": 12, "vjepa2_1_vitl16": 24,
    "eva02_small_patch14": 12, "eva02_base_patch14": 12, "eva02_large_patch14": 24,
    "samvit_base": 12, "beit_base_patch16": 12,
}


@pytest.fixture
def app(monkeypatch):
    """Import demo/app.py with gradio stubbed so building the Blocks UI can't fail on import."""
    monkeypatch.setitem(sys.modules, "gradio", MagicMock(name="gradio"))
    spec = importlib.util.spec_from_file_location("featlens_demo_app", APP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # runs the whole module, including the `with gr.Blocks()` block
    return mod


def test_demo_imports_and_builds(app):
    assert app.MODELS, "registry produced no models"
    assert app.DEFAULT_MODEL in app.MODELS
    assert app.METHODS == ["pca", "cosine", "kmeans", "foreground", "saliency"]


def test_depth_for_every_registry_model(app):
    from featlens.registry import BACKBONE_REGISTRY

    for name in BACKBONE_REGISTRY:
        assert name in EXPECTED_DEPTH, f"new model {name!r}: add it to EXPECTED_DEPTH"
        assert app._depth_for(name) == EXPECTED_DEPTH[name], name


def test_clamp_layer_stays_in_range(app):
    assert app._clamp_layer("dinov2_vits14", 99) == 11   # 12-block model -> max index 11
    assert app._clamp_layer("dinov2_vitl14", 99) == 23   # 24-block model -> max index 23
    assert app._clamp_layer("dinov2_vits14", -5) == 0


def test_on_method_change_returns_three_updates(app):
    # (k slider, seed hint, seed row) visibility updates.
    assert len(app.on_method_change("kmeans")) == 3
    assert len(app.on_method_change("pca")) == 3
