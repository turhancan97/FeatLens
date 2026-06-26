"""HuggingFace ``transformers`` backend.

HF vision encoders expose every layer directly via ``output_hidden_states=True`` — no
hooks needed. ``hidden_states`` has length ``num_layers + 1`` (index 0 is the embedding
output), so block index ``i`` maps to ``hidden_states[i + 1]`` (``hidden_states_offset=1``).
"""

from __future__ import annotations

from .base import LoadedModel
from ..preprocess import IMAGENET_MEAN, IMAGENET_STD


def load(identifier: str, img_size: int = 224, pretrained: bool = True) -> LoadedModel:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "The hf backend requires transformers. Install it with "
            "`pip install \"featlens[hf]\"`."
        ) from exc

    model = AutoModel.from_pretrained(identifier)
    model.eval()
    cfg = model.config

    embed_dim = int(getattr(cfg, "hidden_size", getattr(cfg, "embed_dim", 0)))
    if embed_dim <= 0:
        raise ValueError(f"Could not infer hidden size for HF model '{identifier}'.")
    num_blocks = int(getattr(cfg, "num_hidden_layers", 0))
    if num_blocks <= 0:
        raise ValueError(f"Could not infer depth for HF model '{identifier}'.")
    patch_size = int(getattr(cfg, "patch_size", 16))

    # Prefer the model's image processor normalization if available.
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    try:
        from transformers import AutoImageProcessor

        proc = AutoImageProcessor.from_pretrained(identifier)
        if getattr(proc, "image_mean", None) and getattr(proc, "image_std", None):
            mean, std = tuple(proc.image_mean), tuple(proc.image_std)
    except Exception:
        pass

    return LoadedModel(
        model=model,
        num_blocks=num_blocks,
        embed_dim=embed_dim,
        patch_size=patch_size,
        mean=mean,
        std=std,
        mode="hidden_states",
        hidden_states_offset=1,
        name=identifier,
    )
