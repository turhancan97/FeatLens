"""Adapter registry + dispatch.

``load_spec(spec, ...)`` resolves a string spec (via ``registry.resolve_spec``) to a
``LoadedModel`` using the right backend. The ``custom`` and ``external`` backends are not
string-dispatched — call ``custom_adapter.load(model, ...)`` /
``external_adapter.load(repo_dir, builder, ...)`` directly (they need Python objects).
"""

from __future__ import annotations

from .base import LoadedModel, infer_patch_size, real_index
from . import timm_adapter, hf_adapter, torchhub_adapter, custom_adapter, external_adapter
from ..registry import resolve_spec

_STRING_BACKENDS = {
    "timm": timm_adapter.load,
    "hf": hf_adapter.load,
    "hub": torchhub_adapter.load,
}


def load_spec(spec: str, img_size: int = 224, pretrained: bool = True, **kwargs) -> LoadedModel:
    backend, ident = resolve_spec(spec)
    if backend not in _STRING_BACKENDS:
        raise ValueError(
            f"Backend '{backend}' is not string-dispatchable. Use "
            "custom_adapter.load(model, ...) or external_adapter.load(repo_dir, builder, ...)."
        )
    return _STRING_BACKENDS[backend](ident, img_size=img_size, pretrained=pretrained, **kwargs)


__all__ = [
    "LoadedModel",
    "infer_patch_size",
    "real_index",
    "load_spec",
    "timm_adapter",
    "hf_adapter",
    "torchhub_adapter",
    "custom_adapter",
    "external_adapter",
]
