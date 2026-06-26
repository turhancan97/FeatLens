"""FeatureExtractor: load any model and pull dense feature maps from any layer.

``forward(images) -> [B, L, D, H_feat, W_feat]`` — a stack of per-layer dense maps. This is
the single contract the visualization layer consumes. Derived from the proven
``FrozenBackbone`` access pattern, generalized across backends via the adapter layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

from .adapters import LoadedModel, load_spec, real_index
from .tokens import tokens_to_grid
from .preprocess import build_transform, denormalize


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        model: Union[str, LoadedModel],
        layers: Optional[Sequence[int]] = None,
        img_size: int = 224,
        pretrained: bool = True,
        frozen: bool = True,
        **load_kwargs,
    ):
        super().__init__()
        if isinstance(model, LoadedModel):
            self.lm = model
        elif isinstance(model, str):
            self.lm = load_spec(model, img_size=img_size, pretrained=pretrained, **load_kwargs)
        else:
            raise TypeError(
                "Pass a model spec string or a LoadedModel. For a raw nn.Module use "
                "layerlens.adapters.custom_adapter.load(...) / external_adapter.load(...)."
            )

        self.model = self.lm.model
        self.name = self.lm.name
        self.img_size = int(img_size)
        self.patch_size = int(self.lm.patch_size)
        self.mean, self.std = self.lm.mean, self.lm.std

        if self.lm.mode == "callable":
            # A feature_fn yields a single feature map; it's one "layer".
            self.layers = [0]
        else:
            self.layers = [int(x) for x in (layers if layers is not None else [-1])]
            for idx in self.layers:
                real_index(idx, self.lm.num_blocks)  # validate range (allows negatives)

        self._features = {}
        self._hook_handles = []
        if self.lm.mode == "hook":
            self._register_hooks()

        if frozen:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    # ---- transforms / IO -------------------------------------------------
    @property
    def transform(self):
        return build_transform(self.img_size, self.mean, self.std)

    def load_images(self, paths: Sequence[Union[str, Path]]) -> torch.Tensor:
        from PIL import Image

        tf = self.transform
        tensors = []
        for p in paths:
            with Image.open(p) as im:
                tensors.append(tf(im.convert("RGB")))
        return torch.stack(tensors, dim=0)

    def denormalize(self, image: torch.Tensor):
        return denormalize(image, self.mean, self.std)

    # ---- hooks -----------------------------------------------------------
    def _register_hooks(self):
        for idx in self.layers:
            module = self.lm.hook_module_fn(self.model, idx)
            self._hook_handles.append(module.register_forward_hook(self._make_hook(idx)))

    def _make_hook(self, layer_idx: int):
        store = self._features

        def hook(_module, _inp, output):
            tensor = output
            if isinstance(output, (tuple, list)):
                tensor = next((o for o in output if torch.is_tensor(o)), None)
            if not torch.is_tensor(tensor):
                raise RuntimeError(f"Hook at layer {layer_idx} got non-tensor output.")
            store[layer_idx] = tensor.detach()

        return hook

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # ---- forward ---------------------------------------------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input size {(H, W)} must be divisible by patch_size={self.patch_size}."
            )
        h_feat, w_feat = H // self.patch_size, W // self.patch_size

        if self.lm.mode == "callable":
            maps = [self._extract_callable(images, h_feat, w_feat)]
        elif self.lm.mode == "hidden_states":
            maps = self._extract_hidden_states(images, h_feat, w_feat)
        else:
            maps = self._extract_via_hooks(images, h_feat, w_feat)

        return torch.stack(maps, dim=1)  # [B, L, D, h, w]

    def _run_model(self, images: torch.Tensor, **kwargs):
        inp = images.unsqueeze(2) if self.lm.uses_temporal else images
        with torch.no_grad():
            return self.model(inp, **kwargs)

    def _extract_via_hooks(self, images, h_feat, w_feat) -> List[torch.Tensor]:
        self._features.clear()
        # V-JEPA 2.1 encoders expose an out_layers shortcut that conflicts with arbitrary
        # block hooks; disable it for the duration of the forward.
        prev = getattr(self.model, "out_layers", "__missing__")
        if prev != "__missing__":
            self.model.out_layers = None
        try:
            self._run_model(images)
        finally:
            if prev != "__missing__":
                self.model.out_layers = prev
        out = []
        for idx in self.layers:
            if idx not in self._features:
                raise RuntimeError(f"No hooked feature captured for layer {idx}.")
            grid, _ = tokens_to_grid(self._features[idx], images.shape[0], h_feat, w_feat, idx)
            out.append(grid)
        return out

    def _extract_hidden_states(self, images, h_feat, w_feat) -> List[torch.Tensor]:
        out = self._run_model(images, output_hidden_states=True)
        hs = out.hidden_states  # tuple len num_blocks+1
        result = []
        for idx in self.layers:
            real = real_index(idx, self.lm.num_blocks)
            tokens = hs[real + self.lm.hidden_states_offset]
            grid, _ = tokens_to_grid(tokens, images.shape[0], h_feat, w_feat, idx)
            result.append(grid)
        return result

    def _extract_callable(self, images, h_feat, w_feat) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.lm.feature_fn(self.model, images)
        grid, _ = tokens_to_grid(tokens, images.shape[0], h_feat, w_feat, 0)
        return grid
