"""VGGT (Wang et al., 2025): feature maps from its single-image patch encoder.

VGGT's aggregator first encodes every frame with a DINOv2 ViT-L/14 with 4 register tokens
(``aggregator.patch_embed``), then mixes frames in its multi-view blocks. This loader exposes that
per-image encoder, the part probed for single images, and reads its 24 blocks at ``norm2``, the
same point FeatLens reads for timm ViTs. Images are normalized with ImageNet mean/std, exactly as
VGGT's aggregator does before ``patch_embed``.

    git clone https://github.com/facebookresearch/vggt
    VGGT_REPO=./vggt python examples/backbones/load_vggt.py --image my.jpg

The 1B checkpoint (~5 GB) downloads once into $FEATLENS_CACHE.
"""

from __future__ import annotations

import torch
from torch import nn

from common import cli, download, norm2_hook, require_repo

from featlens.adapters import external_adapter

CHECKPOINT = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


class _PatchEncoder(nn.Module):
    """Runs only VGGT's per-image DINOv2 encoder; ``blocks`` is exposed for hooking."""

    def __init__(self, vggt):
        super().__init__()
        self.encoder = vggt.aggregator.patch_embed
        self.blocks = self.encoder.blocks

    def forward(self, x):
        return self.encoder.forward_features(x)


def load(img_size: int = 224, repo_dir: str | None = None):
    repo_dir = repo_dir or require_repo("VGGT_REPO", "https://github.com/facebookresearch/vggt")

    def build():
        from vggt.models.vggt import VGGT  # importable once repo_dir is on sys.path

        model = VGGT()
        model.load_state_dict(torch.load(download(CHECKPOINT, "vggt_1b.pt"), map_location="cpu"))
        return _PatchEncoder(model.eval())

    lm = external_adapter.load(repo_dir, build, hook_target=norm2_hook("blocks"), num_blocks=24,
                               patch_size=14, name="vggt_l14")
    assert len(lm.model.blocks) == lm.num_blocks, "unexpected VGGT encoder depth"
    return lm


if __name__ == "__main__":
    cli(load, "Feature maps from VGGT's per-image encoder (ViT-L/14 + registers).")
