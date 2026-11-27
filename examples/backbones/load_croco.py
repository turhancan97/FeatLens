"""CroCo and CroCo v2 (Weinzaepfel et al., 2022, 2023): feature maps from the ViT-B/16 encoder.

CroCo is pretrained by cross-view completion, so its ``forward`` takes an image *pair*. This loader
wraps the encoder alone (``CroCoNet._encode_image``) and reads its 12 ``enc_blocks`` at ``norm2``,
the same point FeatLens reads for timm ViTs. CroCo has no CLS token; inputs use ImageNet mean/std as
in CroCo's training.

    git clone https://github.com/naver/croco
    CROCO_REPO=./croco python examples/backbones/load_croco.py --version v2 --image my.jpg

CroCo v2 uses RoPE; without CroCo's optional compiled kernel (models/curope) it falls back to a
slower pure-PyTorch version, which is fine for visualization.
"""

from __future__ import annotations

import torch
from torch import nn

from common import cli, download, norm2_hook, require_repo

from featlens.adapters import external_adapter

CHECKPOINTS = {
    "v1": "https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo.pth",
    "v2": "https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTBase_BaseDecoder.pth",
}


class _Encoder(nn.Module):
    """Runs only CroCo's encoder on one image; ``enc_blocks`` is exposed for hooking."""

    def __init__(self, croco):
        super().__init__()
        self.croco = croco
        self.enc_blocks = croco.enc_blocks

    def forward(self, x):
        return self.croco._encode_image(x, do_mask=False)[0]


def load(img_size: int = 224, version: str = "v1", repo_dir: str | None = None):
    repo_dir = repo_dir or require_repo("CROCO_REPO", "https://github.com/naver/croco")
    url = CHECKPOINTS[version]

    def build():
        from models.croco import CroCoNet  # importable once repo_dir is on sys.path

        # Official checkpoint that stores its constructor kwargs next to the weights, hence
        # weights_only=False (only load checkpoints you trust this way).
        ckpt = torch.load(download(url, url.rsplit("/", 1)[-1]), map_location="cpu", weights_only=False)
        model = CroCoNet(img_size=img_size, **ckpt.get("croco_kwargs", {}))
        model.load_state_dict(ckpt["model"], strict=True)
        return _Encoder(model.eval())

    name = "croco_b16" if version == "v1" else "crocov2_b16"
    lm = external_adapter.load(repo_dir, build, hook_target=norm2_hook("enc_blocks"), num_blocks=12,
                               patch_size=16, name=name)
    assert len(lm.model.enc_blocks) == lm.num_blocks, "unexpected CroCo encoder depth"
    return lm


if __name__ == "__main__":
    cli(load, "Feature maps from the CroCo / CroCo v2 ViT-B/16 encoder.",
        extra_args=lambda p: p.add_argument("--version", choices=["v1", "v2"], default="v1"))
