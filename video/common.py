"""Shared helpers for the explainer scenes."""

import json
import sys
from pathlib import Path

import numpy as np
from manim import ImageMobject, ManimColor, Text, config

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from palette import BACKGROUND, TEXT  # noqa: E402

config.background_color = BACKGROUND
FONT = "DejaVu Sans"
MONO = "DejaVu Sans Mono"
ASSETS = HERE / "assets"
REPO = HERE.parent


def label(s, size=26, color=TEXT, font=FONT, **kw):
    return Text(s, font=font, font_size=size, color=color, **kw)


def to_color(rgb):
    return ManimColor.from_rgb(tuple(float(c) for c in np.clip(rgb, 0, 1)))


def image(path, height=None, width=None):
    m = ImageMobject(str(path))
    if height is not None:
        m.height = height
    if width is not None:
        m.width = width
    return m


def load_json(path):
    return json.loads(Path(path).read_text())


def point_on(img_mob, xy):
    """Scene point for a normalized (x, y) position (origin top-left) on an ImageMobject."""
    x, y = xy
    left, top = img_mob.get_left()[0], img_mob.get_top()[1]
    return np.array([left + x * img_mob.width, top - y * img_mob.height, 0])
