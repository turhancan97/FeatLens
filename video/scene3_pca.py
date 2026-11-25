"""Scene 3 — the core trick: 768 dimensions -> 3 colors (robust PCA).

Render (from the repo root, in an env with manim + kokoro):
    manim -ql video/scene3_pca.py PCAScene     # quick preview
    manim -qh video/scene3_pca.py PCAScene     # 1080p

Everything shown is real: the cloud is the 196 patches of DINO ViT-B/16 (last layer) on
cat_hires.jpg, plotted at their top-3 PCA coordinates; both color grids are the real
min/max and robust FeatLens maps (see prepare_data.py).
"""

import sys
from pathlib import Path

import numpy as np
from manim import *  # noqa: F403

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from palette import (BACKGROUND, EMPHASIS, OUTLIER, PC1, PC2, PC3, SEED, TEXT,  # noqa: E402
                     TEXT_DIM, TOKEN)
from voiceover import NarratedScene  # noqa: E402

config.background_color = BACKGROUND
FONT = "DejaVu Sans"

DATA = np.load(HERE / "assets" / "cat_dino.npz")
GRID_N = 14
CLOUD_C = np.array([-3.5, 0.55, 0])
GRID_C = np.array([3.45, 0.55, 0])
CELL = 0.28
SEED_IDX = 6 * GRID_N + 4  # a patch on the cat's face (row 6, col 4)

NARRATION = [
    "Now the problem: we can't see seven hundred sixty-eight dimensions. "
    "We can see exactly three. Red, green and blue.",
    "So think of every patch as a point in a seven-hundred-sixty-eight-dimensional space, "
    "where similar patches land close together.",
    "Principal component analysis finds the three directions along which these points "
    "spread out the most.",
    "Project each patch onto them, call the three numbers red, green and blue, "
    "and paint the patch.",
    "Patches the model sees as similar get similar colors, and the parts of the scene separate.",
    "One catch: the far tails of the distribution stretch the color range, "
    "and every color fades.",
    "So FeatLens sets the range from the bulk of the patches, measured around the median, "
    "and clips the tails at the edges.",
]


def label(s, size=26, color=TEXT, **kw):
    return Text(s, font=FONT, font_size=size, color=color, **kw)


def to_color(rgb):
    return ManimColor.from_rgb(tuple(float(c) for c in np.clip(rgb, 0, 1)))


class PCAScene(NarratedScene):
    def construct(self):
        coords = DATA["coords"]
        centered = coords - coords.mean(0)
        pts = centered / np.abs(centered).max() * 2.0          # display units
        robust = DATA["rgb_robust"].reshape(-1, 3)
        naive = DATA["rgb_naive"].reshape(-1, 3)
        lo, hi = DATA["rgb_min"][0], DATA["rgb_max"][0]
        pc1 = coords[:, 0]
        tail = (pc1 < lo) | (pc1 > hi)                          # outside the robust PC1 range

        # ---- trackers driving the point cloud ---------------------------------------------
        angle = ValueTracker(0.3)
        angle.add_updater(lambda m, dt: m.increment_value(0.22 * dt))
        mix = ValueTracker(0)       # 0 = all blue tokens, 1 = PCA colors
        use_naive = ValueTracker(0)  # 0 = robust colors, 1 = plain min/max colors
        pinkness = ValueTracker(0)   # 1 = tail patches shown pink
        axis_len = ValueTracker(0.001)
        self.add(angle)

        def project(v, tilt=0.38):
            th = angle.get_value()
            x, y, z = v
            x1 = x * np.cos(th) + z * np.sin(th)
            z1 = -x * np.sin(th) + z * np.cos(th)
            y1 = y * np.cos(tilt) - z1 * np.sin(tilt)
            z2 = y * np.sin(tilt) + z1 * np.cos(tilt)
            return CLOUD_C + np.array([x1, y1, 0]), z2

        dots = VGroup(*[Dot(radius=0.045) for _ in pts])

        def update_dots(group):
            m, nv, pk = mix.get_value(), use_naive.get_value(), pinkness.get_value()
            for i, d in enumerate(group):
                pos, depth = project(pts[i])
                d.move_to(pos)
                rgb = (1 - nv) * robust[i] + nv * naive[i]
                c = interpolate_color(TOKEN, to_color(rgb), m)
                if tail[i]:
                    c = interpolate_color(c, OUTLIER, pk)
                d.set_fill(c, opacity=float(np.clip(0.55 + 0.2 * depth, 0.3, 1)))

        dots.add_updater(update_dots)

        axis_colors, axis_names = [PC1, PC2, PC3], ["PC1", "PC2", "PC3"]

        def make_axes():
            g = VGroup()
            for k in range(3):
                e = np.zeros(3)
                e[k] = 2.3 * axis_len.get_value()
                tip, _ = project(e)
                g.add(Arrow(CLOUD_C, tip, buff=0, color=axis_colors[k], stroke_width=5,
                            max_tip_length_to_length_ratio=0.12))
                if axis_len.get_value() > 0.6:
                    g.add(label(axis_names[k], 22, axis_colors[k]).move_to(
                        tip + 0.32 * normalize(tip - CLOUD_C + 1e-6)))
            return g

        axes = always_redraw(make_axes)

        # ---- the image / feature grid -------------------------------------------------------
        cat = ImageMobject(str(HERE / "assets" / "cat_224.png"))
        cat.height = CELL * GRID_N
        cat.move_to(GRID_C)
        cells = VGroup()
        for r in range(GRID_N):
            for c in range(GRID_N):
                sq = Square(CELL, stroke_width=0.6, stroke_color=BACKGROUND, fill_opacity=0)
                sq.move_to(GRID_C + np.array([(c - 6.5) * CELL, (6.5 - r) * CELL, 0]))
                cells.add(sq)
        grid_caption = label("the 14 × 14 patch grid", 22, TEXT_DIM).next_to(cat, DOWN, 0.25)

        # ---- line 1: 768 dimensions vs 3 ----------------------------------------------------
        with self.voiceover(NARRATION[0]) as vo:
            big = label("768", 120, EMPHASIS)
            sub = label("dimensions per patch (ViT-B)", 30, TEXT_DIM).next_to(big, DOWN)
            self.play(FadeIn(big, scale=0.8), FadeIn(sub), run_time=1.2)
            self.wait(vo.duration * 0.35)
            rgb = VGroup(*[
                VGroup(Circle(0.55, color=c, fill_opacity=0.9, stroke_width=0),
                       label(n, 30, BACKGROUND, weight=BOLD))
                for c, n in zip(axis_colors, "RGB")
            ]).arrange(RIGHT, buff=0.5)
            three = label("we can see 3", 30, TEXT_DIM).next_to(rgb, DOWN)
            self.play(FadeOut(big, shift=UP), FadeOut(sub, shift=UP),
                      LaggedStart(*[GrowFromCenter(x) for x in rgb], lag_ratio=0.25),
                      FadeIn(three), run_time=1.6)

        # ---- line 2: patches as points ------------------------------------------------------
        with self.voiceover(NARRATION[1]) as vo:
            self.play(FadeOut(rgb), FadeOut(three), run_time=0.6)
            self.play(FadeIn(cat), FadeIn(grid_caption), run_time=0.8)
            cloud_caption = label("each dot = one patch", 22, TEXT_DIM)
            cloud_caption.move_to(CLOUD_C + DOWN * 2.75)
            self.add(dots)
            dots.set_opacity(0)
            self.play(LaggedStart(*[FadeIn(d, scale=0.3) for d in dots], lag_ratio=0.01),
                      FadeIn(cloud_caption), run_time=min(3.0, vo.duration * 0.5))
            dots.set_opacity(1)

        # ---- line 3: principal directions ---------------------------------------------------
        with self.voiceover(NARRATION[2]) as vo:
            self.add(axes)
            self.play(axis_len.animate.set_value(1), run_time=min(2.5, vo.duration * 0.6),
                      rate_func=smooth)
            spread = label("directions of greatest spread", 22, TEXT_DIM).move_to(cloud_caption)
            self.play(ReplacementTransform(cloud_caption, spread), run_time=0.8)

        # ---- line 4: project one patch, paint it ---------------------------------------------
        with self.voiceover(NARRATION[3]) as vo:
            ring = always_redraw(lambda: Circle(0.13, color=SEED, stroke_width=4).move_to(
                dots[SEED_IDX].get_center()))
            src_cell = cells[SEED_IDX]
            cell_ring = Square(CELL * 1.25, color=SEED, stroke_width=4).move_to(src_cell)
            self.play(Create(ring), Create(cell_ring), run_time=0.8)

            vals = robust[SEED_IDX]
            base = np.array([-0.05, -1.55, 0])
            bars = VGroup(*[
                Rectangle(width=0.32, height=max(1.6 * v, 0.04), fill_color=axis_colors[k],
                          fill_opacity=0.95, stroke_width=0)
                .move_to(base + np.array([(k - 1) * 0.45, 0, 0]), aligned_edge=DOWN)
                for k, v in enumerate(vals)
            ])
            bar_labels = VGroup(*[label(n, 20, axis_colors[k]).next_to(bars[k], DOWN, 0.12)
                                  .set_y(base[1] - 0.22) for k, n in enumerate("RGB")])
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.3),
                      FadeIn(bar_labels), run_time=1.4)
            swatch = Square(0.7, fill_color=to_color(vals), fill_opacity=1, stroke_width=0)
            swatch.move_to(base + UP * 0.8)
            self.play(ReplacementTransform(bars, swatch), FadeOut(bar_labels), run_time=1.0)
            self.play(swatch.animate.scale(CELL / 0.7).move_to(src_cell), run_time=0.9)
            src_cell.set_fill(to_color(vals), opacity=1)
            self.add(cells)
            self.remove(swatch)
            self.play(FadeOut(ring), FadeOut(cell_ring), run_time=0.5)

        # ---- line 5: paint every patch -------------------------------------------------------
        with self.voiceover(NARRATION[4]) as vo:
            order = [i for i in range(GRID_N * GRID_N) if i != SEED_IDX]
            fill_anims = [cells[i].animate.set_fill(to_color(robust[i]), opacity=1) for i in order]
            self.play(mix.animate.set_value(1),
                      LaggedStart(*fill_anims, lag_ratio=0.012),
                      run_time=min(4.0, vo.duration * 0.8))
            pca_caption = label("FeatLens PCA → RGB", 22, TEXT_DIM).move_to(grid_caption)
            self.play(ReplacementTransform(grid_caption, pca_caption), run_time=0.6)

        # ---- line 6: the tails stretch the range ---------------------------------------------
        line_y, x0, x1 = -2.6, -5.7, -1.0
        pmin, pmax = pc1.min(), pc1.max()

        def xpos(v):
            return x0 + (v - pmin) / (pmax - pmin) * (x1 - x0)

        number_line = Line([x0 - 0.2, line_y, 0], [x1 + 0.2, line_y, 0], color=TEXT_DIM,
                           stroke_width=2)
        nl_label = label("PC1", 22, PC1).next_to(number_line, LEFT, 0.2)
        ticks = VGroup(*[
            Line([xpos(v), line_y - 0.13, 0], [xpos(v), line_y + 0.13, 0],
                 stroke_width=2, color=OUTLIER if tail[i] else PC1, stroke_opacity=0.8)
            for i, v in enumerate(pc1)
        ])

        def bracket(x, side):
            b = label("[" if side < 0 else "]", 44, TEXT)
            return b.move_to([x, line_y, 0])

        left_b, right_b = bracket(xpos(pmin) - 0.08, -1), bracket(xpos(pmax) + 0.08, 1)
        range_text = label("range = min … max", 22, TEXT_DIM).move_to([(x0 + x1) / 2, line_y - 0.45, 0])

        with self.voiceover(NARRATION[5]) as vo:
            self.play(FadeOut(spread), Create(number_line), FadeIn(nl_label),
                      LaggedStart(*[Create(t) for t in ticks], lag_ratio=0.004),
                      pinkness.animate.set_value(1), run_time=1.6)
            self.play(FadeIn(left_b), FadeIn(right_b), FadeIn(range_text), run_time=0.7)
            pale_caption = label("plain min/max range", 22, TEXT_DIM).move_to(pca_caption)
            self.play(use_naive.animate.set_value(1),
                      *[cells[i].animate.set_fill(to_color(naive[i])) for i in range(len(cells))],
                      ReplacementTransform(pca_caption, pale_caption),
                      run_time=min(2.0, vo.duration * 0.4))

        # ---- line 7: robust range, clip the tails ---------------------------------------------
        with self.voiceover(NARRATION[6]) as vo:
            robust_text = label("range = median ± 2 · MAD", 22, TEXT_DIM).move_to(range_text)
            clamped = [t.animate.set_x(float(np.clip(xpos(v), xpos(lo), xpos(hi))))
                       for t, v in zip(ticks, pc1)]
            self.play(left_b.animate.set_x(xpos(lo) - 0.08), right_b.animate.set_x(xpos(hi) + 0.08),
                      ReplacementTransform(range_text, robust_text), run_time=1.4)
            self.play(*clamped, run_time=1.0)
            vivid_caption = label("robust range, tails clipped", 22, TEXT_DIM).move_to(pale_caption)
            self.play(use_naive.animate.set_value(0), pinkness.animate.set_value(0),
                      *[cells[i].animate.set_fill(to_color(robust[i])) for i in range(len(cells))],
                      ReplacementTransform(pale_caption, vivid_caption),
                      run_time=min(2.0, vo.duration * 0.35))

        # ---- outro: real FeatLens output across layers --------------------------------------------
        dots.clear_updaters()
        everything = [m for m in self.mobjects if m is not angle]
        row = ImageMobject(str(HERE.parent / "examples" / "feat_cat_hires.png"))
        row.width = 13
        row_caption = label("FeatLens · DINO ViT-S/8 · layers 2, 5, 8, 11", 24, TEXT_DIM)
        row_caption.next_to(row, DOWN, 0.35)
        self.play(*[FadeOut(m) for m in everything], run_time=0.8)
        self.play(FadeIn(row, scale=0.95), FadeIn(row_caption), run_time=1.0)
        self.wait(2.5)
        self.play(FadeOut(row), FadeOut(row_caption), run_time=0.8)
