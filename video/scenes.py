"""Scenes 1, 2, 4, 5, 6 of the FeatLens explainer (scene 3 lives in scene3_pca.py).

    manim -qh video/scenes.py ColdOpen TokensScene MethodsScene BeyondScene CodeScene

All feature maps shown are real FeatLens outputs (prepare_data.py / prepare_scenes.py).
"""

import numpy as np
from manim import *  # noqa: F403

from common import (ASSETS, MONO, REPO, image, label, load_json, point_on,  # noqa: E402
                    to_color)
from palette import (BACKGROUND, CLUSTERS, CODE, EMPHASIS, LAYER, OUTLIER, SEED,  # noqa: E402
                     TEXT, TEXT_DIM, TOKEN)
from voiceover import NarratedScene  # noqa: E402

DATA = np.load(ASSETS / "cat_dino.npz")
CAT = DATA["image"]                       # [224, 224, 3] uint8, the model input
GRID_N = 14


def token_shades(n, seed=0):
    """Blue-ish token colors with a little variation (real PC1 values drive the shade)."""
    pc1 = DATA["coords"][:, 0]
    t = (pc1 - pc1.min()) / (pc1.max() - pc1.min())
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(t))[:n] if n < len(t) else np.arange(n) % len(t)
    return [interpolate_color(TOKEN.darker(0.45), TOKEN.lighter(0.35), float(t[i])) for i in idx]


def cat_grid(cell, center, shades=None):
    cells = VGroup()
    shades = shades or token_shades(GRID_N * GRID_N)
    for r in range(GRID_N):
        for c in range(GRID_N):
            sq = Square(cell, stroke_width=0.8, stroke_color=BACKGROUND,
                        fill_color=shades[r * GRID_N + c], fill_opacity=1)
            sq.move_to(center + np.array([(c - 6.5) * cell, (6.5 - r) * cell, 0]))
            cells.add(sq)
    return cells


# =====================================================================================
class ColdOpen(NarratedScene):
    def construct(self):
        cat = ImageMobject(CAT).set_height(4.6)
        grid_lines = VGroup(*[
            Line(cat.get_corner(UL) + RIGHT * cat.width * i / GRID_N,
                 cat.get_corner(DL) + RIGHT * cat.width * i / GRID_N)
            for i in range(1, GRID_N)
        ], *[
            Line(cat.get_corner(UL) + DOWN * cat.height * i / GRID_N,
                 cat.get_corner(UR) + DOWN * cat.height * i / GRID_N)
            for i in range(1, GRID_N)
        ]).set_stroke(BACKGROUND, 1.5)
        cells = cat_grid(cat.height / GRID_N, cat.get_center())

        with self.voiceover("When a vision model looks at this cat, it doesn't see a cat. "
                            "It sees a grid of numbers, rewritten layer after layer.") as vo:
            self.play(FadeIn(cat, scale=1.04), run_time=1.2)
            self.play(Create(grid_lines, lag_ratio=0.05), run_time=1.0)
            nums = VGroup(*[
                label(f"{v:+.1f}", 13, TEXT).move_to(cells[i])
                for i, v in zip(range(0, 196, 9), DATA["coords"][::9, 0] / 10)
            ])
            self.play(FadeIn(cells, lag_ratio=0.004), FadeOut(grid_lines),
                      FadeIn(nums, lag_ratio=0.02), run_time=1.8)
            self.remove(cat)
            base = VGroup(cells, nums)
            self.play(base.animate.scale(0.75).shift(LEFT * 1.6 + DOWN * 0.5), run_time=0.8)
            # Twelve layers start hidden exactly behind the front grid, already in depth order
            # (deepest drawn first), then slide back along one diagonal: the stack grows
            # backwards smoothly and nothing is re-ordered mid-shot.
            step = np.array([0.25, 0.17, 0])
            layers = []
            for k in range(12):
                lay = VGroup(cells.copy().set_fill(opacity=0.9 - 0.06 * k),
                             SurroundingRectangle(cells, buff=0, color=LAYER, stroke_width=2))
                layers.append(lay.move_to(base))
            for lay in layers:          # k = 0 ends up just behind the front, k = 11 deepest
                self.add(lay)
                self.bring_to_back(lay)
            self.play(LaggedStart(*[lay.animate.shift((k + 1) * step)
                                    for k, lay in enumerate(layers)], lag_ratio=0.08),
                      run_time=min(2.6, max(1.2, vo.duration - 5.0)), rate_func=smooth)

        logo = image(REPO / "assets" / "logo.png", height=2.4)
        name = label("FeatLens", 64, TEXT, weight=BOLD)
        tag = label("See what any vision model encodes.", 30, EMPHASIS)
        card = Group(logo, VGroup(name, tag).arrange(DOWN, buff=0.3)).arrange(RIGHT, buff=0.6)
        with self.voiceover("So how do we see what it has learned? That's what FeatLens is for.") as vo:
            self.play(FadeOut(Group(*self.content())), run_time=0.9)
            self.wait(max(0.0, vo.duration * 0.45 - 0.9))
            self.play(FadeIn(logo, scale=0.8), Write(name), run_time=1.2)
            self.play(FadeIn(tag, shift=UP * 0.2), run_time=0.8)
        self.wait(0.8)
        self.play(FadeOut(card), run_time=0.7)


# =====================================================================================
class TokensScene(NarratedScene):
    def construct(self):
        cat = ImageMobject(CAT).set_height(4.2).shift(LEFT * 3.2)
        cell = cat.height / GRID_N
        lines = VGroup(*[
            Line(cat.get_corner(UL) + RIGHT * cell * i, cat.get_corner(DL) + RIGHT * cell * i)
            for i in range(1, GRID_N)
        ], *[
            Line(cat.get_corner(UL) + DOWN * cell * i, cat.get_corner(UR) + DOWN * cell * i)
            for i in range(1, GRID_N)
        ]).set_stroke(TEXT, 1, opacity=0.7)

        # ---- 1: patches ----
        setup_note = label("typical ViT-B/16 · 224 × 224 image", 24, TEXT_DIM).next_to(cat, UP, 0.3)
        with self.voiceover("A vision transformer cuts the image into square patches. In a typical "
                            "setup, a two hundred twenty-four pixel image with sixteen-pixel patches "
                            "gives a fourteen-by-fourteen grid: one hundred ninety-six patches. "
                            "Other models use other sizes.") as vo:
            self.play(FadeIn(cat), FadeIn(setup_note), run_time=0.8)
            self.play(Create(lines, lag_ratio=0.04), run_time=1.4)
            r, c = 6, 4
            patch_px = CAT[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16]
            patch = ImageMobject(patch_px).set_height(cell)
            patch.move_to(cat.get_corner(UL) + np.array([(c + 0.5) * cell, -(r + 0.5) * cell, 0]))
            ring = SurroundingRectangle(patch, buff=0, color=SEED, stroke_width=3)
            self.play(Create(ring), run_time=0.5)
            big = patch.copy().set_height(2.0).move_to(RIGHT * 2.2 + UP * 0.6)
            big_ring = SurroundingRectangle(big, buff=0, color=SEED, stroke_width=3)
            px_label = label("16 × 16 pixels", 24, TEXT_DIM).next_to(big, DOWN, 0.25)
            self.play(FadeIn(big, target_position=patch), Create(big_ring), FadeIn(px_label),
                      run_time=1.0)
            count = label("14 × 14 = 196 patches", 28, EMPHASIS).next_to(cat, DOWN, 0.35)
            self.play(Write(count), run_time=1.0)
            vary = label("patch size and image size\nvary from model to model", 22, TEXT_DIM)
            vary.next_to(px_label, DOWN, 0.45)
            self.wait(max(0.0, vo.duration - 7.5))
            self.play(FadeIn(vary, shift=UP * 0.15), run_time=0.7)

        # ---- 2: tokens + blocks ----
        bar_h = 1.6
        with self.voiceover("Each patch becomes a token, a vector of numbers: seven hundred "
                            "sixty-eight in a base-sized model like this one, three hundred "
                            "eighty-four in a small model, one thousand twenty-four in a large one. "
                            "Then a stack of transformer blocks, twelve here, rewrites every token "
                            "in turn.") as vo:
            vo_start = self.renderer.time
            token = Rectangle(width=0.22, height=bar_h, fill_color=TOKEN, fill_opacity=1,
                              stroke_width=0).move_to(big).shift(LEFT * 1.2)
            dim = label("768 numbers", 22, TOKEN).next_to(token, UP, 0.15)
            self.play(FadeOut(big, scale=0.3), ReplacementTransform(big_ring, token),
                      FadeOut(px_label), FadeOut(vary), FadeIn(dim), run_time=1.0)
            # Token width by model size (standard ViT widths), base highlighted.
            sizes = [("ViT-Ti", 192), ("ViT-S", 384), ("ViT-B", 768), ("ViT-L", 1024)]
            floor = token.get_bottom()[1]
            chart = VGroup()
            for j, (name, d) in enumerate(sizes):
                bar = Rectangle(width=0.42, height=bar_h * d / 768, fill_color=TOKEN,
                                fill_opacity=1 if d == 768 else 0.4, stroke_width=0)
                bar.move_to([token.get_x() + 1.3 + 0.95 * j, floor, 0], aligned_edge=DOWN)
                txt = label(f"{name}\n{d}", 20, TEXT if d == 768 else TEXT_DIM)
                chart.add(VGroup(bar, txt.next_to(bar, DOWN, 0.15)))
            self.play(LaggedStart(*[GrowFromEdge(c[0], DOWN) for c in chart], lag_ratio=0.25),
                      LaggedStart(*[FadeIn(c[1]) for c in chart], lag_ratio=0.25), run_time=1.4)
            self.wait(max(0.0, vo.duration * 0.55 - 2.4))

            shades = token_shades(14, seed=3)
            bars = VGroup(*[Rectangle(width=0.22, height=bar_h, fill_color=s, fill_opacity=1,
                                      stroke_width=0) for s in shades]).arrange(RIGHT, buff=0.08)
            dots = label("…", 30, TEXT_DIM)
            cls = Rectangle(width=0.22, height=bar_h, fill_color=OUTLIER, fill_opacity=1,
                            stroke_width=0)
            row = VGroup(cls, bars[:7], dots, bars[7:]).arrange(RIGHT, buff=0.12)
            row.move_to(UP * 1.7)
            cls_label = label("CLS", 20, OUTLIER).next_to(cls, UP, 0.12)
            brace = Brace(VGroup(bars[:7], bars[7:]), DOWN, color=TEXT_DIM)
            brace_label = label("196 patch tokens", 22, TEXT_DIM).next_to(brace, DOWN, 0.1)
            self.play(FadeOut(Group(cat, lines, ring, count, dim, chart, setup_note)),
                      ReplacementTransform(token, bars[3]), run_time=0.8)
            self.play(FadeIn(row), FadeIn(brace), FadeIn(brace_label), run_time=1.0)
            self.play(FadeIn(cls_label), run_time=0.4)

            blocks = VGroup(*[
                RoundedRectangle(corner_radius=0.08, width=0.62, height=1.3, color=LAYER,
                                 fill_opacity=0.18, stroke_width=2)
                for _ in range(12)
            ]).arrange(RIGHT, buff=0.2).move_to(DOWN * 1.6)
            nums = VGroup(*[label(str(i + 1), 18, LAYER).next_to(b, DOWN, 0.12)
                            for i, b in enumerate(blocks)])
            self.play(FadeOut(brace), FadeOut(brace_label),
                      LaggedStart(*[FadeIn(b) for b in blocks], lag_ratio=0.06),
                      FadeIn(nums), run_time=1.0)
            packet = row.copy().scale(0.28).move_to(blocks[0])
            self.play(TransformFromCopy(row, packet), run_time=0.7)
            sweep = max(1.2, vo.duration - self.renderer.time + vo_start - 0.3)
            for k in range(1, 12):
                new = [interpolate_color(s, TOKEN.lighter(0.3) if k % 2 else TOKEN.darker(0.3), 0.35)
                       for s in shades]
                self.play(packet.animate.move_to(blocks[k]),
                          *[b.animate.set_fill(n) for b, n in zip(packet[1], new[:7])],
                          *[b.animate.set_fill(n) for b, n in zip(packet[3], new[7:])],
                          blocks[k].animate.set_fill(opacity=0.45), run_time=sweep / 11,
                          rate_func=linear)
                blocks[k].set_fill(opacity=0.18)

        # ---- 3: the hook ----
        with self.voiceover("FeatLens can hook any of those layers and copy what it holds.") as vo:
            k = 7
            hook = Arrow(blocks[k].get_top() + UP * 1.05, blocks[k].get_top(), buff=0.05,
                         color=LAYER, stroke_width=6)
            hook_label = label("hook · layer 8", 24, LAYER).next_to(hook, UP, 0.1)
            self.play(packet.animate.move_to(blocks[k]), FadeOut(row), FadeOut(cls_label),
                      run_time=0.6)
            self.play(GrowArrow(hook), FadeIn(hook_label), blocks[k].animate.set_fill(opacity=0.5),
                      run_time=0.8)
            copy = packet.copy()
            self.add(copy)
            self.play(copy.animate.scale(1 / 0.28).move_to(UP * 1.9), run_time=1.0)

        # ---- 4: drop extras, fold to grid ----
        with self.voiceover("It drops the extra tokens that don't belong to any location, like the "
                            "class token, and folds the rest back into the fourteen-by-fourteen "
                            "grid.") as vo:
            cls_copy = copy[0]
            cross = Cross(cls_copy, stroke_color=OUTLIER, stroke_width=5).scale(1.3)
            self.play(Create(cross), run_time=0.6)
            self.play(FadeOut(VGroup(cls_copy, cross), shift=UP * 0.4),
                      FadeOut(VGroup(blocks, nums, hook, hook_label, packet)), run_time=0.8)
            grid = cat_grid(0.28, DOWN * 0.3)
            self.play(FadeTransform(VGroup(copy[1], copy[2], copy[3]), grid),
                      run_time=1.6)
            depth = VGroup(*[grid.copy().set_fill(opacity=0.35 - 0.06 * i).set_stroke(opacity=0.2)
                             .shift((i + 1) * np.array([0.09, 0.07, 0])) for i in range(5)])
            self.bring_to_back(depth)
            self.play(FadeIn(depth, lag_ratio=0.2), run_time=0.8)
            dims = label("[768, 14, 14]", 30, EMPHASIS, font=MONO).next_to(grid, RIGHT, 0.7)
            sub = label("one 768-number vector\nper patch, in place", 22, TEXT_DIM)
            sub.next_to(dims, DOWN, 0.25, aligned_edge=LEFT)
            self.play(Write(dims), FadeIn(sub), run_time=1.0)

        # ---- 5: any source ----
        with self.voiceover("And it works the same whether the model comes from timm, Hugging "
                            "Face, torch hub, or your own code.") as vo:
            fmap = VGroup(grid, depth)
            box_label = label("feature map  [D, h, w]", 26, CODE, font=MONO).move_to(DOWN * 1.4)
            box = SurroundingRectangle(box_label, buff=0.35, corner_radius=0.15, color=CODE,
                                       fill_opacity=0.12)
            names = ["timm", "Hugging Face", "torch.hub", "your own model"]
            badges = VGroup(*[
                VGroup(RoundedRectangle(corner_radius=0.12, width=2.7, height=0.7, color=TEXT_DIM,
                                        stroke_width=2), label(n, 22, TEXT))
                for n in names
            ]).arrange(RIGHT, buff=0.35).move_to(UP * 1.6)
            targets = [box.get_top() + RIGHT * box.width * f for f in (-0.33, -0.11, 0.11, 0.33)]
            arrows = VGroup(*[Arrow(b.get_bottom(), t, buff=0.1, color=TEXT_DIM, stroke_width=3)
                              for b, t in zip(badges, targets)])
            self.play(FadeOut(fmap), FadeOut(dims), FadeOut(sub), run_time=0.6)
            self.play(LaggedStart(*[FadeIn(b, shift=DOWN * 0.2) for b in badges], lag_ratio=0.3),
                      run_time=1.6)
            self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.15),
                      FadeIn(box), Write(box_label), run_time=1.4)
        self.wait(0.4)
        self.play(FadeOut(Group(*self.content())), run_time=0.7)


# =====================================================================================
class MethodsScene(NarratedScene):
    def construct(self):
        seed_info = load_json(ASSETS / "methods" / "seed.json")
        photo = ImageMobject(CAT).set_height(4.4).shift(LEFT * 3.0 + UP * 0.2)
        cosine = image(ASSETS / "methods" / "cosine.png", height=4.4).shift(RIGHT * 3.0 + UP * 0.2)
        seed_pt = point_on(photo, seed_info["seed"])
        seed_dot = Dot(seed_pt, radius=0.11, color=SEED)
        seed_ring = Circle(0.2, color=SEED, stroke_width=4).move_to(seed_pt)
        cap_photo = label("seed patch", 24, SEED).next_to(photo, DOWN, 0.3)
        cap_cos = label("cosine similarity to the seed", 24, TEXT_DIM).next_to(cosine, DOWN, 0.3)

        with self.voiceover("FeatLens has other views too. Cosine similarity lights up every patch "
                            "that looks like a seed you pick.") as vo:
            self.play(FadeIn(photo), run_time=0.8)
            self.play(GrowFromCenter(seed_dot), Create(seed_ring), FadeIn(cap_photo), run_time=0.8)
            beam = Arrow(seed_pt, cosine.get_left() + UP * 0.6, color=SEED, stroke_width=4, buff=0.2)
            self.play(GrowArrow(beam), run_time=0.7)
            self.play(FadeIn(cosine, scale=0.92), FadeIn(cap_cos), run_time=1.2)

        names = [("cosine", "cosine"), ("kmeans", "k-means"), ("foreground", "foreground"),
                 ("saliency", "saliency")]
        tiles = Group(*[image(ASSETS / "methods" / f"{n}.png", height=2.9) for n, _ in names])
        tiles.arrange(RIGHT, buff=0.35).shift(UP * 0.35)
        caps = VGroup(*[label(t, 26, TEXT).next_to(m, DOWN, 0.3) for m, (_, t) in zip(tiles, names)])
        colors = [SEED, CLUSTERS[1], TEXT, EMPHASIS]
        for cap, col in zip(caps, colors):
            cap.set_color(col)
        model = label("DINOv2 ViT-B/14 · last layer", 22, TEXT_DIM).move_to(DOWN * 2.3)

        with self.voiceover("K-means groups patches into segments, a foreground mask splits object "
                            "from background, and saliency shows where the activations are "
                            "strongest.") as vo:
            self.play(FadeOut(Group(photo, seed_dot, seed_ring, cap_photo, beam, cap_cos)),
                      Transform(cosine, tiles[0]), run_time=1.0)
            self.play(FadeIn(caps[0]), FadeIn(model), run_time=0.4)
            step = max(0.8, (vo.duration - 1.6) / 3)
            for i in (1, 2, 3):
                self.play(FadeIn(tiles[i], shift=UP * 0.2), FadeIn(caps[i]), run_time=step * 0.6)
                self.wait(step * 0.4)
        self.wait(0.6)
        self.play(FadeOut(Group(*self.content())), run_time=0.7)


# =====================================================================================
class BeyondScene(NarratedScene):
    def construct(self):
        m = load_json(ASSETS / "correspond" / "match.json")
        photo = image(ASSETS / "correspond" / "photo.png", height=4.3).shift(LEFT * 3.1 + DOWN * 0.1)
        water = image(ASSETS / "correspond" / "watercolor.png", height=4.3).shift(RIGHT * 3.1 + DOWN * 0.1)
        a, b = point_on(photo, m["seed"]), point_on(water, m["match"])
        cap_a = label("photo", 24, TEXT_DIM).next_to(photo, DOWN, 0.3)
        cap_b = label("watercolor", 24, TEXT_DIM).next_to(water, DOWN, 0.3)

        with self.voiceover("The same features match across images. Seed a cat's eye in a photo, "
                            "and the best match in a watercolor lands on the painted eye.") as vo:
            self.play(FadeIn(photo), FadeIn(water), FadeIn(cap_a), FadeIn(cap_b), run_time=1.0)
            self.wait(max(0.0, vo.duration * 0.3 - 1.0))
            ring_a = Circle(0.2, color=SEED, stroke_width=5).move_to(a)
            self.play(Create(ring_a), Flash(a, color=SEED, line_length=0.15), run_time=0.8)
            arc = CurvedArrow(a + UP * 0.2, b + UP * 0.2, angle=-PI / 3, color=SEED, stroke_width=4)
            self.play(Create(arc), run_time=1.2)
            ring_b = Circle(0.2, color=SEED, stroke_width=5).move_to(b)
            sim = label(f"cosine similarity {m['similarity']:.2f}", 24, SEED)
            sim.move_to([0, max(arc.get_top()[1], photo.get_top()[1]) + 0.35, 0])
            self.play(Create(ring_b), FadeIn(sim), run_time=0.7)

        models = load_json(ASSETS / "compare" / "models.json")
        tiles = Group(*[image(ASSETS / "compare" / f"{spec}.png", height=2.3) for spec, _ in models])
        tiles.arrange_in_grid(rows=2, cols=3, buff=(0.3, 0.75)).shift(RIGHT * 1.6 + UP * 0.3)
        caps = VGroup(*[label(name, 21, TEXT).next_to(t, DOWN, 0.12)
                        for t, (_, name) in zip(tiles, models)])
        market = image(REPO / "examples" / "images" / "market.jpg", width=3.1).to_edge(LEFT, buff=0.5)
        market_cap = label("same image,\nsix ViT-B/16 models", 22, TEXT_DIM).next_to(market, DOWN, 0.3)

        with self.voiceover("Six models with the same architecture, trained differently, carve up "
                            "the same scene in very different ways.") as vo:
            self.play(FadeOut(Group(photo, water, cap_a, cap_b, ring_a, ring_b, arc, sim)),
                      run_time=0.7)
            self.play(FadeIn(market), FadeIn(market_cap), run_time=0.7)
            self.play(LaggedStart(*[FadeIn(Group(t, c), shift=UP * 0.2) for t, c in zip(tiles, caps)],
                                  lag_ratio=0.35), run_time=max(2.0, vo.duration - 2.0))

        frames = sorted((ASSETS / "vjepa").glob("frame_*.png"), key=lambda p: int(p.stem.split("_")[1]))
        clip = [image(p, width=10) .shift(UP * 0.3) for p in frames]
        cap_in = label("input clip", 24, TEXT_DIM).next_to(clip[0], DOWN, 0.3).shift(LEFT * 2.55)
        cap_out = label("V-JEPA 2.1 features", 24, TEXT_DIM).next_to(clip[0], DOWN, 0.3).shift(RIGHT * 2.55)

        with self.voiceover("And with a video model like V-JEPA, one shared color basis lets you "
                            "watch the bird move.") as vo:
            self.play(FadeOut(Group(tiles, caps, market, market_cap)), run_time=0.6)
            self.play(FadeIn(clip[0]), FadeIn(cap_in), FadeIn(cap_out), run_time=0.6)
            shown, t, i = clip[0], 0.0, 1
            while t < vo.duration + 0.6:
                nxt = clip[i % len(clip)]
                self.remove(shown)
                self.add(nxt)
                shown = nxt
                self.wait(0.25)
                t += 0.25
                i += 1
        self.play(FadeOut(Group(*self.content())), run_time=0.7)


# =====================================================================================
class CodeScene(NarratedScene):
    def construct(self):
        code_lines = [
            "import featlens as ll",
            "",
            'll.visualize("dinov2_vitb14", "cat.jpg",',
            "             layers=[2, 5, 8, 11], img_size=448)",
        ]
        code = Text("\n".join(code_lines), font=MONO, font_size=28, color=TEXT, line_spacing=1.1,
                    t2c={"ll.": CODE, "featlens": CODE, "visualize": CODE,
                         '"dinov2_vitb14"': EMPHASIS, '"cat.jpg"': EMPHASIS,
                         "[2, 5, 8, 11]": LAYER, "448": LAYER})
        code.to_edge(UP, buff=0.9)
        panel = SurroundingRectangle(code, buff=0.35, corner_radius=0.15, color=TEXT_DIM,
                                     stroke_width=1.5, fill_color="#1E2128", fill_opacity=1)
        tiles = Group(*[image(ASSETS / "visualize" / f"layer_{l}.png", height=2.5)
                        for l in (2, 5, 8, 11)]).arrange(RIGHT, buff=0.3).next_to(panel, DOWN, 0.6)
        caps = VGroup(*[label(f"layer {l}", 22, LAYER).next_to(t, DOWN, 0.15)
                        for t, l in zip(tiles, (2, 5, 8, 11))])

        with self.voiceover("All of it is one line of Python. Pick a model, an image, and the "
                            "layers.") as vo:
            self.play(FadeIn(panel), run_time=0.5)
            self.play(AddTextLetterByLetter(code, time_per_char=0.03))
            self.play(LaggedStart(*[FadeIn(Group(t, c), shift=UP * 0.2) for t, c in zip(tiles, caps)],
                                  lag_ratio=0.25), run_time=1.6)

        logo = image(REPO / "assets" / "logo.png", height=1.6)
        name = label("FeatLens", 54, TEXT, weight=BOLD)
        head = Group(logo, name).arrange(RIGHT, buff=0.4).to_edge(UP, buff=0.8)
        links = VGroup(
            label("pip install featlens", 30, CODE, font=MONO),
            label("github.com/turhancan97/FeatLens", 26, TEXT, font=MONO),
            label("demo · huggingface.co/spaces/turhancan97/FeatLens-demo", 24, TEXT, font=MONO),
        ).arrange(DOWN, buff=0.35).next_to(head, DOWN, 0.7)
        closer = label("Look at what it sees.", 40, EMPHASIS, weight=BOLD).next_to(links, DOWN, 0.75)
        doi = label("DOI 10.5281/zenodo.21082488", 20, TEXT_DIM).next_to(closer, DOWN, 0.4)

        with self.voiceover("FeatLens is open source and on pip, and you can try it right now in the "
                            "browser demo. So don't just check what your model scores. Look at what "
                            "it sees.") as vo:
            self.play(FadeOut(Group(panel, code, tiles, caps)), run_time=0.6)
            self.play(FadeIn(logo, scale=0.8), Write(name), run_time=1.0)
            self.play(LaggedStart(*[FadeIn(l, shift=UP * 0.15) for l in links], lag_ratio=0.5),
                      run_time=2.2)
            self.wait(max(0.0, vo.duration - 5.6))
            self.play(Write(closer), FadeIn(doi), run_time=1.2)
        self.wait(2.0)
        self.play(FadeOut(Group(*self.content())), run_time=1.0)
