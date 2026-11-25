"""FeatLens explainer palette (Manim).

Main colors are the user's "apple" palette (originally LaTeX \\definecolor RGB values).
Each color has a fixed *role* in the video so the same idea is always the same color.
"""

from manim import ManimColor

# --- the eight palette colors (RGB from the LaTeX definitions) ----------------
APPLE_BLUE = ManimColor.from_rgb((80, 122, 255))      # softer, muted blue
APPLE_PINK = ManimColor.from_rgb((217, 127, 174))     # muted, soft pink
APPLE_TEAL = ManimColor.from_rgb((85, 163, 152))      # soft, calming teal
APPLE_PURPLE = ManimColor.from_rgb((138, 107, 221))   # muted, soft purple
APPLE_MUSTARD = ManimColor.from_rgb((234, 179, 77))   # warm, soft mustard yellow
APPLE_ORANGE = ManimColor.from_rgb((255, 127, 80))    # vibrant, distinctive orange
APPLE_GREEN = ManimColor.from_rgb((52, 199, 89))
APPLE_YELLOW = ManimColor.from_rgb((255, 204, 0))

# --- neutrals (not part of the palette; used for background and body text) ---
BACKGROUND = ManimColor("#16181D")   # near-black, slightly cool, 3b1b-like
TEXT = ManimColor("#ECEDEF")         # off-white body text
TEXT_DIM = ManimColor("#8B8F98")     # secondary labels, axes, grid lines

# --- semantic roles ------------------------------------------------------------
TOKEN = APPLE_BLUE        # patch tokens / feature vectors flowing through the model
LAYER = APPLE_PURPLE      # transformer blocks, layer indices, the "hook" point
CODE = APPLE_TEAL         # code snippets, API names, pip install
EMPHASIS = APPLE_MUSTARD  # key words in on-screen text, titles
SEED = APPLE_YELLOW       # the seed patch (cosine similarity, correspondence)
OUTLIER = APPLE_PINK      # outlier patches that robust PCA ignores; CLS/register tokens

# PCA components -> color channels. The palette has no pure red, so orange stands in
# for the "red" channel; the final rendered feature maps still use true RGB.
PC1 = APPLE_ORANGE
PC2 = APPLE_GREEN
PC3 = APPLE_BLUE

# k-means clusters: cycle through the palette in a high-contrast order.
CLUSTERS = [APPLE_BLUE, APPLE_ORANGE, APPLE_GREEN, APPLE_PINK,
            APPLE_MUSTARD, APPLE_PURPLE, APPLE_TEAL, APPLE_YELLOW]

PALETTE = [APPLE_BLUE, APPLE_PINK, APPLE_TEAL, APPLE_PURPLE,
           APPLE_MUSTARD, APPLE_ORANGE, APPLE_GREEN, APPLE_YELLOW]
