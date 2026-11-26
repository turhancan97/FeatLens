"""Build assets/social_preview.png (1280x640, white) for GitHub's social preview.

Run from the repository root:  python assets/make_social_preview.py
Uses the logo unchanged (scaled only), the project palette, and real FeatLens tiles from
video/assets/ (DINOv2 ViT-B/14 on cat_hires.jpg). Upload the result in the repository
settings: General -> Social preview.
"""
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640
BLUE, PINK, TEAL, PURPLE = (80, 122, 255), (217, 127, 174), (85, 163, 152), (138, 107, 221)
MUSTARD, ORANGE, GREEN, YELLOW = (234, 179, 77), (255, 127, 80), (52, 199, 89), (255, 204, 0)
PALETTE = [BLUE, PINK, TEAL, PURPLE, MUSTARD, ORANGE, GREEN, YELLOW]
INK, MUTED = (29, 29, 31), (110, 112, 120)
F = "/usr/share/fonts/truetype/dejavu/"
bold = lambda s: ImageFont.truetype(F + "DejaVuSans-Bold.ttf", s)
reg = lambda s: ImageFont.truetype(F + "DejaVuSans.ttf", s)
mono = lambda s: ImageFont.truetype(F + "DejaVuSansMono.ttf", s)

img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

# --- palette band at the bottom ---------------------------------------------------------
band = 14
for i, c in enumerate(PALETTE):
    d.rectangle([i * W // 8, H - band, (i + 1) * W // 8, H], fill=c)

# --- logo, unchanged (only scaled, aspect kept) ------------------------------------------
logo = Image.open("assets/logo.png").convert("RGBA")
logo = logo.resize((150, 150), Image.LANCZOS)
img.paste(logo, (64, 70), logo)

# --- title + tagline ---------------------------------------------------------------------
d.text((232, 92), "FeatLens", font=bold(84), fill=INK)
d.text((236, 190), "See what any vision model encodes.", font=reg(27), fill=MUTED)
assert d.textlength("See what any vision model encodes.", font=reg(27)) + 236 < 750

# --- three "any" lines with palette dots -------------------------------------------------
lines = [(BLUE, "any model", "DINO · CLIP · SigLIP · MAE · V-JEPA · CNNs"),
         (ORANGE, "any source", "timm · Hugging Face · torch.hub · your own"),
         (GREEN, "any layer", "early blocks to the last, side by side"),
         (PURPLE, "five views", "PCA · cosine · k-means · foreground · saliency")]
y = 268
for color, head, rest in lines:
    d.ellipse([70, y + 9, 88, y + 27], fill=color)
    d.text((104, y), head, font=bold(26), fill=INK)
    d.text((104, y + 35), rest, font=reg(20), fill=MUTED)
    y += 72

# --- pip install pill ----------------------------------------------------------------------
pill = "pip install featlens"
tw = d.textlength(pill, font=mono(24))
d.rounded_rectangle([66, 560, 66 + tw + 40, 606], radius=23, fill=(236, 246, 244), outline=TEAL, width=2)
d.text((86, 569), pill, font=mono(24), fill=TEAL)

# --- real FeatLens output: cat + DINOv2 layers 5 / 8 / 11 -------------------------------------
tiles = [("video/assets/correspond/photo.png", "input", MUTED),
         ("video/assets/visualize/layer_5.png", "layer 5", PURPLE),
         ("video/assets/visualize/layer_8.png", "layer 8", PURPLE),
         ("video/assets/visualize/layer_11.png", "layer 11", PURPLE)]
S, GAP, Y0 = 212, 22, 58
X0 = W - 64 - (2 * S + GAP)
for k, (path, lab, col) in enumerate(tiles):
    r, c = divmod(k, 2)
    x, yy = X0 + c * (S + GAP), Y0 + r * (S + GAP + 34)
    t = Image.open(path).convert("RGB").resize((S, S), Image.LANCZOS if k == 0 else Image.NEAREST)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=18, fill=255)
    img.paste(t, (x, yy), mask)
    d.rounded_rectangle([x, yy, x + S - 1, yy + S - 1], radius=18, outline=(225, 226, 230), width=2)
    d.text((x + S / 2, yy + S + 17), lab, font=bold(20) if k else reg(20), fill=col, anchor="mm")
d.text((X0 + S + GAP / 2, 604), "DINOv2 ViT-B/14 · robust PCA → RGB", font=reg(18), fill=MUTED, anchor="mm")

img.save("assets/social_preview.png", optimize=True)
print("saved", img.size)
