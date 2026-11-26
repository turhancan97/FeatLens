"""Add the input image to the paper's two gallery figures (run from the repository root).

    python paper/make_figures.py

Reads the gallery figures produced by examples/quickstart.py and places the input image, resized
to a square exactly as FeatLens feeds it to the model (``resize_mode="squash"``), to the left of
the feature maps, in the same tile size, frame and title style:

  paper/feat_peacock.png        <- examples/feat_peacock.png        + examples/images/peacock.jpg
  paper/compare_b16_market.png  <- examples/compare_b16_market.png  + examples/images/market.jpg
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font_for_height(px, sample="layer 2"):
    """DejaVu Sans whose rendering of ``sample`` (an existing title) is ``px`` pixels tall."""
    size = 8
    while True:
        box = ImageFont.truetype(FONT, size + 1).getbbox(sample)
        if box[3] - box[1] > px:
            return ImageFont.truetype(FONT, size)
        size += 1


def framed(img, side, frame=2):
    tile = Image.new("RGB", (side, side), "black")
    tile.paste(img.convert("RGB").resize((side - 2 * frame,) * 2, Image.LANCZOS), (frame, frame))
    return tile


def add_input(figure, source, out, tile_top, tile_side, title_rows, gap, center_y=None,
              sample="layer 2"):
    fig = Image.open(ROOT / figure).convert("RGB")
    W, H = fig.size
    margin = 20
    canvas = Image.new("RGB", (margin + tile_side + gap + W, H), "white")
    canvas.paste(fig, (margin + tile_side + gap, 0))
    top = tile_top if center_y is None else center_y - tile_side // 2
    canvas.paste(framed(Image.open(ROOT / source), tile_side), (margin, top))
    t0, t1 = title_rows
    title_bottom = t1 if center_y is None else top - (tile_top - t1)
    f = font_for_height(t1 - t0, sample)
    # Baseline of the existing titles sits above their descender ("y" in "layer"); match it.
    descent = f.getbbox(sample)[3] - f.getbbox("layer")[3] + (f.getbbox("ly")[3] - f.getbbox("l")[3])
    ImageDraw.Draw(canvas).text((margin + tile_side / 2, title_bottom - descent), "input",
                                font=f, fill="black", anchor="ms")
    canvas.save(ROOT / out, optimize=True)
    print(f"wrote {out} {canvas.size}")


if __name__ == "__main__":
    # Figure 1: one row of 446 px tiles (frames at y=58..503), titles at y=20..48.
    add_input("examples/feat_peacock.png", "examples/images/peacock.jpg",
              "paper/feat_peacock.png", tile_top=58, tile_side=446, title_rows=(20, 48), gap=40)
    # Figure 2: 2 x 3 grid of 567 px tiles (rows y=48..615 and 657..1225), titles at y=16..36.
    # The input is centred vertically beside the grid.
    add_input("examples/compare_b16_market.png", "examples/images/market.jpg",
              "paper/compare_b16_market.png", tile_top=48, tile_side=567, title_rows=(16, 36),
              gap=25, center_y=(48 + 1225) // 2, sample="DINOv3")
