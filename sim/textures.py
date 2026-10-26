"""Procedurally generated sign textures, written to sim/assets/ on first use.

The point of the textured-3D sim is that the YOLO detector (trained on real
road imagery) has something realistic to fire on. A flat-shaded box doesn't
look like a stop sign to it; a red octagon with white "STOP" does. These are
generated with Pillow so the repo carries no binary image assets — same spirit
as the Webots world pulling its textures from URLs rather than vendoring them.

Swap in a photograph of a real sign by dropping a file at the same path and
deleting the generated one; everything downstream just reads the path.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASSET_DIR = Path(__file__).parent / "assets"

# System fonts to try, in order, for crisp sign lettering. Falls back to an
# upscaled bitmap of Pillow's default font if none are present.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",   # macOS
    "/System/Library/Fonts/HelveticaNeue.ttc",             # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
    "Arial Bold.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_centered(draw: ImageDraw.ImageDraw, box, text, font, fill):
    """Draw text centred inside box=(x0,y0,x1,y1)."""
    x0, y0, x1, y1 = box
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    draw.text((x0 + (x1 - x0 - tw) / 2 - l, y0 + (y1 - y0 - th) / 2 - t),
              text, font=font, fill=fill)


def stop_sign(size: int = 512) -> str:
    """A US stop sign: red regular octagon, white ring, white STOP. Returns path.
    Background is transparent so the quad shows only the octagon."""
    path = ASSET_DIR / "stop_sign.png"
    if path.exists():
        return str(path)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx = cy = size / 2
    r = size * 0.48
    # a regular octagon, flat side up (standard orientation)
    pts = [
        (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
        for a in range(22, 360, 45)
    ]
    draw.polygon(pts, fill=(197, 24, 24, 255))          # sign red
    # white ring just inside the edge
    inner = [
        (cx + r * 0.86 * math.cos(math.radians(a)), cy + r * 0.86 * math.sin(math.radians(a)))
        for a in range(22, 360, 45)
    ]
    draw.line(pts + [pts[0]], fill=(255, 255, 255, 255), width=max(3, size // 40))
    draw.line(inner + [inner[0]], fill=(255, 255, 255, 255), width=max(2, size // 90))
    _draw_centered(draw, (0, size * 0.33, size, size * 0.67),
                   "STOP", _load_font(int(size * 0.28)), (255, 255, 255, 255))
    # The sign faces oncoming traffic, so the camera sees the quad's back face;
    # pre-flipping horizontally makes the lettering read correctly in-scene.
    img.transpose(Image.FLIP_LEFT_RIGHT).save(path)
    return str(path)


def speed_limit(value: int = 35, size: int = 512) -> str:
    """A US speed-limit sign (scene realism; not a detector class). Returns path."""
    path = ASSET_DIR / f"speed_limit_{value}.png"
    if path.exists():
        return str(path)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGBA", (int(size * 0.62), size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    w, h = img.size
    draw.rounded_rectangle([2, 2, w - 3, h - 3], radius=size // 18,
                           fill=(255, 255, 255, 255), outline=(0, 0, 0, 255),
                           width=max(3, size // 48))
    _draw_centered(draw, (0, h * 0.06, w, h * 0.34), "SPEED",
                   _load_font(int(size * 0.13)), (0, 0, 0, 255))
    _draw_centered(draw, (0, h * 0.28, w, h * 0.52), "LIMIT",
                   _load_font(int(size * 0.13)), (0, 0, 0, 255))
    _draw_centered(draw, (0, h * 0.5, w, h * 0.96), str(value),
                   _load_font(int(size * 0.4)), (0, 0, 0, 255))
    img.transpose(Image.FLIP_LEFT_RIGHT).save(path)  # see stop_sign() note
    return str(path)


if __name__ == "__main__":
    print("stop sign ->", stop_sign())
    print("speed limit ->", speed_limit(35))
    print("speed limit ->", speed_limit(25))
