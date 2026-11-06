"""Read the number off a speed-limit sign — classical CV, template matching.

This is PERCEPTION, not control. Given a frame and a sign's bounding box (from
the detector, or the sim's ground truth), it crops the sign and RECOGNIZES the
digits from pixels — so the limit is *read*, not handed to us (that was the whole
point of doing this with computer vision rather than reading a value off GT).

How it works:
  1. crop the sign, keep only the lower "number" band (drops SPEED/LIMIT + border)
  2. upscale, grayscale, Otsu-threshold -> black digits become white blobs
  3. find the digit blobs, left-to-right
  4. match each against rendered 0-9 templates; join them into a number

Templates are drawn from the same bold font the sim letters its signs with, so a
rendered '2' matches an on-screen '2'. For real-world footage you'd swap the
recognizer for a trained classifier or Tesseract; the interface stays the same.
"""
from __future__ import annotations

import cv2
import numpy as np

# Same font candidates the sim uses (sim/textures.py), so templates match the
# on-screen lettering. Kept local to avoid a vision->sim import dependency.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "Arial Bold.ttf",
]

_CANON = (40, 64)          # (w, h) every glyph is resized to before matching
_MATCH_MIN = 0.35          # reject a digit whose best template score is below this
_MIN_SIGN_PX = 22          # legibility gate: sign bbox shorter than this = unreadable
_templates: dict[int, np.ndarray] | None = None  # lazy cache {digit: canon glyph}


def _load_font(size: int):
    from PIL import ImageFont
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _tight_canon(binary: np.ndarray) -> np.ndarray | None:
    """Crop a white-on-black glyph to its ink and resize to the canonical box."""
    ys, xs = np.where(binary > 127)
    if len(xs) == 0:
        return None
    crop = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(crop, _CANON, interpolation=cv2.INTER_AREA)


def _build_templates() -> dict[int, np.ndarray]:
    from PIL import Image, ImageDraw
    font = _load_font(200)
    tpl = {}
    for d in range(10):
        img = Image.new("L", (240, 300), 0)             # black background
        ImageDraw.Draw(img).text((40, 10), str(d), font=font, fill=255)
        glyph = _tight_canon(np.array(img))
        if glyph is not None:
            tpl[d] = glyph
    return tpl


def _match_digit(glyph: np.ndarray) -> tuple[int | None, float]:
    assert _templates is not None
    g = _tight_canon(glyph)
    if g is None:
        return None, 0.0
    best_d, best_s = None, -1.0
    for d, t in _templates.items():
        s = float(cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)[0, 0])
        if s > best_s:
            best_s, best_d = s, d
    return best_d, best_s


def read_speed_limit(frame: np.ndarray, bbox, min_sign_px: int = _MIN_SIGN_PX) -> int | None:
    """Recognize the limit on the speed-limit sign at `bbox`, or None if it can't.

    Returns None when the sign is too small to read (legibility gate), when no
    digits are found, or when any digit is matched with low confidence — a clean
    'not sure this frame' that the caller debounces across frames.
    """
    global _templates
    if _templates is None:
        _templates = _build_templates()

    x1, y1, x2, y2 = (int(v) for v in bbox)
    if y2 - y1 < min_sign_px:                           # too far / small to read
        return None
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    h, w = crop.shape[:2]
    if h < 3 or w < 3:
        return None

    # Keep only the number band: lower part of the sign, inset from the border,
    # so SPEED/LIMIT and the black outline don't get mistaken for digits.
    band = crop[int(0.46 * h):int(0.94 * h), int(0.12 * w):int(0.88 * w)]
    if band.size == 0:
        return None

    # Upscale so the digits have enough pixels to segment and match.
    s = max(1.0, 120.0 / max(1, band.shape[0]))
    band = cv2.resize(band, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    if not boxes:
        return None
    band_h = th.shape[0]
    # Digits are the tall, not-too-wide blobs; drop specks and any wide
    # leftover border fragment.
    digit_boxes = sorted(
        (b for b in boxes
         if b[3] > 0.45 * band_h and b[3] < 1.02 * band_h and b[2] < 0.6 * th.shape[1]),
        key=lambda b: b[0],
    )
    if not digit_boxes:
        return None

    digits = []
    for (bx, by, bw, bh) in digit_boxes:
        d, score = _match_digit(th[by:by + bh, bx:bx + bw])
        if d is None or score < _MATCH_MIN:
            return None                                 # unsure -> skip this frame
        digits.append(d)
    if not (1 <= len(digits) <= 3):
        return None
    return int("".join(str(d) for d in digits))
