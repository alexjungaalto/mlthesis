#!/usr/bin/env python3
"""Render web/social-card.png, the 1200x630 Open Graph preview image for
ml-theses.org (referenced from web/overrides/main.html).

Run once after changing the wording below, then commit the PNG:
    python3 web/make_social_card.py
Requires Pillow (pip install pillow) and a system sans-serif font.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
INDIGO = (63, 81, 181)       # Material "indigo" primary, matches the site theme
INDIGO_DARK = (48, 63, 159)
WHITE = (255, 255, 255)
LIGHT = (197, 202, 233)

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def font(size, bold=True):
    for path in FONT_CANDIDATES:
        if ("Bold" in path) == bold and Path(path).exists():
            return ImageFont.truetype(path, size)
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


img = Image.new("RGB", (W, H), INDIGO)
d = ImageDraw.Draw(img)
# Darker band at the bottom for the byline.
d.rectangle([0, H - 110, W, H], fill=INDIGO_DARK)

x = 80
d.text((x, 90), "ML Theses", font=font(96), fill=WHITE)
d.text((x, 215), "ml-theses.org", font=font(40, bold=False), fill=LIGHT)

lines = [
    "A guide to doing a bachelor's, master's, or doctoral",
    "thesis in applied machine learning,",
    "plus open topics and supervised theses.",
]
y = 310
for line in lines:
    d.text((x, y), line, font=font(44, bold=False), fill=WHITE)
    y += 60

d.text((x, H - 78), "Alex Jung  ·  Aalto University  ·  IMC Krems",
       font=font(34, bold=False), fill=LIGHT)

out = Path(__file__).with_name("social-card.png")
img.save(out, optimize=True)
print(f"wrote {out} ({W}x{H})")
