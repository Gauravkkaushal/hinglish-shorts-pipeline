"""Reel cover: the most striking frame of the edit + a big neon title. Instagram's profile grid shows only the middle
3:4 of a 9:16 cover, so the title sits inside that safe area."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .eyes import faces
from .ff import ROOT, probe

FONTS = ROOT / "assets" / "fonts"


def best_frame(video, samples=24):
    """-> PIL image of the sharpest, brightest, most colourful frame (skipping the first/last second); frames with a
    character's face count double - faces get the clicks on a profile grid."""
    import cv2
    dur = probe(video)[0]
    cap, best, best_score = cv2.VideoCapture(str(video)), None, -1.0
    for t in np.linspace(1.0, max(1.0, dur - 1.0), samples):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(frame, (270, 480))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        score = (cv2.Laplacian(gray, cv2.CV_64F).var() ** 0.5) * (gray.mean() / 255) * \
                (1 + cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255)
        if faces(cv2.resize(frame, (540, int(frame.shape[0] * 540 / frame.shape[1])))):
            score *= 2
        if score > best_score:
            best, best_score = frame, score
    cap.release()
    return Image.fromarray(best[..., ::-1]) if best is not None else None


def wrap(draw, text, font, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and draw.textlength(trial, font=font) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur]


def make_cover(video, title, kicker, out, accent=(255, 31, 61), size=(1080, 1920)):
    """Frame filling the cover, darkened towards the title, `kicker` (small, Bebas) above a huge neon `title`."""
    w, h = size
    frame = best_frame(video)
    bg = frame.resize((w, int(frame.height * w / frame.width))) if frame else Image.new("RGB", size, "black")
    if bg.height < h:  # landscape frame: fill with a blurred copy, keep the sharp frame in the middle
        blur = frame.resize((int(frame.width * h / frame.height), h)).filter(ImageFilter.GaussianBlur(30))
        canvas = blur.crop(((blur.width - w) // 2, 0, (blur.width - w) // 2 + w, h))
        canvas.paste(bg, (0, (h - bg.height) // 2))
        bg = canvas
    bg = bg.crop((0, (bg.height - h) // 2, w, (bg.height - h) // 2 + h)).convert("RGBA")
    shade = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(shade).rectangle((0, 0, w, h), fill=(0, 0, 0, 110))  # overall dim so the title pops
    img = Image.alpha_composite(bg, shade)

    title_font = ImageFont.truetype(str(FONTS / "AbrilFatface-Regular.ttf"), 120)
    kicker_font = ImageFont.truetype(str(FONTS / "BebasNeue-Regular.ttf"), 96)
    d = ImageDraw.Draw(img)
    lines = wrap(d, title, title_font, w - 140)
    line_h = 138
    y0 = h // 2 - (len(lines) * line_h) // 2 + 40  # centred: inside the 3:4 grid crop
    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for i, line in enumerate(lines):
        x = (w - d.textlength(line, font=title_font)) / 2
        gd.text((x, y0 + i * line_h), line, font=title_font, fill=accent + (255,), stroke_width=14, stroke_fill=accent + (255,))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(18)))
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        x = (w - d.textlength(line, font=title_font)) / 2
        d.text((x, y0 + i * line_h), line, font=title_font, fill="white", stroke_width=3, stroke_fill="black")
    kx = (w - d.textlength(kicker, font=kicker_font)) / 2
    d.text((kx, y0 - 130), kicker, font=kicker_font, fill=accent, stroke_width=4, stroke_fill="black")
    img.convert("RGB").save(out, quality=95)
    return out


if __name__ == "__main__":
    import sys
    print(make_cover(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
