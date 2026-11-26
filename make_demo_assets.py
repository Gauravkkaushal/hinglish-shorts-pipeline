"""Creates library/ folders + emoji PNGs + a few _demo_ placeholder clips so the pipeline runs today.
Replace the _demo_ files with your real 200+ clips (just drop them in the matching folder)."""
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
LIB = ROOT / "library"
TAGS = {  # folder (= tag the Director can pick) -> emoji shown with it
    "laugh": "😂", "cry": "😭", "shock": "😱", "angry": "😡", "facepalm": "🤦",
    "awkward": "😬", "celebrate": "🎉", "thinking": "🤔", "bruh": "💀", "cool": "😎", "love": "❤️",
}
COLORS = [(30, 30, 60), (60, 20, 40), (20, 50, 40)]
EMOJI_FONT = "C:/Windows/Fonts/seguiemj.ttf"
TEXT_FONT = "C:/Windows/Fonts/impact.ttf"


def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def emoji_png(ch, path, size=256):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((size // 2, size // 2), ch, font=ImageFont.truetype(EMOJI_FONT, int(size * 0.75)),
                             embedded_color=True, anchor="mm")
    img.save(path)


def card(path, emoji, label, color, w=1080, h=1080):
    img = Image.new("RGB", (w, h), color)
    d = ImageDraw.Draw(img)
    d.text((w // 2, h // 2 - 80), emoji, font=ImageFont.truetype(EMOJI_FONT, 420), embedded_color=True, anchor="mm")
    d.text((w // 2, h - 160), label, font=ImageFont.truetype(TEXT_FONT, 110), fill="white", anchor="mm",
           stroke_width=6, stroke_fill="black")
    img.save(path)


def main():
    for d in ["images", "sfx", "music", "broll", "emoji", *TAGS]:
        (LIB / d).mkdir(parents=True, exist_ok=True)
    tmp = LIB / "emoji"
    for tag, ch in TAGS.items():
        emoji_png(ch, tmp / f"{tag}.png")
        for i, color in enumerate(COLORS[:2]):
            out = LIB / tag / f"_demo_{tag}_{i}.mp4"
            if out.exists():
                continue
            png = tmp / f"_card_{tag}_{i}.png"
            card(png, ch, f"{tag.upper()} (demo)", color)
            freq = 300 + 60 * i + 20 * len(tag)
            ff("-loop", "1", "-i", str(png), "-f", "lavfi", "-i", f"sine=f={freq}:d=2.5",
               "-filter_complex", "[0:v]zoompan=z='1+0.003*on':d=75:s=1080x1080:fps=30[v];[1:a]tremolo=f=8:d=0.8,volume=0.5[a]",
               "-map", "[v]", "-map", "[a]", "-t", "2.5", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out))
            png.unlink()
    for i, (ch, label) in enumerate([("🗿", "SIGMA (demo)"), ("🤡", "CLOWN (demo)")]):
        card(LIB / "images" / f"_demo_meme_{i}.png", ch, label, COLORS[i])
    # sfx: named <tag>.wav plays for that tag, pop.wav is the default
    sfx = LIB / "sfx"
    ff("-f", "lavfi", "-i", "sine=f=900:d=0.12", "-af", "afade=t=out:d=0.1,volume=0.6", str(sfx / "pop.wav"))
    ff("-f", "lavfi", "-i", "sine=f=55:d=0.9", "-af", "afade=t=out:st=0.1:d=0.8,volume=3", str(sfx / "shock.wav"))
    ff("-f", "lavfi", "-i", "anoisesrc=d=0.4:c=pink", "-af", "highpass=f=800,afade=t=in:d=0.15,afade=t=out:st=0.15:d=0.25,volume=0.5", str(sfx / "whoosh.wav"))
    for i, colors in enumerate(["0x1a1a40 0x5a189a 0x0f3460", "0x3a0ca3 0xf72585 0x101020"]):
        out = LIB / "broll" / f"_demo_bg_{i}.mp4"
        if not out.exists():
            ff("-f", "lavfi", "-i", f"gradients=s=1080x1920:d=20:speed=0.02:c0={colors.split()[0]}:c1={colors.split()[1]}:c2={colors.split()[2]}:n=3",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", str(out))
    print("library ready:", LIB)


if __name__ == "__main__":
    main()
