"""The viral GTA-clip Reel look: ONE clip with its own audio, kept 16:9 (not cropped) in the middle of a black 9:16
frame, a white meme-caption box with emoji above it ("When Lucia gets her period in GTA 6, you have to take care of
her 😅"), small Hinglish subtitles below. One funny moment = one Reel."""
import re

from PIL import Image, ImageDraw, ImageFont

from . import qa
from .ai import check_clear, clean_word, hinglish, llm, transcribe
from .captions import HEAD, caption_events, ts
from .ff import LOUD, FPS, H, OUT, W, encode, probe, safe_name

TEXT_FONT, EMOJI_FONT = "C:/Windows/Fonts/georgiab.ttf", "C:/Windows/Fonts/seguiemj.ttf"
VID_W = 1180  # scale a little past the frame width and crop, so the clip is bigger than plain letterbox
CAPTION_PROMPT = """This is a clip from {channel} ("{title}"). What the frames show, in order: {shows}
What is said in it (speech recognition, may contain errors): {text}
Describe only what is really shown or said; invent nothing.

Write the on-screen caption for an Indian Instagram Reel that makes people stop, relate, laugh and SEND it to a friend.
First decide the vibe of the clip:
- comedy -> desi meme humour in Hinglish (Roman script), the kind that goes viral in India: relatable "jab ..." /
  "when ..." situations, savage, and if it fits naturally a cheeky DOUBLE-MEANING line (innuendo Indians instantly get,
  like "Bhai ne pehli baar mein hi ghusa diya 😏" about parking a car) - suggestive at most, never explicit, no slurs.
- serious / cinematic / emotional -> a dramatic, quotable line that matches the moment.
Examples of the style that works: "When Lucia gets her period in GTA 6, you have to take care of her 😅",
"Bro stole a fighter jet like it's just another car 💀". One sentence, max 16 words, ends with ONE fitting emoji,
describe the situation, never explain the joke, no hashtags.
Also: "post": the Instagram caption (2 short lines in the same vibe, the second a question that gets comments),
"hashtags": exactly 5, no #. Return JSON: {{"vibe": "comedy|serious", "caption": "...", "post": "...", "hashtags": [...]}}"""


def is_emoji(ch):
    o = ord(ch)
    return o >= 0x1F000 or 0x2600 <= o <= 0x27BF or o in (0x2764, 0xFE0F, 0x200D)


def runs(text):
    """Split into (is_emoji, text) runs so each part is drawn with the right font."""
    out = []
    for ch in text:
        e = is_emoji(ch)
        if out and out[-1][0] == e:
            out[-1][1] += ch
        else:
            out.append([e, ch])
    return out


def caption_png(text, path, max_w=940, size=50):
    """White rounded box, black bold serif text, colour emoji - centred lines."""
    # Pillow can't join ZWJ emoji sequences (🏃‍♂️ = 🏃 + ♂): keep the base emoji only
    text = re.sub("‍[♀♂⚕✈❤]️?|‍|️|[\U0001F3FB-\U0001F3FF]", "", text)
    font, emoji = ImageFont.truetype(TEXT_FONT, size), ImageFont.truetype(EMOJI_FONT, int(size * 0.9))
    probe_d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    width = lambda s: sum(probe_d.textlength(t, font=emoji if e else font) for e, t in runs(s))
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and width(trial) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    lines.append(cur)
    pad, gap = 30, int(size * 1.25)
    box_w = int(max(width(l) for l in lines)) + 2 * pad
    box_h = gap * len(lines) + 2 * pad - int(size * 0.2)
    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, box_w - 1, box_h - 1), radius=28, fill="white")
    for i, line in enumerate(lines):
        x, y = (box_w - width(line)) / 2, pad + i * gap
        for e, t in runs(line):
            d.text((x, y + (size * 0.05 if e else 0)), t, font=emoji if e else font, fill="black", embedded_color=e)
            x += probe_d.textlength(t, font=emoji if e else font)
    img.save(path)
    return box_w, box_h


def reel(clip, caption, job, out, k=0):
    """clip (16:9, own audio) + caption -> finished 9:16 Reel."""
    dur, _, sw, sh = probe(clip)
    vid_h = round(sh * VID_W / sw / 2) * 2
    top = (H - vid_h) // 2 + 60  # a little below centre: room for the caption box
    png = job / f"caption_{k}.png"
    _, box_h = caption_png(caption, png)
    cap_y = max(60, top - box_h - 36)
    ws = [w for s in transcribe(clip) for w in s["words"]]
    check_clear(clip)
    roman = hinglish([w[2] for w in ws])
    subs = [[w[0], w[1], r, False] for w, r in zip(ws, roman)]
    sub_style = "Style: Cap,Segoe UI Black,62,&H00FFFFFF,&H000000FF,&H00000000,&H99000000,0,0,0,0,100,100,1,0,1,6,3,2,70,70,"
    head = re.sub(r"Style: Cap,[^\n]*", sub_style + f"{H - top - vid_h - 150},1", HEAD)
    (job / f"subs_{k}.ass").write_text(head + "\n".join(f"Dialogue: 0,{ts(a)},{ts(b)},Cap,,0,0,0,,{t}"
                                                       for a, b, t in caption_events(subs)) + "\n", encoding="utf-8")
    fc = (f"[0:v]scale={VID_W}:{vid_h},crop={W}:{vid_h},eq=contrast=1.06:saturation=1.15,unsharp=5:5:0.5,"
          f"pad={W}:{H}:0:{top}:black,setsar=1,fps={FPS}[bg];[bg][1:v]overlay=(W-w)/2:{cap_y}[c];"
          f"[c]ass=subs_{k}.ass,format=yuv420p[vout];[0:a]{LOUD}[aout]")
    encode(["ffmpeg", "-y", "-i", clip, "-i", png.name, "-filter_complex", fc, "-map", "[vout]", "-map", "[aout]",
            "-t", f"{round(dur * FPS) / FPS:.3f}", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"],
           out.resolve(), cwd=job)
    return job / f"subs_{k}.ass"



def singles(job, topic, clips):
    """One Reel per clip: [{file, channel, title, url, shows, tag}] -> [(video, meta, qa problems)]. Nothing is added to the
    clip's own sound."""
    made = []
    for c in clips:
        text = " ".join(s["text"] for s in transcribe(c["file"]))
        r = llm(CAPTION_PROMPT.format(channel=c.get("channel", ""), title=c.get("title", ""), shows="; ".join(c.get("shows", [])) or "unknown",
                                      text=text[:3000])) or {}
        caption = " ".join(map(clean_word, (r.get("caption") or f"{topic} me ye kya ho gaya 💀").split()))
        out = OUT / f"{safe_name(topic, 20)}_reel_{safe_name(c['tag'], 40)}.mp4"
        OUT.mkdir(exist_ok=True)
        subs = reel(c["file"], caption, job, out, safe_name(c["tag"], 40))
        tags = [str(t).lstrip("#").replace(" ", "") for t in r.get("hashtags") or [topic.replace(" ", ""), "gaming", "funny"]][:5]
        meta = {"title": caption, "caption": (r.get("post") or caption) + "\n\n" + " ".join("#" + t for t in tags), "hashtags": tags,
                "credits": [f"{c.get('channel')} - {c.get('url')}"]}
        out, problems = qa.deliver(out, meta, "reel", [subs], length=round(probe(c["file"])[0] * FPS) / FPS)
        made.append((out, meta, problems))
    return made
