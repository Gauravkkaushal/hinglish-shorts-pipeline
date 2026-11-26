"""Hinglish captions (ASS): 1-3 word chunks in caps, the spoken word lights up yellow, emphasis words green,
each chunk pops in, shown a hair before the audio so the eye reads ahead. Captions vanish during meme cut-ins."""
import re

YELLOW, GREEN = r"&H0000E1FF&", r"&H0088FF2B&"  # ASS colours are BGR
HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Segoe UI Black,92,&H00FFFFFF,&H000000FF,&H00000000,&H99000000,0,0,0,0,100,100,1,0,1,7,4,2,70,70,560,1
Style: Title,Segoe UI Black,62,&H00000000,&H000000FF,&H0000E1FF,&H0000E1FF,0,0,0,0,100,100,0,0,3,18,0,8,90,90,150,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def lines_of(words, gap=0.5, max_words=14, max_chars=None):
    """Group [start, end, text, ...] words into lines: break on pauses, word count, or line length."""
    lines, cur = [], []
    for w in words:
        too_long = max_chars and len(" ".join(x[2] for x in cur + [w])) > max_chars
        if cur and (w[0] - cur[-1][1] > gap or len(cur) >= max_words or too_long):
            lines.append(cur)
            cur = []
        cur.append(w)
    return lines + [cur] if cur else lines


def ts(t):
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def caption_events(words, stops=(), lead=0.08, max_words=3, max_chars=14):
    """words: [start, end, text, emphasised]. stops: meme cut-in times (captions end there). -> [(start, end, ass_text)]"""
    words = sorted(words, key=lambda w: w[0])  # ASR timestamps are not always monotonic
    out, chunks, rest = [], [], words
    for s in [*sorted(stops), 1e9]:  # no caption runs across a meme or the end of the hook
        chunks += lines_of([w for w in rest if w[0] < s], gap=0.35, max_words=max_words, max_chars=max_chars)
        rest = [w for w in rest if w[0] >= s]
    for n, chunk in enumerate(chunks):
        nxt = min([chunks[n + 1][0][0] if n + 1 < len(chunks) else 1e9] + [s for s in stops if s > chunk[0][0]])
        texts = [re.sub(r"[{}\\]", "", w[2]).upper() for w in chunk]
        for i, w in enumerate(chunk):
            end = chunk[i + 1][0] if i + 1 < len(chunk) else min(w[1] + 0.25, nxt)
            parts = []
            for j, (t, cw) in enumerate(zip(texts, chunk)):
                colour = GREEN if cw[3] else YELLOW if j == i else None
                size = r"\fscx110\fscy110" if j == i else ""
                parts.append(f"{{\\c{colour}{size}}}{t}{{\\r}}" if colour or size else t)
            pop = r"{\fscx70\fscy70\t(0,90,\fscx106\fscy106)\t(90,150,\fscx100\fscy100)}" if i == 0 else ""
            out.append([w[0] - lead, max(end, w[0] + 0.05) - lead, pop + " ".join(parts)])
    for cur, nxt in zip(out, out[1:]):  # never two captions on screen at once
        cur[1] = min(cur[1], nxt[0])
    return [tuple(e) for e in out if e[1] - e[0] > 0.01]


def write_ass(path, words, title="", stops=(), outro=None):
    """outro: (start, end, text) shown under the captions at the end of a part."""
    rows = [f"Dialogue: 1,{ts(0)},{ts(2.8)},Title,,0,0,0,,{{\\fad(120,250)}}{re.sub(r'[{}]', '', title)}"] if title else []
    rows += [f"Dialogue: 0,{ts(a)},{ts(b)},Cap,,0,0,0,,{txt}" for a, b, txt in caption_events(words, stops)]
    if outro:
        pop = r"{\an8\pos(540,1405)\1c&HFFFFFF&\3c&H1E1EDC&\fscx60\fscy60\t(0,120,\fscx106\fscy106)\t(120,200,\fscx100\fscy100)}"
        rows.append(f"Dialogue: 2,{ts(outro[0])},{ts(outro[1])},Title,,0,0,0,,{pop}{outro[2]}")
    path.write_text(HEAD + "\n".join(rows) + "\n", encoding="utf-8")
