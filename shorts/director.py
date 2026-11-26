"""The Director: decides where a reaction meme cuts in, which words get emphasis, and the title."""
import difflib, json, random

from .ai import HINDI, bones, llm
from .captions import lines_of
from .ff import IMG_EXT, JOBS, LIB, VID_EXT
from .rights import has_music
from .eyes import shows_someone

NOT_TAGS = {"sfx", "music", "broll", "emoji", "_quarantine_copyrighted_music"}
# Only used without an LLM key. Short keys (<=2 chars) must match the whole word, longer keys match the word start.
KEYWORDS = {
    "laugh": ["हँस", "हंस", "हँसी", "हंसी", "मज़ा", "मजा", "मज़ाक", "मजाक", "पागल", "हाहा", "lol", "funny", "joke"],
    "cry": ["रो", "रोया", "दुख", "दर्द", "आँसू", "इमोशनल", "sad", "cry"],
    "shock": ["तभी", "अचानक", "ट्विस्ट", "सन्न", "सीरियसली", "करोड़", "shock", "omg", "seriously"],
    "facepalm": ["गलती", "ग़लती", "बेवकूफ", "fail"],
    "angry": ["गुस्सा", "गुस्से", "चिल्ला", "angry"],
    "awkward": ["घूर", "शर्म", "awkward"],
    "celebrate": ["जीत", "पार्टी", "win", "success"],
}
PROMPT = """You are India's best meme-video editor. Transcript of a short video; every word is numbered as index:word and
each line starts with its time in seconds:
{lines}

Reaction memes you can CUT TO (tag: clip name, and what the clip says/sounds like):
{memes}

How a cut works: the video PAUSES right after word "after_word", the meme plays full screen WITH its own sound, then the
video continues. Use one ONLY where it makes the moment clearly funnier - a punchline that just landed, a savage or
shocking line, an awkward admission, an emotional hit - like a viewer's instant reaction to exactly what was just said.
after_word = the LAST word of that punchline. 0-3 cuts; quality over quantity; if nothing is truly funny return [].
No two cuts within 6 seconds, none in the first 2 seconds. Score each cut honestly 1-10.

Also pick "emphasis": indexes of 4-10 words that carry the meaning/punch (punchline words, numbers, names, strong words);
they get highlighted and a quick zoom.
"punch": the LAST word index of each line (1-4 of them) where the audience laughs or gasps hardest.
"hook": ONE short spoken line (a whole sentence of 4-12 words, copied word for word from the transcript) that is played first, before the video starts, to stop
the scroll: a strange, bold or forbidden-sounding line from later in the talk that makes people ask "what? why?" without
giving the final payoff away. It must make sense alone. "" if there is none.
Return JSON:
{{"cuts": [{{"after_word": <index>, "tag": "<tag>", "file": "<exact clip name>", "why": "<short>", "score": <1-10>}}],
 "emphasis": [<indexes>], "punch": [<indexes>], "hook": "<words>",
 "title": "<3-6 words, natural Hinglish (Roman) a friend would text about the setup; never gives the punchline away, never makes
 a group of people the joke, never the words meme/reaction/video. Good: 'Girls school ka asli sach', 'Cake hai ya EMI?'>",
 "hashtags": ["5 tags without #"]}}"""


def library():
    """{tag: [clip paths]} - every folder in library/ (except sfx/music/broll/emoji) is a tag. Left out: clips with music in
    them (their songs would be copyrighted), clips that do not show a face, the demo pictures."""
    lib = {}
    for d in sorted(p for p in LIB.iterdir() if p.is_dir() and p.name not in NOT_TAGS):
        files = [f for f in d.iterdir() if f.suffix.lower() in IMG_EXT or f.suffix.lower() in VID_EXT and not has_music(f) and shows_someone(f)]
        files = [f for f in files if not f.name.startswith("_demo_")]
        if files:
            lib[d.name] = files
    return lib


USED = JOBS / "memes_used.json"


def fresh(lib, used, least=8):
    """The library without the memes of the last videos, as long as enough remain."""
    for n in range(len(used), -1, -1):
        left = {t: ok for t, fs in lib.items() if (ok := [f for f in fs if f.name not in used[len(used) - n:]])}
        if sum(map(len, left.values())) >= min(least, sum(map(len, lib.values()))):
            return left


def remember(cuts):
    used = json.loads(USED.read_text(encoding="utf-8")) if USED.exists() else []
    USED.parent.mkdir(exist_ok=True)
    USED.write_text(json.dumps((used + [c["path"].name for c in cuts])[-12:], ensure_ascii=False), encoding="utf-8")


def says(clip):
    txt = clip.with_suffix(".txt")
    return txt.read_text(encoding="utf-8").strip() if txt.exists() else ""


def rule_plan(words):
    """No-LLM fallback: keyword hits cut in right after the word."""
    cuts = []
    for i, (_, w1, txt) in enumerate(words):
        word = txt.strip(" ,.!?।\"'").lower()  # not \w: it would strip Hindi matras
        for tag, keys in KEYWORDS.items():
            if any(word == k or (len(k) > 2 and word.startswith(k)) for k in keys):
                cuts.append({"t": w1, "tag": tag, "score": 7})
                break
    return {"cuts": cuts, "emphasis": [], "punch": [], "hook": None, "title": "", "hashtags": []}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def word_end(words, i):
    """LLM word index -> that word's end time; None if the index is invalid."""
    try:
        i = int(i)
    except (TypeError, ValueError):
        return None
    return words[i][1] if 0 <= i < len(words) else None


def hook_of(h, words, start=6.0, lo=1.0, hi=4.0):
    """The hook's words -> (first, last) index, if that line can open the video: late enough to be a tease and 1-4s long.
    Found by its words: the model's own word numbers are often a few off."""
    want = [bones([w]) for w in str(h or "").split()]
    have = [bones([w[2]]) for w in words]
    n = len(want)
    if not 3 <= n <= len(have):
        return None
    alike = [difflib.SequenceMatcher(None, "|".join(have[i:i + n]), "|".join(want), autojunk=False).ratio() for i in range(len(have) - n + 1)]
    i = max(range(len(alike)), key=alike.__getitem__)
    j = i + n - 1
    return (i, j) if alike[i] >= 0.8 and words[i][0] >= start and lo <= words[j][1] - words[i][0] <= hi else None


def from_llm(r, words):
    """LLM answer (word indexes) -> plan in seconds."""
    cuts = [{"t": t, "tag": c.get("tag"), "file": c.get("file"), "score": num(c.get("score")), "why": c.get("why", "")}
            for c in r.get("cuts") or [] if isinstance(c, dict) and (t := word_end(words, c.get("after_word"))) is not None]
    emph = sorted({int(i) for i in r.get("emphasis") or [] if word_end(words, i) is not None})
    punch = sorted({int(i) for i in r.get("punch") or [] if word_end(words, i) is not None})[:4]
    return {"cuts": cuts, "emphasis": emph, "punch": punch, "hook": hook_of(r.get("hook"), words), "title": str(r.get("title") or ""), "hashtags": r.get("hashtags") or []}


def clean_plan(plan, lib, dur, min_score=8, min_gap=6.0):
    """Keep only strong cuts on known tags, inside the video, >= min_gap apart; attach the real clip file."""
    cuts, used, last = [], set(), -99.0
    for c in sorted((c for c in plan["cuts"] if c.get("tag") in lib), key=lambda c: c["t"]):
        t = float(c["t"])
        if c.get("score", 0) < min_score or not 2.0 <= t <= dur or t - last < min_gap:
            continue
        files = lib[c["tag"]]
        named = [f for f in files if f.stem == c.get("file")]
        fresh = [f for f in files if f not in used] or files
        path = (named or [random.choice(fresh)])[0]
        used.add(path)
        cuts.append({"t": round(t, 2), "tag": c["tag"], "path": path, "why": c.get("why", "")})
        last = t
    return {**plan, "cuts": cuts, "title": plan["title"][:60],
            "hashtags": [str(h).lstrip("#").replace(" ", "") for h in plan["hashtags"]][:8]}


def direct(words, dur, lib):
    lib = fresh(lib, json.loads(USED.read_text(encoding="utf-8")) if USED.exists() else [])
    lines, i = [], 0
    for line in lines_of(words, gap=0.5, max_words=14):
        lines.append(f"[{line[0][0]:.1f}] " + " ".join(f"{i + j}:{w[2]}" for j, w in enumerate(line)))
        i += len(line)
    memes = "\n".join(f"{t}: " + "; ".join(f'{f.stem} (says: "{says(f)}")' if says(f) else f.stem for f in fs[:25])
                      for t, fs in lib.items())
    r = llm(PROMPT.format(lines="\n".join(lines)[:14000], memes=memes), model=HINDI)
    plan = from_llm(r, words) if r else rule_plan(words)
    for c in plan["cuts"]:
        print(f"  director: cut at {c['t']:.1f}s -> {c['tag']} '{c.get('file') or ''}' score {c.get('score')} {c.get('why', '')}")
    plan = clean_plan(plan, lib, dur)
    remember(plan["cuts"])
    return plan
