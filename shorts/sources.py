"""Where the raw material comes from: an AI story, your clips/links, or auto-discovery on YouTube.
Every source comes out as: video (30fps CFR, 48kHz) + Devanagari/English words with timestamps."""
import json, math, random, re
from itertools import combinations, pairwise
from pathlib import Path

from .ai import llm, transcribe, tts
from .ff import COVER, FPS, H, JOBS, LIB, VID_EXT, W, X264, crop_box, run, probe

NORM = (f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
        f"setsar=1,fps={FPS}")  # landscape canvas so clips from different videos concat cleanly
AUDIO = ["-c:a", "aac", "-ar", "48000", "-ac", "2"]

STORY_PROMPT = """Write a funny/emotional Hinglish YouTube Short voiceover about: {topic}
Rules: Hindi in Devanagari script (English words allowed in Devanagari), 90-130 words, first line is a strong hook,
short punchy sentences, a clear story with build-up and a punchline/twist, end with a question asking viewers to comment.
Return JSON: {{"title": "<max 6 word hook in Hinglish Roman script>", "script": "<voiceover text>"}}"""
SEGMENT_PROMPT = """You are a viral Shorts editor. Transcript segments from {n} source video(s) ("S<src> [start-end] text").
{topic}Pick 1-4 segments that together make ONE coherent story of 30-50 seconds: a hook at the start, a clear setup, and
a punchline/payoff at the end. Each segment starts at the beginning of a sentence and the last ends right after the
payoff. Mix sources only if they truly continue the same thought.
Return JSON: {{"segments": [{{"src": 0, "start": 12.3, "end": 20.1}}]}}
{lines}"""
STORY_PICK_PROMPT = """You are a viral Hindi Shorts editor. Transcript of part of the video "{title}" ([start-end] seconds):
{lines}

Find the single best self-contained moment for a 30-50 second Short. It must make complete sense to someone who never
saw this video: a clear setup, then a funny / savage / emotional punchline or payoff at the end. ONE continuous range
that starts at the beginning of a sentence (never mid-thought) and ends right after the payoff.
Only if the story truly needs it (the payoff comes later and every second until then stays gripping) take 62-150
seconds instead: it is then posted as a series of 30-50 second parts.
Rate honestly how much Indian Shorts viewers would enjoy and share it (1-10); most moments are 5-6.
Return JSON: {{"start": <sec>, "end": <sec>, "score": <1-10>, "summary": "<what happens, one line>",
 "title": "<3-6 words, natural Hinglish a friend would text, about this moment; relatable or cheeky, never gives the
 punchline away, correct grammar. Good: 'Girls school ka asli sach', 'Sharma ji ka beta bhi fail'>"}}"""
BREAK_PROMPT = """This talk is posted as {n} Shorts in a row, "Part 1" to "Part {n}" ([start-end] seconds):
{lines}

Choose where the earlier part(s) stop so that a viewer must watch the next one: right after a line that raises a question
or sets up the joke, just BEFORE its payoff; never after a payoff. Every part is {lo:.0f}-{hi:.0f} seconds.
Pick {cuts} of these times: {ends}
Return JSON: {{"breaks": [<sec>]}}"""
DEMO_SCRIPT = ("भाई, कल मेरे साथ ऐसा कांड हुआ कि सुनकर तुम हँस हँस के पागल हो जाओगे। मम्मी ने बोला, बेटा सब्ज़ी ले आ। "
               "मैं पूरे स्टाइल में बाइक लेकर निकला। दुकान पहुँचा, और तभी पता चला, पैसे तो घर पर ही रह गए! "
               "मैंने सोचा, कोई बात नहीं, ऑनलाइन पेमेंट कर देता हूँ। फ़ोन निकाला, बैटरी दो परसेंट। पेमेंट करते ही फ़ोन बंद! "
               "घर आया तो मम्मी ने पूछा, सब्ज़ी कहाँ है? मैं बस रो दिया। "
               "और सबसे बड़ा ट्विस्ट, पापा पहले ही सब्ज़ी ले आए थे! कमेंट में बताओ, तुम्हारे साथ ऐसा कभी हुआ?")


def seg_lines(trans, src=None):
    return "\n".join(f"{'' if src is None else f'S{src} '}[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in trans)


def snap_to_segments(trans, a, b):
    """Snap a chosen range to whole transcript segments, so it never starts or ends mid-sentence."""
    starts, ends = [s["start"] for s in trans], [s["end"] for s in trans]
    a = min(starts, key=lambda x: abs(x - a))
    b = min((e for e in ends if e > a), key=lambda x: abs(x - b), default=b)
    return a, b


def fit_length(trans, a, b, lo=30.0, hi=50.0):
    """Grow/shrink a sentence-aligned range to lo-hi seconds using whole segments: following sentences first (the aftermath),
    then earlier ones (more setup); trim trailing sentences past hi."""
    starts, ends = [s["start"] for s in trans], [s["end"] for s in trans]
    while b - a < lo:
        later, earlier = [e for e in ends if e > b], [s for s in starts if s < a]
        if later and min(later) - a <= hi:
            b = min(later)
        elif earlier and b - max(earlier) <= hi:
            a = max(earlier)
        else:
            break
    if b - a > hi:
        b = max([e for e in ends if a < e <= a + hi], default=a + hi)
    return a, b


LONGEST = 150  # seconds of one story: at most a few parts, nobody watches part 6


def split_parts(trans, a, b, lo=30.0, hi=50.0):
    """A story too long for one Short -> [(start, end)] of lo-hi seconds each, cut between sentences where the suspense is
    highest; [] when it cannot be split that way."""
    cuts = {}  # cut point -> end of the sentence before it
    for s, nxt in pairwise(trans):
        if a < s["end"] < b:
            cuts[max(s["end"], nxt["start"] - 0.15)] = s["end"]  # the laugh stays with the line that earned it
    for n in range(math.ceil((b - a) / hi), int((b - a) // lo) + 1):
        ok = [c for c in combinations(cuts, n - 1) if all(lo <= y - x <= hi for x, y in pairwise([a, *c, b]))]
        if not ok:
            continue
        ideal = [a + (b - a) * k / n for k in range(1, n)]
        r = llm(BREAK_PROMPT.format(n=n, lines=seg_lines([s for s in trans if a <= s["start"] < b]), lo=lo, hi=hi, cuts=n - 1,
                                    ends=", ".join(f"{cuts[t]:.1f}" for t in sorted({t for c in ok for t in c})))) or {}
        try:
            wish = sorted(float(x) for x in r.get("breaks") or [])
        except (TypeError, ValueError):
            wish = []
        ideal = wish if len(wish) == n - 1 else ideal
        return list(pairwise([a, *min(ok, key=lambda c: sum(abs(x - y) for x, y in zip(c, ideal))), b]))
    return []


def series(name, job, video, spans, make, title="", credits=(), until=None):
    """One video per span of the source; several spans become "Part 1", "Part 2"... until: where the story ends (what follows is
    only its laugh). -> made videos"""
    made = []
    for k, span in enumerate(spans, 1):
        many = len(spans) > 1
        d = job / f"part{k}" if many else job
        d.mkdir(parents=True, exist_ok=True)
        src = clips(d, [str(video)], span=span)
        src["credits"] = list(credits) or src["credits"]
        src["until"] = until and k == len(spans) and until - span[0]
        if many:
            src["part"], src["talk"] = (k, len(spans)), video
        made.append(make(f"{name}_part{k}" if many else name, src, d, title))
        title = title or made[0][1]["name"]  # every part carries the first one's title
    return made


# ---------- AI story ----------
def story(job, topic=None, script=None):
    if script:
        text, title = Path(script).read_text(encoding="utf-8"), ""
    elif topic and (r := llm(STORY_PROMPT.format(topic=topic), temperature=0.8)) and r.get("script"):
        text, title = r["script"], r.get("title", "")
    else:
        print("no topic/LLM -> demo script")
        text, title = DEMO_SCRIPT, ""
    words = tts(text, job / "voice.mp3")
    dur = words[-1][1] + 0.6
    brolls = [f for f in (LIB / "broll").glob("*") if f.suffix.lower() in VID_EXT]
    cmd, parts, seg = ["ffmpeg", "-y"], [], 4.0  # a new b-roll shot every 4s
    n = math.ceil(dur / seg)
    for i in range(n):
        if brolls:
            f = brolls[i % len(brolls)]
            cmd += ["-stream_loop", "-1", "-ss", f"{random.uniform(0, max(0.0, probe(f)[0] - seg)):.2f}", "-i", f]
        else:
            cmd += ["-f", "lavfi", "-i", f"gradients=s={W}x{H}:d={seg}:speed=0.02:n=3:seed={i}"]
        parts.append(f"[{i}:v]trim=0:{seg},setpts=PTS-STARTPTS,{COVER}[s{i}]")
    fc = ";".join(parts) + ";" + "".join(f"[s{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    run(cmd + ["-i", job / "voice.mp3", "-filter_complex", fc, "-map", "[v]", "-map", f"{n}:a", "-t", f"{dur:.2f}",
               *X264, *AUDIO, job / "source.mp4"])
    return dict(video=job / "source.mp4", words=words, dur=dur, title=title, credits=[], kind="story",
                problems=[] if any(not f.name.startswith("_demo_") for f in brolls) else
                ["only placeholder backgrounds: put real footage you may use in library/broll"])


# ---------- clips from your files / links ----------
def pick_segments(trans, durs, topic):
    lines = "\n".join(seg_lines(t, i) for i, t in enumerate(trans))
    r = llm(SEGMENT_PROMPT.format(n=len(trans), topic=f"Topic: {topic}\n" if topic else "", lines=lines[:14000]))
    picks = []
    for p in (r or {}).get("segments", []):
        try:
            s, a, b = int(p["src"]), float(p["start"]), float(p["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= s < len(trans) and trans[s]:
            a, b = snap_to_segments(trans[s], a, min(b, durs[s]))
            if b - a >= 2:
                picks.append((s, a, b))
    if not picks:  # no LLM: equal share of every source, skipping its (usually cut-off) first segment
        for i, t in enumerate(trans):
            t = t[1:] if len(t) > 2 else t
            if t:
                a = t[0]["start"]
                picks.append((i, a, next((s["end"] for s in t if s["end"] - a >= 45 / len(trans)), t[-1]["end"])))
    out, total = [], 0.0
    for s, a, b in picks:  # cap at 58s
        b = min(b, a + 50 - total)
        if b - a >= 1:
            out.append((s, a, b))
            total += b - a
    return out


def fetch(job, srcs):
    files = []
    for i, s in enumerate(srcs):
        if re.match(r"https?://", s):
            f = job / f"src{i}.mp4"
            if not f.exists():
                run(["yt-dlp", "--no-warnings", "-f", "bv*[height<=1080]+ba/b", "--merge-output-format", "mp4", "-o", f, s])
            files.append((f, s))
        else:
            files.append((Path(s), Path(s).name))
    return files


def own_clips(name, job, srcs, topic, make):
    """Your file(s): one long clip without a topic becomes a series, everything else one video. -> made videos"""
    files = fetch(job, srcs)
    if len(files) == 1 and not topic and 50 < probe(files[0][0])[0] <= LONGEST:
        trans = transcribe(files[0][0])
        if spans := trans and split_parts(trans, max(0.0, trans[0]["start"] - 0.15), min(probe(files[0][0])[0], trans[-1]["end"] + 2.0)):
            return series(name, job, files[0][0], spans, make, credits=[files[0][1]])
    return [make(name, clips(job, srcs, topic), job)]


def clips(job, srcs, topic=None, span=None):
    """srcs: files or URLs. span: exactly this part of the first one, nothing is picked."""
    files = fetch(job, srcs)
    trans = [transcribe(f) for f, _ in files]
    durs = [probe(f)[0] for f, _ in files]
    picks = [(0, *span)] if span else [(i, 0.0, d) for i, d in enumerate(durs)] if sum(durs) <= 50 else pick_segments(trans, durs, topic)
    words, t, listing = [], 0.0, []
    made, sig = job / "picks.json", json.dumps([[str(f), f.stat().st_mtime] for f, _ in files] + [list(p) for p in picks])
    again = made.exists() and (job / "source.mp4").exists() and made.read_text(encoding="utf-8") == sig  # what was heard stays valid
    for k, (si, a, b) in enumerate(picks):
        part = job / f"part{k}.mp4"
        vf = NORM
        if len(picks) == 1:  # only picture: no black bars, none of the source's own subtitles below it
            (_, _, w, h), low, box = probe(files[si][0]), floor(files[si][0]), crop_box(files[si][0])
            cw, ch, x, y = map(int, (box or f"{w}:{h}:0:0").split(":"))
            ch = min(ch, int(h * low) - y) // 2 * 2
            vf = f"crop={cw // 2 * 2}:{ch}:{x}:{y},scale=1920:-2:flags=lanczos,setsar=1,fps={FPS}" if low >= LOWEST and cw > ch else NORM
        if not again:
            run(["ffmpeg", "-y", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", files[si][0], "-vf", vf,
                 "-map", "0:v:0", "-map", "0:a:0", *X264, *AUDIO, part])
        words += [[w0 - a + t, w1 - a + t, txt] for seg in trans[si] for w0, w1, txt in seg["words"] if a <= w0 < b]
        t += probe(part)[0]
        listing.append(f"file '{part.name}'")
    (job / "parts.txt").write_text("\n".join(listing), encoding="utf-8")
    if not again:
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "parts.txt", "-c", "copy", "source.mp4"], cwd=job)
        made.write_text(sig, encoding="utf-8")
    return dict(video=job / "source.mp4", words=words, dur=t, title="", credits=[s for _, s in files], kind="clip")


# ---------- auto: find videos, pick their best self-contained story ----------
RISKY = ["song", "trailer", "full movie", "movie", "anime", "highlights", "ipl", "gaana", "lyrics", "cartoon", "web series"]


def ytjson(*args):
    return json.loads(run(["yt-dlp", "-J", "--no-warnings", *args]))


def discover(sources, done, per_source=25):
    """Search queries / channel URLs -> unseen, non-risky 8min-4h videos: the most viewed of each source, then the second..."""
    pools, seen = [], set(done)
    for s in sources:
        url = s if s.startswith("http") else f"ytsearch{per_source}:{s}"
        if re.search(r"youtube\.com/(@|channel/)", url) and not url.rstrip("/").endswith("/videos"):
            url = url.rstrip("/") + "/videos"
        try:
            entries = ytjson("--flat-playlist", "--playlist-end", str(per_source), url).get("entries") or []
        except RuntimeError as e:
            print("search failed:", s, str(e)[-200:])
            continue
        ok = [e for e in entries if e.get("id") and e["id"] not in seen and 480 <= (e.get("duration") or 0) <= 4 * 3600
              and not any(r in (e.get("title") or "").lower() for r in RISKY)]
        seen |= {e["id"] for e in ok}
        pools.append(sorted(ok, key=lambda e: e.get("view_count") or 0, reverse=True))
    return [p[k] for k in range(max(map(len, pools), default=0)) for p in pools if k < len(p)]


def best_window(info, before=150, after=210):
    """Around YouTube's 'most replayed' peak (intro/outro ignored) -> (start, end, peak) to listen to, or None
    (video too short)."""
    dur = info.get("duration") or 0
    hm = [h for h in info.get("heatmap") or [] if 60 <= h["start_time"] <= dur - 40]
    if not hm:  # new or unpopular: the six minutes after the intro; the story score still decides
        a = min(60.0, dur * 0.1)
        return (a, min(dur, a + before + after), a + before) if dur >= 120 else None
    peak = max(hm, key=lambda h: h["value"])
    mid = (peak["start_time"] + peak["end_time"]) / 2
    return max(0.0, mid - before), min(dur, mid + after), mid


def pick_story(trans, title, center):
    """Best self-contained story in a transcript window -> {start, end, score, summary, title} (seconds in the window)."""
    r = llm(STORY_PICK_PROMPT.format(title=title, lines=seg_lines(trans)[:14000]))
    try:
        a, b = snap_to_segments(trans, float(r["start"]), float(r["end"]))
        long = fit_length(trans, a, b, 62, LONGEST) if 55 < b - a <= LONGEST else (0, 0)
        if 62 <= long[1] - long[0] <= LONGEST:
            return {**r, "start": long[0], "end": long[1], "score": float(r.get("score") or 0)}
        a, b = fit_length(trans, a, b)
        if 30 <= b - a <= 50:
            return {**r, "start": a, "end": b, "score": float(r.get("score") or 0)}
    except (TypeError, KeyError, ValueError):
        pass
    if r is not None:
        return None
    # no LLM: the ~45s around the replay peak, on sentence boundaries
    segs = [s for s in trans if s["start"] >= center - 25] or trans
    a = segs[0]["start"]
    a, b = fit_length(trans, a, next((s["end"] for s in segs if s["end"] - a >= 40), segs[-1]["end"]))
    return {"start": a, "end": b, "score": 7, "summary": "replay peak", "title": ""}


def refuse(section):
    """Why a downloaded story cannot be used ('' = fine): music behind the talk, or subtitles of its own that cropping cannot lose."""
    from .rights import has_music
    if has_music(section):
        return "has background music"
    return "has its own subtitles in the middle of the picture" if floor(section) < LOWEST else ""


LOWEST = 0.72  # of the picture's height: cropping higher than this leaves too little of it


def floor(section, n=14):
    """Where the source's own subtitles begin (fraction of the height; 1 = it has none). Everything below is cropped off."""
    from . import eyes
    cache = Path(f"{section}.floor.json")
    if cache.exists() and cache.stat().st_mtime >= Path(section).stat().st_mtime:
        return json.loads(cache.read_text())
    dur = probe(section)[0]
    fr = [eyes.grab_frame(section, dur * (k + 0.5) / n) for k in range(n)]
    tops = [min(b[1] for b in s) for f in fr if f is not None and (s := eyes.subtitles(eyes.texts(f)))]
    at = round(float(min(tops)) - 0.03, 3) if len(tops) >= 2 else 1.0
    cache.write_text(json.dumps(at))
    return at


def auto_stories(sources, count, make, min_score=7, max_tries=8):
    """For each candidate video: listen around its replay peak, keep it only if its best story scores >= min_score, download
    just that story, refuse it if it has music or its own subtitles, else make(name, src, job, title). -> made videos."""
    donef = JOBS / "done.txt"
    donef.parent.mkdir(exist_ok=True)
    done = set(donef.read_text().split()) if donef.exists() else set()
    made, tried = [], 0
    for e in discover(sources, done):
        if sum(not p for _, _, p in made) >= count or tried >= count * max_tries:
            break
        tried += 1
        vid, url = e["id"], f"https://www.youtube.com/watch?v={e['id']}"
        try:  # YouTubeBlocked is not a RuntimeError: it ends the run and the video stays unmarked
            info = ytjson("--skip-download", url)
            win = best_window(info)
            if not win:
                print("  too short, skipping:", info.get("title"))
            else:
                job = JOBS / f"auto_{vid}"
                job.mkdir(parents=True, exist_ok=True)
                wide = job / "wide.mp3"
                if not wide.exists():  # audio only: cheap to download, enough to judge the story
                    run(["yt-dlp", "--no-warnings", "-f", "ba", "-x", "--audio-format", "mp3", "--download-sections",
                         f"*{win[0]:.0f}-{win[1]:.0f}", "-o", job / "wide.%(ext)s", url])
                st = pick_story(transcribe(wide), info.get("title", ""), center=win[2] - win[0])
                if not st or st["score"] < min_score:
                    print(f"  skip '{info.get('title')}': best story scored {st and st['score']} ({st and st.get('summary')})")
                else:
                    a, b = win[0] + st["start"] - 0.3, win[0] + st["end"] + 2.0  # with the laugh after the last line
                    print(f"  story {st['score']}/10 in '{info.get('title')}' at {a:.0f}-{b:.0f}s: {st.get('summary')}")
                    section = job / f"story_{a:.0f}_{b:.0f}.mp4"
                    if not section.exists():
                        run(["yt-dlp", "--no-warnings", "-f", "bv*[height<=1080]+ba/b", "--download-sections", f"*{a:.2f}-{b:.2f}",
                             "--force-keyframes-at-cuts", "--merge-output-format", "mp4", "-o", section, url])
                    why = refuse(section)
                    if why:
                        print(f"  refused: {why}")
                    else:
                        dur = probe(section)[0]
                        spans = [(0.0, dur)] if st["end"] - st["start"] <= 50 else split_parts(transcribe(section), 0.0, dur)
                        if not spans:
                            print("  long story, but it has no good places to split")
                        made += series(f"auto_{vid}", job, section, spans, make, st.get("title", ""), [f"{info.get('channel')} - {url}"],
                                       until=st["end"] - st["start"] + 0.45)
        except RuntimeError as err:
            print("  skipped", vid, str(err)[-200:].replace("\n", " "))
        with open(donef, "a") as f:
            f.write(vid + "\n")
    if not tried:
        print("no unseen video of 8 minutes to 4 hours found: try another creator")
    print(f"auto: {sum(not p for _, _, p in made)} passed QA out of {len(made)} made, {tried} video(s) tried")
    return made
