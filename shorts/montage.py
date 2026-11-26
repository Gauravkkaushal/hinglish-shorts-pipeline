"""Music edit: a topic -> the most-replayed moments of popular videos -> shots scored for motion/colour/detail and checked
by the vision model -> cut on the beat of an ORIGINAL (or licence-filed) track. No shot is ever shown twice."""
import json, math, shutil, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import eyes, qa, rights
from .ai import clean_word, llm, look
from .captions import HEAD, ts
from .creators import about
from .ff import LOUD, YouTubeBlocked, FPS, H, LIB, OUT, W, crop_box, encode, env, fit, run, safe_name
from .sources import ytjson

MIN_VIEWS = 5000  # below that it is usually a re-upload
NO = ["reaction", "live", "stream", "podcast", "explained", "theory", "news", "#shorts", "dub", "reupload", "re-upload",
      "leak", " edit", "montage", "phonk", "bodycam", "body cam", "dashcam", "dash cam", "caught on camera", "arrest",
      "mod ", " mod", "fan made", "fanmade", "concept"]  # talking heads, others' edits, real police footage, fakes
FALLBACK = ["{topic} official trailer", "{topic} cinematic 4k", "{topic} gameplay no commentary 4k"]
SEARCH_PROMPT = """Write 3 YouTube searches that find clean, cinematic, high-quality {topic} footage (each must contain
"{topic}"; no vlogs, reviews, reactions or other people's edits). Cover three different moods of {topic}.
Return JSON {{"queries": ["...", "...", "..."]}}"""
TEXT_PROMPT = """Topic: {topic}. An Instagram Reel shows these shots, in order: {shots}.
Write for young Indian viewers (simple English or natural Hinglish, Roman script). Only say things that fit what is shown;
no invented facts, no city or person names that are not in the shots:
- "hook": one complete line, max 7 words: a feeling or a funny everyday situation a young Indian fan knows, NOT a list of
  what is on screen. Good: "Bas 5 minute khelunga, mummy" / "Is game ke liye PC bechna padega". No hashtags, no emojis.
- "end": max 5 words, makes people send it to a friend (e.g. "Send this to your gaming buddy").
- "caption": 2 short lines, the second a question that gets comments, 1-2 emojis.
- "hashtags": exactly 5 specific hashtags without #.
Return JSON with exactly those keys."""
BOX_H, RAMP = 608, 1.6
FONTS = LIB.parent / "assets" / "fonts"
ACCENT = "FF1F3D"


# ---------- clips ----------
def topic_word(topic):
    """First word, singular: "Supercars" must match a title saying "supercar"."""
    w = topic.lower().split()[0]
    return w[:-1] if len(w) > 4 and w.endswith("s") else w


def searches(topic):
    r = llm(SEARCH_PROMPT.format(topic=topic), effort="low") or {}
    qs = [q for q in r.get("queries") or [] if isinstance(q, str) and topic_word(topic) in q.lower()]
    return [q.replace("{", "").replace("}", "") for q in qs[:3]] + [FALLBACK[0]]


def find_videos(topic, n, queries, per_query=15):
    """The most-viewed usable video of each search, round-robin."""
    pools = []
    for q in queries:
        try:
            where = q if q.startswith("http") else f"ytsearch{per_query}:{q.format(topic=topic)}"
            entries = ytjson("--flat-playlist", "--playlist-end", str(per_query), where).get("entries") or []
        except RuntimeError as e:
            print("  search failed:", q, str(e)[-120:])
            continue
        ok = [e for e in entries if e.get("id") and 30 <= (e.get("duration") or 0) <= 2400
              and (e.get("view_count") or 0) >= MIN_VIEWS
              and topic_word(topic) in (e.get("title") or "").lower()
              and not any(x in (e.get("title") or "").lower() for x in NO)]
        pools.append(sorted(ok, key=lambda e: e.get("view_count") or 0, reverse=True))
    return pick_round_robin(pools, n)


def pick_round_robin(pools, n):
    out, seen = [], set()
    for rank in range(max((len(p) for p in pools), default=0)):
        for pool in pools:
            if rank < len(pool) and pool[rank]["id"] not in seen and len(out) < n:
                out.append(pool[rank])
                seen.add(pool[rank]["id"])
    return out


def peak_windows(info, k=3, length=10.0, min_dist=25.0):
    """Top-k 'most replayed' moments (or evenly spaced ones if YouTube has no heatmap) -> [(start, end)]."""
    dur, picks = info.get("duration") or 0, []
    for h in sorted(info.get("heatmap") or [], key=lambda h: -h["value"]):
        mid = (h["start_time"] + h["end_time"]) / 2
        if dur * 0.03 < mid < dur - 8 and all(abs(mid - p) >= min_dist for p in picks):
            picks.append(mid)
        if len(picks) == k:
            break
    picks = picks or [dur * f for f in np.linspace(0.25, 0.75, k)]
    return [(max(0.0, m - length / 2), min(dur, m + length / 2)) for m in sorted(picks)]


def snapshot(video, t, out, width=512):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", "-vf", f"scale={width}:-2",
                    str(out)], check=False)
    return out


def window(job, vid, i, a, b):
    """One replay-peak window (30fps / 1080p, black bars off, no sound)."""
    raw, norm = job / f"raw_{vid}_{i}.mp4", job / f"win_{vid}_{i}.mp4"
    if not norm.exists():
        run(["yt-dlp", "--no-warnings", "-f", "bv*[height<=1080]/b", "--download-sections", f"*{a:.1f}-{b:.1f}",
             "--force-keyframes-at-cuts", "--merge-output-format", "mp4", "-o", raw, f"https://www.youtube.com/watch?v={vid}"])
        box = crop_box(raw)
        tmp = norm.with_suffix(".tmp.mp4")
        run(["ffmpeg", "-y", "-i", raw, "-an", "-vf", (f"crop={box}," if box else "") + f"scale=-2:1080,setsar=1,fps={FPS}",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", tmp])
        tmp.replace(norm)  # an interrupted run never leaves half a window behind
        raw.unlink(missing_ok=True)
    return norm


def grab(topic, job, n_videos, queries, k=4):
    """Windows of n_videos clean videos. The first window decides quickly; a video is dropped as soon as one window shows a
    facecam or watermark: it will be there in all of it."""
    files, credits, titles = [], [], []
    known_f = job / "videos.json"  # {video id: [credit, title]} of the clean ones
    known = json.loads(known_f.read_text(encoding="utf-8")) if known_f.exists() else {}
    for e in find_videos(topic, 3 * n_videos, queries):
        if len(credits) >= n_videos:
            break
        url, mine, why = f"https://www.youtube.com/watch?v={e['id']}", [], []
        try:
            info = ytjson("--skip-download", url)
            spans = list(enumerate(peak_windows(info, k)))
            for part in (spans[:1], spans[1:]):
                with ThreadPoolExecutor(2) as pool:
                    mine += pool.map(lambda w: window(job, e["id"], w[0], *w[1]), part)
                why = [x for f in mine for x in eyes.window_overlays(f)]
                if why:
                    print(f"  not using ({', '.join(sorted(set(why)))}): {info.get('title')}")
                    mine = []
                    break
        except RuntimeError as err:
            print("  skipped", e["id"], " ".join(str(err)[-160:].split()))
            mine = []
        if mine:
            files += mine
            credits.append(f"{info.get('channel')} - {url}")
            titles.append(str(info.get("title")))
            known[e["id"]] = [credits[-1], titles[-1]]
            known_f.write_text(json.dumps(known, ensure_ascii=False), encoding="utf-8")
            print(f"  clips from: {info.get('title')} ({info.get('view_count')} views)")
    return files, credits, titles


def on_disk(job):
    """The clean footage earlier runs of this job already downloaded."""
    known = json.loads((job / "videos.json").read_text(encoding="utf-8")) if (job / "videos.json").exists() else {}
    files = [f for vid in known for f in sorted(job.glob(f"win_{vid}_?.mp4"))]
    return files, [c for c, _ in known.values()], [t for _, t in known.values()]


# ---------- shots ----------
def analyze(path):
    """Per-frame stats: (frame diff, horizontal motion centre 0-1, brightness, saturation, edge density). Cached."""
    import cv2
    cache = Path(f"{path}.stats.npy")
    if cache.exists():
        return np.load(cache)
    cap, prev, rows = cv2.VideoCapture(str(path)), None, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (160, max(1, int(160 * h / w))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        d = np.abs(gray - prev) if prev is not None else np.zeros_like(gray)
        col = d.sum(0)
        cx = float((col * np.arange(col.size)).sum() / col.sum() / col.size) if col.sum() > 0 else 0.5
        edges = float((cv2.Canny(gray.astype(np.uint8), 80, 160) > 0).mean())
        rows.append((float(d.mean()), cx, float(gray.mean()), float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1].mean()), edges))
        prev = gray
    cap.release()
    stats = np.array(rows) if rows else np.zeros((0, 5))
    np.save(cache, stats)
    return stats


def split_shots(stats, src, size=36, min_frames=36):
    """Frame stats -> independent shots [{src, a, b, score, cx}]: split at hard cuts, long takes into pieces of ~`size`
    frames with a gap between them, so two pieces of one take never look like the same shot again."""
    if len(stats) < min_frames:
        return []
    diff = stats[:, 0]
    thr = max(25.0, 4 * float(np.median(diff[1:])))
    bounds = [0] + [i for i in range(1, len(diff)) if diff[i] > thr] + [len(diff)]
    shots = []
    for a, b in zip(bounds, bounds[1:]):
        a, b = a + 2, b - 2  # stay clear of the cut itself
        while b - a >= min_frames:
            e = min(b, a + size) if b - a - size >= 0 else b
            s = stats[a:e]
            motion, bright, sat, edges = (float(s[:, c].mean()) for c in (0, 2, 3, 4))
            score = min(motion, 12) / 12 * 0.5 + sat / 255 * 0.25 + min(edges, 0.15) / 0.15 * 0.25
            score *= (0.1 if bright < 35 else 1) * (0.3 if motion < 0.5 else 1) * (0.2 if edges < 0.03 else 1)
            shots.append({"src": src, "a": a, "b": e, "score": round(score, 3), "cx": float(np.median(s[:, 1]))})
            a = e + size  # skip a stretch before the next piece of the same take
    return shots


def cut_points(beats, length, step=2):
    pts = [0.0] + [b for b in beats[::step] if 0.3 < b < length - 0.3] + [length]
    return [f / FPS for f in sorted({round(p * FPS) for p in pts})]


def assign(shots, slots, hits=()):
    """One shot per slot, never reused, never two neighbours from one source window; slots that start on a hard beat choose
    first and play faster when the shot is long enough. Raises when there are too few shots."""
    if len(shots) < len(slots):
        raise RuntimeError(f"only {len(shots)} usable shots for {len(slots)} cuts")
    free, out = sorted(shots, key=lambda s: -s["score"]), [None] * len(slots)
    for i in sorted(range(len(slots)), key=lambda i: (i not in hits, i)):
        n = max(1, round(slots[i] * FPS))
        near = {p["src"] for p in (out[i - 1] if i else None, out[i + 1] if i + 1 < len(out) else None) if p}
        pool = [s for s in free if s["b"] - s["a"] >= n] or free
        s = next((s for s in pool if s["src"] not in near), pool[0])
        free.remove(s)
        m = min(round(n * RAMP), s["b"] - s["a"]) if i in hits else min(n, s["b"] - s["a"])
        out[i] = {"src": s["src"], "a": s["a"], "n": n, "m": m, "cx": s["cx"], "shot": s}
    return out


def signature(img):
    """Small colour histogram + small grey picture: two shots of one boat ride come out nearly equal."""
    import cv2
    hist = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    grey = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (16, 9)).ravel().astype(np.float32)
    return np.sqrt(hist / max(hist.sum(), 1)), (grey - grey.mean()) / max(np.linalg.norm(grey - grey.mean()), 1)


def alike(p, q):
    return float(p[0] @ q[0]) > 0.85 or float(p[1] @ q[1]) > 0.75


def vet(shots, files, job, need, twins=0):
    """Best first until `need` clean, different shots are found: no subtitles, menus or title cards in three frames of the
    shot, at most `twins` earlier shots that look like it, at most a third from one video. Verdicts are cached."""
    cache_f = job / "look.json"
    cache = json.loads(cache_f.read_text(encoding="utf-8")) if cache_f.exists() else {}
    good, sigs, per = [], [], {}
    for s in sorted(shots, key=lambda s: -s["score"]):
        if len(good) >= need:
            break
        vid = files[s["src"]].name.rsplit("_", 1)[0]
        if per.get(vid, 0) >= math.ceil(need / 3):
            continue
        key = f"{files[s['src']].name}:{s['a']}"
        if key not in cache:
            why = []
            for q in (0.1, 0.5, 0.9):
                img = eyes.grab_frame(files[s["src"]], (s["a"] + q * (s["b"] - s["a"])) / FPS)
                why += ["unreadable"] if img is None else eyes.facts(img)
            cache[key] = sorted(set(why))
            cache_f.write_text(json.dumps(cache), encoding="utf-8")
        if cache[key]:
            continue
        sig = signature(eyes.grab_frame(files[s["src"]], (s["a"] + s["b"]) / 2 / FPS, 320))
        if sum(alike(sig, t) for t in sigs) <= twins:
            good.append(s)
            sigs.append(sig)
            per[vid] = per.get(vid, 0) + 1
    return good


def sheet(files, pieces, out, cols=5):
    """Contact sheet of the chosen shots, in order: one picture for the vision model instead of thirty."""
    from PIL import Image
    tiles = [eyes.grab_frame(files[p["src"]], (p["a"] + p["m"] / 2) / FPS, 256) for p in pieces[:20]]
    tiles = [Image.fromarray(t[..., ::-1]).resize((256, 144)) for t in tiles if t is not None]
    img = Image.new("RGB", (256 * cols, 144 * math.ceil(len(tiles) / cols)))
    for k, t in enumerate(tiles):
        img.paste(t, (256 * (k % cols), 144 * (k // cols)))
    img.save(out)
    return out


# ---------- edit ----------
def render(files, pieces, job, beats):
    """Each piece on its own (a 16:9 box on a blurred copy of itself, slow zoom drift, a small punch on every beat), joined
    without re-encoding: one big graph runs a 7GB laptop out of memory."""
    pdir = job / "pieces"
    shutil.rmtree(pdir, ignore_errors=True)
    pdir.mkdir()
    listing, t0 = [], 0.0
    for k, p in enumerate(pieces):
        hits = [(b + 0.05 - t0) * FPS for b in beats if t0 <= b < t0 + p["n"] / FPS]
        bounce = "+".join(f"max(0,1-abs(on-{f:.1f})/3)" for f in hits) or "0"
        drift = f"1+0.10*on/{p['n']}" if k % 2 == 0 else f"1.10-0.10*on/{p['n']}"
        t0 += p["n"] / FPS
        vf = (f"trim=end_frame={p['m']},setpts=(PTS-STARTPTS)*{p['n'] / p['m']:.5f},fps={FPS},tpad=stop_mode=clone:stop={p['n']},"
              f"trim=end_frame={p['n']},scale={W}:{BOX_H}:force_original_aspect_ratio=increase,crop={W}:{BOX_H},"
              f"zoompan=z='{drift}+0.05*min(1,{bounce})':d=1:x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':s={W}x{BOX_H}:fps={FPS},"
              f"{fit(k)},setsar=1,format=yuv420p")
        out = pdir / f"p{k:03d}.mp4"
        run(["ffmpeg", "-y", "-ss", f"{p['a'] / FPS:.4f}", "-i", files[p["src"]], "-an", "-filter_complex", f"[0:v]{vf}[v]",
             "-map", "[v]", "-frames:v", p["n"], "-r", FPS, "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-bf", "0", out])
        listing.append(f"file 'pieces/{out.name}'")
    (job / "pieces.txt").write_text("\n".join(listing), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "pieces.txt", "-c", "copy", "cut.mp4"], cwd=job)


def bgr(hex_rgb):
    """'FF3EA5' -> ASS colour '&H00A53EFF&' (ASS is BGR)."""
    return f"&H00{hex_rgb[4:6]}{hex_rgb[2:4]}{hex_rgb[0:2]}&"


def text_ass(path, hook, end, length, accent=ACCENT):
    """Hook above the video box (word by word within 1.2s, then held), end line below it; nothing ever covers the video."""
    acc, words = bgr(accent), hook.upper().split()
    style = (f"Style: Hook,Bebas Neue,132,&H00FFFFFF,&H000000FF,&H00000000,&HAA000000,0,0,0,0,100,100,3,0,1,6,3,2,80,80,1330,1\n"
             f"Style: End,Bebas Neue,124,{acc},&H000000FF,&H00000000,&HAA000000,0,0,0,0,100,100,3,0,1,6,3,8,80,80,1330,1\n")
    step = int(1200 / max(1, len(words)))
    reveal = " ".join(f"{{\\alpha&HFF&\\t({i * step},{i * step + 60},\\alpha&H00&)"
                      + (f"\\1c{acc}" if i == len(words) - 1 else "") + f"}}{w}" for i, w in enumerate(words))
    rows = [f"Dialogue: 0,{ts(0)},{ts(3.4)},Hook,,0,0,0,,{reveal}",
            f"Dialogue: 0,{ts(length - 2.4)},{ts(length)},End,,0,0,0,,{{\\fscx130\\fscy130\\t(0,110,\\fscx100\\fscy100)}}{end.upper()}"]
    path.write_text(HEAD.replace("[Events]", style + "\n[Events]", 1) + "\n".join(rows) + "\n", encoding="utf-8")


def finish(job, track, plan, length, hook, end, out):
    text_ass(job / "text.ass", hook, end, length, env("ACCENT_COLOR", ACCENT).lstrip("#"))
    (job / "fonts").mkdir(exist_ok=True)
    for f in FONTS.glob("*.ttf"):
        shutil.copy(f, job / "fonts" / f.name)
    hits = plan["strong"]
    gate = lambda d: "+".join(f"between(t,{b:.3f},{b + d:.3f})" for b in hits) or "0"
    vf = (f"eq=brightness='0.22*min(1,{gate(0.07)})':eval=frame,rgbashift=rh=-12:bh=12:enable='{gate(0.10)}',"
          f"scale={round(W * 1.05 / 2) * 2}:{round(H * 1.05 / 2) * 2},"
          f"crop={W}:{H}:x='(iw-ow)/2+14*min(1,{gate(0.2)})*sin(t*90)':y='(ih-oh)/2+10*min(1,{gate(0.2)})*cos(t*70)',"
          f"ass=text.ass:fontsdir=fonts,format=yuv420p")
    encode(["ffmpeg", "-y", "-i", "cut.mp4", "-ss", f"{plan['start']:.3f}", "-i", Path(track).resolve(), "-vf", vf, "-map", "0:v", "-map", "1:a",
            "-af", LOUD, "-t", f"{length:.3f}", "-r", FPS, "-c:a", "aac", "-b:a", "256k",
            "-movflags", "+faststart"], out.resolve(), cwd=job)


# ---------- music ----------
def track_plan(track, length):
    """Beats of a licence-filed track: loudest stretch, hard-hitting beats marked."""
    import librosa
    lic = rights.licence_of(track)
    y, sr = librosa.load(str(track), sr=22050, mono=True)
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr)
    beats, rms = librosa.frames_to_time(frames, sr=sr), librosa.feature.rms(y=y)[0]
    t = librosa.times_like(rms, sr=sr)
    start = max((b for b in beats if b + length <= t[-1]), key=lambda b: rms[(t >= b) & (t < b + length)].mean(), default=0.0)
    inside = (beats >= start) & (beats < start + length)
    rel, strength = beats[inside] - start, onset[frames[inside]]
    strong = rel[strength >= np.quantile(strength, 0.8)] if len(strength) else rel
    return {"start": float(start), "beats": rel.tolist(), "strong": strong.tolist(), "tempo": float(np.atleast_1d(tempo)[0]), "licence": lic}


USED = OUT / "used_shots.json"  # [window file, first frame, last frame, edit]


def load_used():
    return json.loads(USED.read_text(encoding="utf-8")) if USED.exists() else []


def was_used(used, name, a, b):
    return any(u[0] == name and u[1] < b and a < u[2] for u in used)


def montage(topic, job, track=None, length=32.0, n_videos=7, creators=(), only=False):
    """creators: channel addresses to take the footage from; only: from nobody else. track: a file in library/music/licensed (needs its licence record); default: an original synthesised beat."""
    seed = len(list(OUT.glob(f"montage_{safe_name(topic, 20)}_*.mp4")))
    if track:
        plan = track_plan(track, length)
    else:
        track = job / f"beat_{seed}.wav"
        plan = rights.original_beat(track, length, bpm=[100, 92, 108, 116][seed % 4], seed=seed + sum(map(ord, topic)))
    pts = cut_points(plan["beats"], length)
    slots = [b - a for a, b in zip(pts, pts[1:])]
    remembered = job / "searches.json"  # an interrupted run picks up the same downloads
    queries = [about(c, topic) for c in creators]
    if not queries:
        queries = json.loads(remembered.read_text(encoding="utf-8")) if remembered.exists() else searches(topic)
        remembered.write_text(json.dumps(queries), encoding="utf-8")
    print("  searches:", queries)
    try:
        files, credits, titles = grab(topic, job, n_videos, queries)
    except YouTubeBlocked:
        files, credits, titles = on_disk(job)
        if not files:
            raise
        print(f"  YouTube is refusing downloads now: using the {len(files)} clean windows already here")
    size = math.ceil(max(slots) * FPS * RAMP)
    pool = lambda: [s for i, f in enumerate(files) for s in split_shots(analyze(f), i, size, math.ceil(max(slots) * FPS))]
    used, shots = load_used(), pool()
    shots = [s for s in shots if not was_used(used, files[s["src"]].name, s["a"], s["b"])]
    good = vet(shots, files, job, len(slots))
    if len(good) < len(slots):
        print(f"  only {len(good)}/{len(slots)} clean shots, searching wider")
        try:
            more, c2, t2 = grab(topic, job, n_videos + 3, queries if only else searches(topic) + FALLBACK)
        except YouTubeBlocked:
            more, c2, t2 = on_disk(job)
        files += [f for f in more if f not in files]
        credits, titles = list(dict.fromkeys(credits + c2)), titles + t2
        shots = [s for s in pool() if not was_used(used, files[s["src"]].name, s["a"], s["b"])]
        good = vet(shots, files, job, len(slots))
    for step in (3, 4):  # not enough different shots: slower cuts rather than repeats
        if len(good) < len(slots):
            pts = cut_points(plan["beats"], length, step)
            slots = [b - a for a, b in zip(pts, pts[1:])]
    good = good[:len(slots)]
    hits = {i for i, a in enumerate(pts[:-1]) if any(abs(a - h) < 0.06 for h in plan["strong"])}
    pieces = assign(good, slots, hits)
    print(f"  {len(files)} windows, {len(shots)} shots, {len(good)} clean, {plan['tempo']:.0f} BPM, {len(pieces)} cuts")
    render(files, pieces, job, plan["beats"])
    seen = look([sheet(files, pieces, job / "sheet.jpg")], '{"shots": "<what these shots show, in order, 40 words max>"}')
    shown = str(seen[0].get("shots") or "; ".join(titles))
    r = llm(TEXT_PROMPT.format(topic=topic, shots=shown)) or {}
    hook = " ".join(map(clean_word, (r.get("hook") or f"{topic} hits different").split()[:7]))
    end = r.get("end") or "Send this to a friend"
    out = OUT / f"montage_{safe_name(topic, 20)}_{seed + 1}.mp4"
    OUT.mkdir(exist_ok=True)
    finish(job, track, plan, length, hook, end, out)
    tags = [str(t).lstrip("#").replace(" ", "") for t in r.get("hashtags") or [topic.replace(" ", ""), "reels"]][:5]
    meta = {"title": hook, "caption": (r.get("caption") or hook) + "\n\n" + " ".join("#" + t for t in tags), "hashtags": tags,
            "credits": credits, "music": plan["licence"], "shows": shown}
    out, problems = qa.deliver(out, meta, "montage", [job / "text.ass"], music_ok=True, length=length)
    if not problems:
        USED.write_text(json.dumps(used + [[files[p["src"]].name, p["a"], p["a"] + p["m"], out.stem] for p in pieces]), encoding="utf-8")
    return out, meta, problems
