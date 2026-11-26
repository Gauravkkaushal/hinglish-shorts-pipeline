"""The edit: jump-cut dead air, cut memes in after punchlines, keep the speaker's face framed in 9:16, then finish
(punch zooms, progress bar, captions, whoosh, loudness). All cut points sit on the 30fps / 48kHz grid, so the timeline
math below matches the rendered video exactly and captions never drift."""
import subprocess

import numpy as np

from .captions import write_ass
from .ff import LOUD, AF, COVER, FPS, H, IMG_EXT, LIB, W, encode, fit, probe
from .rights import hear

SR = 48000  # every source is normalised to 30fps CFR video + 48kHz audio before it gets here


def snap(t):
    return round(t * FPS) / FPS


# ---------- timeline ----------
def listen(video):
    """-> (loudness in dB per 10ms, how surely somebody talks per 0.25s)"""
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True).stdout
    x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    x = np.sqrt((x[:len(x) // 160 * 160].reshape(-1, 160) ** 2).mean(1))
    return 20 * np.log10(np.convolve(x, np.ones(5) / 5, "same") + 1e-6), hear(video, {"Speech"}, 0.25)


def keep_ranges(words, dur, sound=None, pause=0.3, laugh=1.1, short=(), until=None):
    """What stays: [[a, b, fade]]. Silence beyond ~pause is cut out; the audience's reaction after a line stays for up to laugh
    seconds and is faded out (comic timing lives there); words and anything that sounds like unheard speech are never cut.
    short: word ends where a meme follows at once, so the laugh there is brief. until: nothing after this time."""
    end = min(dur, until or dur)
    words = [w for w in words if w[0] < end]
    if not words:
        return [[0.0, snap(end), 0.0]]
    if sound is None:  # nothing to listen to: every gap is silence
        sound = np.full(int(dur * 100) + 1, -60.0), [0.0] * (int(dur * 4) + 2)
    db, speech = sound
    level = np.percentile(np.concatenate([db[int(w[0] * 100):int(w[1] * 100) + 1] for w in words if int(w[0] * 100) < len(db)] or [db]), 60)
    loud = db > level - 18
    out = [[max(0.0, words[0][0] - 0.1), words[0][1], 0.0]]
    gaps = [(x[1], y[0]) for x, y in zip(words, words[1:])] + [(words[-1][1], end)]
    for k, (a, b) in enumerate(gaps):
        last = k == len(gaps) - 1
        if b - a > 0.02 or last:
            i, j = int(a * 100), max(int(a * 100), int(b * 100))
            noisy = loud[i:j]
            here = speech[int(a * 4):int(b * 4) + 1]
            stop, fade, back = a + pause / 2, 0.0, 0.12  # silence: just a breath after the word
            if noisy.size and noisy[:80].mean() > 0.4:
                if not last and np.mean(here) >= 0.7:  # talk the recogniser missed
                    stop = b
                else:
                    most = 0.35 if any(-0.05 < a - s < 0.6 for s in short) else laugh
                    ends = np.flatnonzero(noisy[:int(most * 100)])
                    stop = min(a + most, (i + ends[-1]) / 100 + 0.1)
                    fade = 0.3 if stop < b - 0.2 or last else 0.0
            elif noisy.size and noisy.any():  # the word rings on a little
                run = int(np.argmin(noisy)) if not noisy.all() else len(noisy)
                stop = max(stop, a + min(run / 100 + 0.08, 0.4))
            out[-1][1] = stop = max(out[-1][1], min(stop, b))
            while back < 0.35 and noisy.size > back * 100 and noisy[-int(back * 100)] and speech[min(round((b - back) * 4), len(speech) - 1)] >= 0.7:
                back += 0.02  # the recogniser often starts a word a little late
            if not last:
                start = max(b - back, stop)
                if start - stop > 0.2:
                    out[-1][2] = min(fade, stop - a) if fade else 0.02
                    out.append([start, b, 0.0])
        if not last:
            out[-1][1] = max(out[-1][1], words[k + 1][1])
    out[-1][1] = min(end, out[-1][1])
    out[-1][2] = min(0.3, max(0.02, out[-1][1] - words[-1][1]))
    return [[snap(a), snap(b), f] for a, b, f in out if snap(b) > snap(a)]


def build_pieces(ranges, cuts, tail=0.1):
    """Ranges + meme cuts ({t, path, d}) -> ordered pieces: {kind: main, a, b, fade} | {kind: meme, path, d}."""
    pieces = [{"kind": "main", "a": a, "b": b, "fade": f[0] if f else 0.02} for a, b, *f in ranges]
    for c in sorted(cuts, key=lambda c: c["t"]):
        t = c["t"]
        i = next((i for i, p in enumerate(pieces) if p["kind"] == "main" and p["a"] < t <= p["b"] + 1e-6), None)
        if i is None:  # punchline ended inside removed dead air: cut in at the end of the previous range
            i = max((i for i, p in enumerate(pieces) if p["kind"] == "main" and p["b"] < t), default=None)
            if i is None:
                continue
            t = pieces[i]["b"]
        p = pieces[i]
        cut = snap(min(t + tail, p["b"]))  # a little air after the last syllable
        new = [{"kind": "main", "a": p["a"], "b": cut, "fade": 0.05}, {"kind": "meme", "path": c["path"], "d": snap(c["d"])}]
        if p["b"] - cut > 0.05:
            new.append({"kind": "main", "a": cut, "b": p["b"], "fade": p["fade"]})
        pieces[i:i + 1] = new
    return [p for p in pieces if p["kind"] == "meme" or p["b"] - p["a"] >= 1 / FPS]


def length(p):
    return p["d"] if p["kind"] == "meme" else p["b"] - p["a"]


def out_time(t, pieces):
    """Time in the source -> time in the edited video (dead air removed, memes inserted)."""
    acc = 0.0
    for p in pieces:
        if p["kind"] == "main":
            if t < p["a"]:
                return acc
            if t <= p["b"]:
                return acc + t - p["a"]
        acc += length(p)
    return acc


def meme_starts(pieces):
    acc, out = 0.0, []
    for p in pieces:
        if p["kind"] == "meme":
            out.append(acc)
        acc += length(p)
    return out


# ---------- face-tracked framing ----------
def face_track(video, step=0.25):
    """-> ([(t, (cx, cy, h) of the main face as fractions of the frame, or None: no face or a wide two-shot)], [hard cut times])"""
    from .eyes import faces
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    every, out, diffs, prev, i = max(1, round(fps * step)), [], [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        diffs.append(0.0 if prev is None else float(np.abs(small - prev).mean()))
        prev = small
        if i % every == 0:
            found = [f for f in faces(cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))) if f[3] > 0.06]
            face = None
            if found:
                x, y, w, h = found[0]
                two_shot = len(found) > 1 and found[1][2] > 0.6 * w and abs(found[1][0] - x) > 0.25
                face = None if two_shot else (x + w / 2, y + h / 2, h)
            out.append((i / fps, face))
        i += 1
    cap.release()
    thr = max(18.0, 5 * float(np.median(diffs))) if diffs else 0
    return out, [snap(k / fps) for k, d in enumerate(diffs) if d > thr]


def camera(video, dur):
    """Framing per shot: [[a, b, path]], never carried over a hard cut; path None = show the whole picture."""
    samples, cuts = face_track(video)
    out, edges = [], [0.0, *[c for c in cuts if 0 < c < dur], snap(dur) + 1]
    for a, b in zip(edges, edges[1:]):
        spans = follow([s for s in samples if a <= s[0] < b], a, b)
        for s in spans:
            if out and s[2] is None and out[-1][2] is None:
                out[-1][1] = s[1]
            else:
                out.append(s)
    return [x for x in out if x[1] > x[0]]


def follow(samples, a, b, min_len=0.8, jump=0.15):
    """One shot's face samples -> [[a, b, path]]. path = (zoom, cy, [(t, cx)]): a smooth pan that keeps the face in the middle;
    a jump to another person starts a new framing, detector flicker does not."""
    if not samples:
        return [[a, b, None]]
    calm = []  # median of 5 neighbours, "no face" counted as a vote
    for i, (t, _) in enumerate(samples):
        w = min(2, i, len(samples) - 1 - i)  # even on both sides, or a moving speaker would be behind at the edges
        near = [f for _, f in samples[i - w:i + w + 1]]
        ok = [f for f in near if f]
        calm.append((t, tuple(np.median(ok, axis=0)) if len(ok) * 2 > len(near) else None))
    runs = []
    for t, f in calm:
        same = runs and (f is None) == (runs[-1][-1][1] is None) and (f is None or abs(f[0] - runs[-1][-1][1][0]) < jump)
        if same:
            runs[-1].append((t, f))
        else:
            runs.append([(t, f)])
    merged = []
    for r in runs:
        if merged and r[-1][0] - r[0][0] + 0.25 < min_len:
            merged[-1][1] = r[-1][0]
        else:
            merged.append([r, r[-1][0]])
    out = []
    for k, (r, _) in enumerate(merged):
        x0 = a if k == 0 else snap(r[0][0])
        x1 = b if k == len(merged) - 1 else snap(merged[k + 1][0][0][0])
        path = None
        if r[0][1] is not None:
            f = np.array([f for _, f in r])
            xs = np.array([f[max(0, i - 3):i + 4, 0].mean() for i in range(len(f))])  # ~1.5s of smoothing
            xs = np.clip(xs, f[:, 0] - 0.05, f[:, 0] + 0.05)  # but a quick step never leaves the speaker at the edge
            if xs.max() - xs.min() < 0.05:
                xs[:] = np.median(f[:, 0])
            zoom = float(np.clip(0.2 / np.median(f[:, 2]), 1.0, 1.35))  # a small face on a big stage: go closer
            path = zoom, float(np.median(f[:, 1])), [(t, float(x)) for (t, _), x in zip(r, xs)][::2]
        out.append([x0, x1, path])
    return out


def _shots(a, b, spans):
    if not spans:
        return [(a, b, "cover")]
    out = []
    for s0, s1, path in spans:
        lo, hi = max(a, s0), min(b, s1)
        if hi - lo < 1 / FPS:
            continue
        if out and hi - lo < 0.4:  # a sliver of a new shot: keep the previous framing
            out[-1] = (out[-1][0], hi, out[-1][2])
        else:
            out.append((lo, hi, path))
    return out or [(a, b, None)]


def pan(keys, t0, half):
    """[(t, cx)] -> ffmpeg expression in t (0 = t0) of the crop's left edge as a fraction of the width, inside the picture."""
    pts = [(t - t0, min(max(x, half), 1 - half) - half) for t, x in keys]
    before = [p for p in pts if p[0] <= 0][-1:]
    pts = before + [p for p in pts if p[0] > 0]
    if len({round(x, 3) for _, x in pts}) == 1:
        return f"{pts[0][1]:.4f}"
    e = f"{pts[-1][1]:.4f}"
    for (u, x), (v, y) in reversed(list(zip(pts, pts[1:]))):
        e = f"if(lt(t,{v:.2f}),{x:.4f}+({y - x:.4f})*clip((t-{u:.2f})/{v - u:.2f},0,1),{e})"
    return e


def _frame(path, k, sw, sh, t0=0.0):
    if path == "cover":
        return COVER
    if path is None:
        return fit(k)
    zoom, cy, keys = path
    ch = int(min(sh, max(sh / zoom, 700))) // 2 * 2  # fewer than 700 lines of picture would look soft
    cw = int(ch * 9 / 16) // 2 * 2
    y = int(min(max(cy * sh - 0.36 * ch, 0), sh - ch))
    return f"crop={cw}:{ch}:'iw*({pan(keys, t0, cw / sw / 2)})':{y},scale={W}:{H}:flags=lanczos,setsar=1"


# ---------- render ----------
def render_edit(video, pieces, spans, job):
    """One ffmpeg pass: every piece trimmed on exact frame/sample numbers, framed, concatenated -> jobs/<name>/edit.mp4"""
    _, _, sw, sh = probe(video)
    if sw * 16 <= sh * 9 + 16:  # already vertical
        spans = None
    inputs, fc, segs, k, jump, flash, shown = ["-i", video], [], [], 0, 0, False, None
    for p in pieces:
        if p["kind"] == "meme":
            path, n = p["path"], round(p["d"] * FPS)
            img = path.suffix.lower() in IMG_EXT
            inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{p['d']:.3f}", "-i", path] if img else ["-i", path]
            i = inputs.count("-i") - 1
            fc.append(f"[{i}:v]fps={FPS},trim=end_frame={n},setpts=PTS-STARTPTS,{fit(f'm{k}')},"
                      f"zoompan=z='max(1,1.2-0.04*on)':d=1:x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':s={W}x{H}:fps={FPS},"
                      f"fade=in:st=0:d=0.08:color=white,setsar=1[v{k}]")
            ns = round(p["d"] * SR)
            has_audio = not img and probe(path)[1]
            # the meme's own famous sound, loudness-matched; padded/trimmed to exactly the video length
            src_a = f"[{i}:a]aresample={SR},atrim=end_sample={ns},loudnorm=I=-14:TP=-1.5" if has_audio else f"anullsrc=r={SR}:cl=stereo"
            fc.append(f"{src_a},{AF},apad=whole_len={ns},atrim=end_sample={ns}[a{k}]")
            segs.append(f"[v{k}][a{k}]")
            k, flash = k + 1, True
            continue
        src, t0 = 0, 0.0
        if p.get("hook"):  # its own input: out of order, it would make ffmpeg hold the whole video in memory
            inputs += ["-ss", f"{p['a']:.4f}", "-t", f"{length(p) + 1:.3f}", "-i", video]
            src, t0 = inputs.count("-i") - 1, p["a"]
        parts = _shots(p["a"], p["b"], spans)
        jump = jump + 1 if shown is not None and parts[0][2] is shown else 0  # same camera before and after the cut
        for a, b, cx in parts:
            vf, shown = _frame(cx, k, sw, sh, a), cx
            if jump % 2:  # a jump cut inside one camera angle looks like a mistake unless the size changes clearly
                vf += f",scale={round(W * 1.14 / 2) * 2}:{round(H * 1.14 / 2) * 2},crop={W}:{H}:(iw-ow)/2:(ih-oh)*0.3"
            if flash:
                vf, flash = vf + ",fade=in:st=0:d=0.08:color=white", False
            fc.append(f"[{src}:v]trim=start_frame={round((a - t0) * FPS)}:end_frame={round((b - t0) * FPS)},setpts=PTS-STARTPTS,{vf},"
                      f"setsar=1[v{k}]")
            fc.append(f"[{src}:a]atrim=start_sample={round((a - t0) * SR)}:end_sample={round((b - t0) * SR)},asetpts=PTS-STARTPTS,"
                      + ("afade=t=in:d=0.02," if a == p["a"] else "")
                      + (f"afade=t=out:st={max(0.0, b - a - p['fade']):.3f}:d={p['fade']:.3f}," if b == p["b"] and p["fade"] else "") + f"{AF}[a{k}]")
            segs.append(f"[v{k}][a{k}]")
            k += 1
        flash = flash or bool(p.get("hook"))
    fc.append(f"{''.join(segs)}concat=n={len(segs)}:v=1:a=1[v][a]")
    (job / "edit_filter.txt").write_text(";\n".join(fc), encoding="utf-8")
    encode(["ffmpeg", "-y", *inputs, "-/filter_complex", "edit_filter.txt", "-map", "[v]", "-map", "[a]",
            "-c:a", "aac", "-ar", str(SR), "-r", str(FPS)], "edit.mp4", cwd=job)


def finish(job, dur, caps, title, stops, zooms, out, outro="", hits=()):
    """Punch zooms [(start, end, how much)], progress bar, captions, whoosh into memes, a low hit on each punchline (hits), -14
    LUFS. No music is ever added."""
    write_ass(job / "captions.ass", caps, title, stops, (dur - 4.0, dur, outro) if outro else None)
    inputs, fc, mix = ["-i", "edit.mp4"], [], ["[0:a]"]
    z = "+".join(f"{amount}*clip((in_time-{a:.2f})/0.1,0,1)*clip(({b:.2f}-in_time)/0.2,0,1)" for a, b, amount in zooms) or "0"
    fc.append(f"[0:v]zoompan=z='1+min(0.25,{z})':d=1:x='iw/2-iw/zoom/2':y='(ih-ih/zoom)*0.35':s={W}x{H}:fps={FPS}[z]")
    fc.append(f"color=c=0xFFE100:s={W}x14:r={FPS}[bar]")
    fc.append(f"[z][bar]overlay=x='-w+w*t/{dur:.3f}':y=H-14:shortest=1,ass=captions.ass,format=yuv420p[vout]")
    whoosh = LIB / "sfx" / "whoosh.wav"
    for k, s in enumerate(stops if whoosh.exists() else []):
        inputs += ["-i", whoosh]
        fc.append(f"[{inputs.count('-i') - 1}:a]adelay={int(max(0.0, s - 0.15) * 1000)}:all=1,{AF}[wh{k}]")
        mix.append(f"[wh{k}]")
    boom = LIB / "sfx" / "shock.wav"
    for k, t in enumerate(hits if boom.exists() else []):
        inputs += ["-i", boom]
        fc.append(f"[{inputs.count('-i') - 1}:a]volume=0.5,adelay={int(t * 1000)}:all=1,{AF}[hit{k}]")
        mix.append(f"[hit{k}]")
    fc.append(f"{''.join(mix)}amix=inputs={len(mix)}:normalize=0:duration=first,{LOUD}[aout]")
    (job / "finish_filter.txt").write_text(";\n".join(fc), encoding="utf-8")
    encode(["ffmpeg", "-y", *inputs, "-/filter_complex", "finish_filter.txt", "-map", "[vout]", "-map", "[aout]",
            "-t", f"{dur:.3f}", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"], out, cwd=job)
