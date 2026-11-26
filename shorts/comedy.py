"""Funny Reels: the best 30-50s stretch of each of several creators' videos on a topic -> one Reel each, with the clip's own
sound and nothing added. Stretches with background music, burned-in subtitles or mostly menu/map screens are refused."""
import subprocess

import numpy as np

from . import eyes
from .ai import llm, look, transcribe
from .creators import about
from .ff import probe, run
from .montage import topic_word
from .reel import singles
from .rights import has_music
from .sources import NORM, fit_length, seg_lines, snap_to_segments, ytjson

QUERIES = ["{topic} live hindi", "{topic} funny moments hindi", "{topic} gameplay hindi", "{topic} first time playing hindi"]
NO = ["trailer", "news", "explained", "leak", "theory", "compilation", "#shorts", "dub", "reupload"]
FUNNY_PROMPT = """Transcript of a Hindi video by {channel}: "{title}" ([start-end] seconds):
{lines}

This becomes a 30-50 second Instagram Reel on its own. Find the best continuous 30-50 second stretch: a mini story that
is funny even with zero context - a situation that builds up (chaos, a plan going wrong, banter) and lands on the
funniest moment (huge reaction, scream, savage roast, epic fail, hilarious line). Start at the beginning of a sentence
where the situation starts, end right after the biggest laugh. Rate honestly how funny the whole stretch is for Indian
Instagram viewers (1-10); most stretches are 4-6.
Return JSON: {{"start": <sec>, "end": <sec>, "score": <1-10>, "why": "<short>"}}"""

def find_creator_videos(topic, n=10, per_query=15, creators=()):
    """Popular videos of creators playing the topic, one per channel, round-robin over the searches."""
    pools = []
    for q in [about(c, topic) for c in creators] or [f"ytsearch{per_query}:{q.format(topic=topic)}" for q in QUERIES]:
        try:
            entries = ytjson("--flat-playlist", "--playlist-end", str(per_query), q).get("entries") or []
        except RuntimeError as e:
            print("  search failed:", q, str(e)[-120:])
            continue
        ok = [e for e in entries if e.get("id") and 300 <= (e.get("duration") or 0) <= 6 * 3600
              and topic_word(topic) in (e.get("title") or "").lower()
              and not any(x in (e.get("title") or "").lower() for x in NO)]
        pools.append(sorted(ok, key=lambda e: e.get("view_count") or 0, reverse=True))
    out, channels = [], set()
    for rank in range(max((len(p) for p in pools), default=0)):
        for pool in pools:
            if rank < len(pool) and len(out) < n:
                ch = pool[rank].get("channel") or pool[rank].get("uploader") or pool[rank]["id"]
                if ch not in channels:
                    out.append(pool[rank])
                    channels.add(ch)
    return out


def top_peaks(series, k, min_dist, skip_start=60, skip_end=30):
    """Indexes (= seconds) of the k highest points of a per-second series, >= min_dist apart, away from intro/outro."""
    picks = []
    for i in np.argsort(series)[::-1]:
        if skip_start <= i < len(series) - skip_end and all(abs(i - p) >= min_dist for p in picks):
            picks.append(int(i))
        if len(picks) == k:
            break
    return sorted(picks)


def candidate_moments(info, url, job, half):
    """YouTube's most-replayed peak if it exists, else the 3 loudest moments of the audio (screams, laughter)."""
    hm = [h for h in info.get("heatmap") or [] if 60 <= h["start_time"] <= (info.get("duration") or 0) - 30]
    if hm:
        peak = max(hm, key=lambda h: h["value"])
        return [(peak["start_time"] + peak["end_time"]) / 2]
    full = job / f"full_{info['id']}.m4a"
    if not full.exists():
        run(["yt-dlp", "--no-warnings", "-f", "wa[ext=m4a]/wa", "-o", full, url])
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(full), "-f", "s16le", "-ac", "1", "-ar", "8000", "-"], capture_output=True).stdout
    x = np.frombuffer(pcm, np.int16).astype(np.float32)
    rms = np.sqrt((x[: len(x) // 8000 * 8000].reshape(-1, 8000) ** 2).mean(1))
    return top_peaks(np.convolve(rms, np.ones(4) / 4, "same"), 3, min_dist=2 * half)


def stretches(e, job, min_score=7, half=75):
    """This creator's candidate stretches, best score first: [{url, a, b, score, channel, title}]."""
    url = f"https://www.youtube.com/watch?v={e['id']}"
    info = ytjson("--skip-download", url)
    channel, full, out = info.get("channel") or "", job / f"full_{info['id']}.m4a", []
    for mid in candidate_moments(info, url, job, half):
        a0, win = max(0.0, mid - half), job / f"win_{info['id']}_{int(mid)}.mp3"
        if not win.exists():
            if full.exists():
                run(["ffmpeg", "-y", "-ss", f"{a0:.1f}", "-t", f"{2 * half}", "-i", full, "-ac", "1", "-b:a", "64k", win])
            else:
                run(["yt-dlp", "--no-warnings", "-f", "ba", "-x", "--audio-format", "mp3", "--download-sections",
                     f"*{a0:.0f}-{mid + half:.0f}", "-o", str(win.with_suffix(".%(ext)s")), url])
        trans = transcribe(win)
        r = llm(FUNNY_PROMPT.format(channel=channel, title=info.get("title", ""), lines=seg_lines(trans)[:12000])) if trans else None
        try:
            a, b = fit_length(trans, *snap_to_segments(trans, float(r["start"]), float(r["end"])))
            score = float(r.get("score") or 0)
        except (TypeError, KeyError, ValueError):
            continue
        print(f"  {channel} @{a0 + a:.0f}s ({b - a:.0f}s): {score}/10 - {r.get('why')}")
        if 30 <= b - a <= 50 and score >= min_score:
            out.append({"url": url, "a": a0 + a, "b": a0 + b, "score": score, "channel": channel, "title": info.get("title", "")})
    return sorted(out, key=lambda m: -m["score"])


def refuse(clip):
    """Why a downloaded stretch cannot be a Reel ('' = fine)."""
    if has_music(clip):
        return "background music"
    dur = probe(clip)[0]
    why = [eyes.facts(f) if (f := eyes.grab_frame(clip, dur * q)) is not None else ["unreadable"]
           for q in (0.01, 0.12, 0.25, 0.38, 0.5, 0.62, 0.75, 0.88)]
    if "menu" in why[0] or sum("menu" in w for w in why) > 3:
        return "opens on or mostly shows a menu/map"
    if sum("subtitles" in w for w in why) >= 3:
        return "has its own burned-in subtitles"
    return "cannot be read" if sum("unreadable" in w for w in why) else ""


def what_it_shows(clip, job):
    """One line from the vision model about six frames in one picture ('' if it is unavailable)."""
    from PIL import Image
    dur, img = probe(clip)[0], Image.new("RGB", (3 * 400, 2 * 225))
    for k in range(6):
        f = eyes.grab_frame(clip, dur * (k + 0.5) / 6, 400)
        if f is not None:
            img.paste(Image.fromarray(f[..., ::-1]).resize((400, 225)), (400 * (k % 3), 225 * (k // 3)))
    img.save(job / "shows.jpg")
    v = look([job / "shows.jpg"], '{"shows": "<what happens across these 6 frames, in order, 30 words max>"}')[0]
    return [str(v["shows"])] if v.get("shows") else []


def build(topic, job, reels=2, n_videos=10, creators=()):
    """-> [(video, meta, problems)], one Reel per creator until `reels` have passed QA."""
    done = []
    for e in find_creator_videos(topic, n_videos, creators=creators):
        if sum(not p for _, _, p in done) >= reels:
            break
        try:
            for m in stretches(e, job):
                tag = f"{m['url'][-11:]}_{m['a']:.0f}_{m['b']:.0f}"
                raw, clip = job / f"raw_{tag}.mp4", job / f"clip_{tag}.mp4"
                if not clip.exists():
                    run(["yt-dlp", "--no-warnings", "-f", "bv*[height<=1080]+ba/b", "--download-sections", f"*{m['a']:.2f}-{m['b']:.2f}",
                         "--force-keyframes-at-cuts", "--merge-output-format", "mp4", "-o", raw, m["url"]])
                    run(["ffmpeg", "-y", "-i", raw, "-vf", NORM, "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
                         "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-ac", "2", clip])
                    raw.unlink(missing_ok=True)
                why = refuse(clip)
                if why:
                    print(f"  refused {tag}: {why}")
                    continue
                done += singles(job, topic, [{**m, "file": clip, "shows": what_it_shows(clip, job), "tag": tag}])
                break
        except RuntimeError as err:
            print("  skipped", e.get("id"), str(err)[-150:].replace("\n", " "))
    return done
