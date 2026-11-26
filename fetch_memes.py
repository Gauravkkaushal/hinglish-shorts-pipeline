"""Fills library/<tag>/ with meme reaction clips from YouTube meme templates.

  python fetch_memes.py                 # 5 per tag
  python fetch_memes.py --per-tag 20    # ~220 clips
  python fetch_memes.py --recut         # re-cut the clips you already have with the current rules

Each clip = the LOUDEST 3.5s of its template (that's where the famous dialogue/sound is), black bars removed,
named after the template, plus <clip>.txt with what it says - so the Director can match it to the moment.
Add your own tags by adding to QUERIES (folder name = tag). Your own clips: just drop them in the folder."""
import argparse, json, subprocess

import numpy as np

from shorts.ai import meme_says
from shorts.eyes import shows_someone
from shorts.ff import JOBS, LIB, crop_box, probe, run, safe_name
from shorts.rights import has_music

QUERIES = {
    "laugh": "hindi meme template laughing",
    "cry": "hindi meme template crying",
    "shock": "hindi meme template shocked",
    "angry": "hindi meme template angry",
    "facepalm": "meme template facepalm",
    "awkward": "hindi meme template awkward",
    "celebrate": "hindi meme template celebration dance",
    "thinking": "hindi meme template thinking",
    "bruh": "meme template bruh",
    "cool": "hindi meme template attitude",
    "love": "hindi meme template romantic",
}
CLIP_LEN = 3.5
RAW = JOBS / "meme_raw"


def loudest_start(energy, win):
    """energy per 0.1s -> index where the loudest `win`-long stretch starts."""
    if len(energy) <= win:
        return 0
    return int(np.argmax(np.convolve(energy, np.ones(win), "valid")))


def cut_clip(src, out):
    dur, has_audio, _, _ = probe(src)
    start = 0.0
    if has_audio and dur > CLIP_LEN:
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-f", "s16le", "-ac", "1", "-ar", "8000", "-"],
                             capture_output=True).stdout
        e = np.frombuffer(pcm, np.int16).astype(np.float32) ** 2
        start = loudest_start(e[: len(e) // 800 * 800].reshape(-1, 800).sum(1), int(CLIP_LEN * 10)) / 10
    box = crop_box(src)
    vf = (f"crop={box}," if box else "") + "scale='min(720,iw)':-2"
    run(["ffmpeg", "-y", "-ss", f"{start:.2f}", "-t", f"{min(CLIP_LEN, dur - start):.2f}", "-i", src, "-vf", vf,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         *(["-c:a", "aac", "-ar", "48000"] if has_audio else ["-an"]), out])
    if has_audio:
        out.with_suffix(".txt").write_text(meme_says(out), encoding="utf-8")


def fetch(tag, query, n):
    have = {f.stem.rsplit("__", 1)[-1] for f in (LIB / tag).glob("*__*.mp4")}
    got = len(have)
    if got >= n:
        return got
    entries = json.loads(run(["yt-dlp", "-J", "--no-warnings", "--flat-playlist", f"ytsearch{n * 4}:{query}"]))["entries"] or []
    for e in entries:
        if got >= n:
            break
        vid, d = e.get("id"), e.get("duration") or 0
        if not vid or vid in have or not 2 <= d <= 40:  # templates are short; skip long compilations
            continue
        try:
            src = RAW / f"{vid}.mp4"
            if not src.exists():
                run(["yt-dlp", "--no-warnings", "-f", "bv*[height<=720]+ba/b[height<=720]/b", "--merge-output-format", "mp4",
                     "-o", src, f"https://www.youtube.com/watch?v={vid}"])
            out = LIB / tag / f"{safe_name(e.get('title') or '')}__{vid}.mp4"
            cut_clip(src, out)
            why = "music" if has_music(out) else "" if shows_someone(out) else "no face or no sound"
            if why:
                for f in (out, out.with_suffix(".txt"), out.with_name(out.name + ".music.json"), out.with_name(out.name + ".face.json")):
                    f.unlink(missing_ok=True)
                print(f"  {tag}: refused ({why}) {out.name}")
                continue
            got += 1
            print(f"  {tag}: {out.name}")
        except RuntimeError as err:
            print(f"  skip {vid}: {str(err)[-150:]}")
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tag", type=int, default=5)
    ap.add_argument("--recut", action="store_true", help="re-cut existing clips from their downloaded originals")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for tag, q in QUERIES.items():
        (LIB / tag).mkdir(parents=True, exist_ok=True)
        if args.recut:
            for clip in (LIB / tag).glob("*__*.mp4"):
                src = RAW / f"{clip.stem.rsplit('__', 1)[-1]}.mp4"
                if src.exists():
                    cut_clip(src, clip)
        try:
            got = fetch(tag, q, args.per_tag)
        except RuntimeError as err:
            print(f"{tag}: search failed {str(err)[-150:]}")
            continue
        if got:  # real clips arrived -> drop the placeholders
            for f in (LIB / tag).glob("_demo_*"):
                f.unlink()
        print(f"{tag}: {got} clips")


if __name__ == "__main__":
    main()
