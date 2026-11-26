"""Paths, settings and small ffmpeg helpers shared by every stage."""
import json, os, subprocess
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
LIB, JOBS, OUT = ROOT / "library", ROOT / "jobs", ROOT / "out"
ENV_FILE = ROOT / ".env"
load_dotenv(ENV_FILE)

W, H, FPS = 1080, 1920, 30
LOUD = "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000,alimiter=limit=0.75:level=false"  # loudnorm alone let drums clip
VID_EXT, IMG_EXT, AUD_EXT = {".mp4", ".mov", ".mkv", ".webm"}, {".png", ".jpg", ".jpeg", ".webp"}, {".mp3", ".wav", ".m4a"}
X264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
AF = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"  # every audio piece in one format for concat
COVER = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,fps={FPS}"


def env(name, default=None):
    return os.getenv(name) or default


class Stop(Exception):
    """Ends the whole run with a message. Not a RuntimeError on purpose: the per-video "skip and go on" handlers must not
    swallow it."""


class YouTubeBlocked(Stop):
    """YouTube asked this connection to prove it is not a bot: stop instead of hammering it."""


def run(cmd, cwd=None):
    cmd = [str(c) for c in cmd]
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        if cmd[0] == "yt-dlp" and "confirm you" in r.stderr:
            raise YouTubeBlocked("YouTube is refusing automated downloads from this connection right now. Try again in a "
                                 "few hours, or use your own files: python pipeline.py clip --src file.mp4")
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr[-3000:]}")
    return r.stdout


def probe(path):
    """-> (duration_s, has_audio, width, height)"""
    j = json.loads(run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height",
                        "-of", "json", path]))
    v = next((s for s in j.get("streams", []) if s["codec_type"] == "video"), {})
    return (float(j["format"].get("duration") or 0), any(s["codec_type"] == "audio" for s in j.get("streams", [])),
            v.get("width", 0), v.get("height", 0))


def crop_box(src):
    """Black bars baked into a video (letterbox/pillarbox) -> 'w:h:x:y' to crop them away, or None."""
    import re
    log = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(src), "-vf", "cropdetect=limit=24:round=2:reset=0", "-f", "null", "-"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    crops = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", log)
    _, _, iw, ih = probe(src)
    if not crops:
        return None
    w, h, x, y = map(int, crops[-1])
    if (w >= iw - 8 and h >= ih - 8) or w < iw * 0.3 or h < ih * 0.3:  # no bars, or the detector got it wrong
        return None
    return f"{w}:{h}:{x}:{y}"


def safe_name(title, n=50):
    import re, unicodedata
    keep = (c if c.isalnum() or unicodedata.category(c).startswith("M") else "_" for c in title)
    return re.sub(r"_+", "_", "".join(keep)).strip("_")[:n] or "clip"


def fit(k):
    """Any aspect -> 9:16: sharp frame centered on a blurred copy of itself (labels unique per k)."""
    return (f"split[fa{k}][fb{k}];[fa{k}]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=25:5[bg{k}];"
            f"[fb{k}]scale={W}:{H}:force_original_aspect_ratio=decrease[fg{k}];"
            f"[bg{k}][fg{k}]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={FPS}")


def encode(cmd, out, cwd=None):
    """Run an ffmpeg command ending in video encoding: GPU (NVENC) first, CPU x264 if that fails."""
    try:
        run(cmd + ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "20", "-b:v", "0", "-pix_fmt", "yuv420p", out], cwd=cwd)
    except RuntimeError as e:
        print("NVENC unavailable, using CPU:", str(e)[-200:])
        run(cmd + X264 + [out], cwd=cwd)
