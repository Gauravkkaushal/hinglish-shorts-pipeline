"""Looking at footage on this laptop, free and without limits: faces (YuNet, Apache-2.0) and on-screen text (RapidOCR)."""
import difflib, json, re, subprocess
from pathlib import Path

import numpy as np

from .ff import ROOT, probe

YUNET = ROOT / "assets" / "models" / "yunet.onnx"
MENU_WORDS = {"map", "brief", "progress", "settings", "stats", "paused", "pause", "inventory", "loading", "resume", "quit", "legend", "waypoint"}
_yn = _ocr = None


def grab_frame(video, t, width=960):
    """-> BGR array of one frame, or None."""
    _, _, w, h = probe(video)
    hh = round(h * width / w / 2) * 2
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0, t):.2f}", "-i", str(video), "-frames:v", "1", "-vf", f"scale={width}:{hh}",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(hh, width, 3).copy() if len(raw) == hh * width * 3 else None


def faces(img, score=0.7):
    """[(x, y, w, h)] as fractions of the frame, biggest first."""
    global _yn
    import cv2
    if _yn is None:
        _yn = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), score)
    H, W = img.shape[:2]
    _yn.setInputSize((W, H))
    _, found = _yn.detect(img)
    found = [] if found is None else found
    sure = max((f[14] for f in found), default=0) - 0.12  # a logo on the wall scores 0.75 next to a real face's 0.92
    out = [(f[0] / W, f[1] / H, f[2] / W, f[3] / H) for f in found if f[14] >= sure]
    return sorted(out, key=lambda f: -f[2] * f[3])


def texts(img, sure=0.8):
    """[(text, (x0, y0, x1, y1))] with the box as fractions of the frame."""
    global _ocr
    if _ocr is None:
        import logging
        from rapidocr import RapidOCR
        logging.disable(logging.WARNING)  # it logs every frame without text
        _ocr = RapidOCR()
    r, (H, W) = _ocr(img), img.shape[:2]
    out = []
    for t, s, box in zip(r.txts or [], r.scores or [], r.boxes if r.boxes is not None else []):
        if s >= sure:
            b = np.array(box)
            out.append((t, (b[:, 0].min() / W, b[:, 1].min() / H, b[:, 0].max() / W, b[:, 1].max() / H)))
    return out


def sentence(t):
    """A burned-in subtitle is a run of words, not a HUD label."""
    words = re.findall(r"[A-Za-z']{2,}", t)
    return len(words) >= 4 or len("".join(words)) >= 18


def subtitles(tx):
    """The boxes of the texts that are burned-in subtitles."""
    low = lambda b: b[1] > 0.8 and 0.35 < (b[0] + b[2]) / 2 < 0.65  # even "Of course." there is dialogue
    lines = [(t, b) for t, b in tx if (b[2] - b[0]) * 16 > 2 * (b[3] - b[1]) * 9]  # a line of text, not a logo on a mug
    return [b for t, b in lines if (sentence(t) and b[1] > 0.5) or (low(b) and len(re.findall(r"[A-Za-z']{2,}", t)) >= 2)]


def facts(img):
    """What disqualifies a frame by itself: subtitles / menu / title."""
    tx, why = texts(img), []
    if subtitles(tx):
        why.append("subtitles")
    words = {w for t, _ in tx for w in re.findall(r"[a-z]+", t.lower())}
    if len(words & MENU_WORDS) >= 2 or len(tx) >= 14:
        why.append("menu")
    if sum((b[2] - b[0]) * (b[3] - b[1]) for _, b in tx) > 0.035 and "subtitles" not in why:
        why.append("title")
    return why


def same(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() > 0.75


def overlays(frames, share=0.6):
    """What stays put while the picture changes: a small face near an edge = facecam; the same word = watermark."""
    F, T, need = [faces(f) for f in frames], [texts(f) for f in frames], max(2, round(share * len(frames)))
    found = []
    edge = lambda f: (f[0] + f[2] / 2 < 0.3 or f[0] + f[2] / 2 > 0.7) and 0.02 < f[2] < 0.2 and f[3] < 0.35
    for f in (f for fs in F for f in fs if edge(f)):
        if sum(any(abs(f[0] - g[0]) < 0.05 and abs(f[1] - g[1]) < 0.07 for g in fs) for fs in F) >= need:
            found.append("facecam")
            break
    for t, b in ((t, b) for tx in T for t, b in tx if len(re.findall("[A-Za-z]", t)) >= 5):
        if sum(any(same(t, u) and abs(b[0] - c[0]) < 0.04 and abs(b[1] - c[1]) < 0.04 for u, c in tx) for tx in T) >= need:
            found.append("watermark")
            break
    return found


def shows_someone(clip, used=3.5):
    """A reaction clip must show a face in (nearly) all of the part we cut in: no logos, cards or screen recordings - and
    have its own sound. Cached."""
    cache = Path(f"{clip}.face.json")
    if not cache.exists():
        d = min(probe(clip)[0], used)
        seen = [(f := grab_frame(clip, d * q)) is not None and bool(faces(f)) for q in (0.1, 0.35, 0.6, 0.85)]
        log = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-t", str(used), "-i", str(clip), "-af", "silencedetect=n=-40dB:d=1.2",
                              "-vn", "-f", "null", "-"], capture_output=True, text=True, errors="replace")
        heard = log.returncode == 0 and "silence_start" not in log.stderr
        cache.write_text(json.dumps(sum(seen) >= 3 and heard))
    return json.loads(cache.read_text())


def window_overlays(video, n=6):
    """facecam / watermark in a short window of footage. Cached."""
    cache = Path(f"{video}.overlay.json")
    if not cache.exists():
        dur = probe(video)[0]
        fr = [f for f in (grab_frame(video, dur * (k + 0.5) / n) for k in range(n)) if f is not None]
        cache.write_text(json.dumps(overlays(fr) if len(fr) >= 3 else ["unreadable"]))
    return json.loads(cache.read_text())
