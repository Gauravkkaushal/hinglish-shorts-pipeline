"""The gate every finished video passes before it reaches out/. A video with any problem goes to out/rejected/ together
with the reasons; post.py refuses anything without a passing certificate for exactly that file."""
import hashlib, json, re, shutil, subprocess
from pathlib import Path

from .ff import Stop, FPS, OUT, run
from .rights import has_music

FONT = {"Cap": ("C:/Windows/Fonts/seguibl.ttf", 1080 - 100)}
LOOK_Q = ('{"unreadable_text": <true if one of OUR captions is garbled, cut off at the frame edge, or overlaps another one. Ours are the bold '
          'white/yellow/green words in the lower middle, the yellow title box and the red box; text in the scene itself (walls, '
          'logos, clothes, memes) is fine, even if cut>, '
          '"bad_frames": <how many of the 9 tiles are black, blank, or just a menu/map/logo>, "shows": "<max 15 words>"}')


def ass_events(path):
    """-> [(start, end, style, plain text)]"""
    sec = lambda t: sum(float(x) * m for x, m in zip(t.split(":"), (3600, 60, 1)))
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("Dialogue:"):
            f = line[9:].split(",", 9)
            out.append((sec(f[1].strip()), sec(f[2].strip()), f[3], re.sub(r"\{[^}]*\}", "", f[9])))
    return out


def ass_problems(path):
    """Devanagari in the caption font, captions that hang, two at once, or wider than the screen."""
    from PIL import ImageFont
    bad, last = [], 0.0
    for a, b, style, text in ass_events(path):
        if re.search("[\u0900-\u097f]", text):
            bad.append(f"Devanagari caption '{text}'")
        if style != "Cap":
            continue
        if b - a > 1.6:
            bad.append(f"caption '{text}' hangs for {b - a:.1f}s at {a:.1f}s")
        if a < last - 0.02:
            bad.append(f"two captions at once at {a:.1f}s")
        last = b
        size = int(re.search(r"Style: Cap,[^,]*,(\d+)", Path(path).read_text(encoding="utf-8")).group(1))
        if ImageFont.truetype(FONT["Cap"][0], size).getlength(text) > FONT["Cap"][1]:
            bad.append(f"caption too wide: '{text}'")
    return bad


LO, HI = 30, 50  # seconds


def media_problems(video, length, lo=LO, hi=HI):
    j = json.loads(run(["ffprobe", "-v", "error", "-count_packets", "-show_entries",
                        "stream=codec_type,width,height,r_frame_rate,duration,nb_read_packets", "-of", "json", video]))
    v = next((s for s in j["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    if not v or not a:
        return ["no video or no audio stream"]
    bad, dv, da = [], float(v["duration"]), float(a["duration"])
    if (v["width"], v["height"]) != (1080, 1920):
        bad.append(f"size {v['width']}x{v['height']}")
    if v["r_frame_rate"] != f"{FPS}/1" or abs(int(v["nb_read_packets"]) - dv * FPS) > 2:
        bad.append(f"frame rate: {v['nb_read_packets']} frames in {dv:.1f}s")
    if abs(dv - da) > 0.12:
        bad.append(f"picture {dv:.2f}s but sound {da:.2f}s")
    if length and abs(dv - length) > 0.2:
        bad.append(f"expected {length:.1f}s, got {dv:.1f}s")
    if not lo - 0.5 <= dv <= hi + 0.5:
        bad.append(f"length {dv:.1f}s is outside {lo}-{hi}s")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(video), "-vf", "freezedetect=n=0.001:d=0.5,blackdetect=d=0.2:pix_th=0.04",
                        "-af", "ebur128=peak=true,silencedetect=n=-45dB:d=1.5", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = r.stderr
    if r.returncode or re.search(r"(Error|Invalid data) ", log.split("Stream mapping")[-1]):
        bad.append("does not decode cleanly")
    frz = [float(x) for x in re.findall(r"freeze_duration: ([\d.]+)", log)]
    if frz and (max(frz) > 2.5 or sum(frz) > 0.12 * dv):
        bad.append(f"picture frozen {sum(frz):.1f}s in total (longest {max(frz):.1f}s)")
    if "black_start" in log:
        bad.append("black frames")
    if "silence_start" in log:
        bad.append("silence longer than 1.5s")
    loud = float(re.findall(r"I:\s+(-?[\d.]+|-inf) LUFS", log)[-1])
    peak = re.findall(r"Peak:\s+(-?[\d.]+|-inf) dBFS", log)
    if not -16.5 <= loud <= -12:
        bad.append(f"loudness {loud} LUFS")
    if peak and float(peak[-1]) > -0.3:
        bad.append(f"clipping (peak {peak[-1]} dB)")
    return bad


def eye_problems(video, kind, n=9):
    """Look at n frames: overlays locally, then all of them as one contact sheet through the vision model."""
    from PIL import Image
    from . import eyes
    from .ai import look
    dur = float(json.loads(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", video]))["format"]["duration"])
    fr = [f for f in (eyes.grab_frame(video, dur * (k + 0.5) / n, 720) for k in range(n)) if f is not None]
    if len(fr) < n:
        return ["some frames cannot be read"]
    bad = []
    if kind == "montage":
        box = [f[int(f.shape[0] * 0.342):int(f.shape[0] * 0.658)] for f in fr]  # the video box, without our own text
        bad += [f"{w} in the edit" for w in eyes.overlays(box)]
        if sum("subtitles" in eyes.facts(f) for f in box) >= 2:
            bad.append("the footage has its own subtitles")
    img = Image.new("RGB", (3 * 300, 3 * 533))
    for k, f in enumerate(fr):
        img.paste(Image.fromarray(f[..., ::-1]).resize((300, 533)), (300 * (k % 3), 533 * (k // 3)))
    s = Path(f"{video}.sheet.jpg")
    img.save(s)
    try:
        v = look([s], LOOK_Q)[0]
    finally:
        s.unlink(missing_ok=True)
    if not v:
        return bad + ["vision check unavailable: nobody has looked at this video"]
    if v.get("unreadable_text") is True:
        bad.append("unreadable or overlapping text")
    if isinstance(v.get("bad_frames"), (int, float)) and v["bad_frames"] > n // 2:
        bad.append("mostly black/menu/map/logo screens")
    return bad


def deliver(video, meta, kind, ass=(), music_ok=False, length=None, eyes=True, extra=()):
    """Check, certify or reject. -> (final path, problems)"""
    bad = list(extra) + media_problems(video, length)
    for p in ass:
        bad += ass_problems(p)
    if not music_ok and has_music(video):
        bad.append("music heard in a video that must have none")
    stopped = None
    if eyes and not bad:
        try:
            bad += eye_problems(video, kind)
        except Stop as e:  # rendered but nobody has looked: it must not sit among the finished ones
            bad, stopped = [UNSEEN], e
    Path(f"{video}.music.json").unlink(missing_ok=True)
    video = certify(video, meta, kind, bad)
    if stopped:
        raise Stop(f"{stopped}, or finish this one with: python pipeline.py check {video.relative_to(OUT.parent)}")
    return video, bad


UNSEEN = "the last look is missing (daily limit)"


def certify(video, meta, kind, bad):
    """Finished -> out/, refused -> out/rejected/, with a certificate that only fits these exact bytes. -> final path"""
    home = OUT / "rejected" if bad else OUT
    home.mkdir(parents=True, exist_ok=True)
    cert = {"passed": not bad, "problems": bad, "kind": kind, "sha256": hashlib.sha256(video.read_bytes()).hexdigest()}
    if video.parent.resolve() != home.resolve():
        video.with_suffix(".json").unlink(missing_ok=True)
        (home / video.name).unlink(missing_ok=True)
        shutil.move(str(video), home / video.name)
        video = home / video.name
    video.with_suffix(".json").write_text(json.dumps({**meta, "qa": cert}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(("  QA PASSED: " if not bad else "  QA REJECTED: ") + video.name + ("" if not bad else " -> " + "; ".join(bad)))
    return video


def finish_check(video):
    """The missing last look at a video that passed everything else. -> its problems, or None when that is not its state."""
    info = video.with_suffix(".json")
    meta = json.loads(info.read_text(encoding="utf-8")) if info.exists() else {}
    cert = meta.pop("qa", {})
    if cert.get("problems") != [UNSEEN] or cert.get("sha256") != hashlib.sha256(video.read_bytes()).hexdigest():
        return None
    bad = eye_problems(video, cert["kind"])
    certify(video, meta, cert["kind"], bad)
    return bad


def certified(video):
    """The metadata of a video that passed QA and has not changed since; raises otherwise."""
    video = Path(video)
    rec = video.with_suffix(".json")
    meta = json.loads(rec.read_text(encoding="utf-8")) if rec.exists() else {}
    cert = meta.get("qa") or {}
    if not cert.get("passed"):
        raise RuntimeError(f"{video.name} has not passed QA: {'; '.join(cert.get('problems', ['no certificate']))}")
    if cert["sha256"] != hashlib.sha256(video.read_bytes()).hexdigest():
        raise RuntimeError(f"{video.name} changed after its QA check")
    return meta
