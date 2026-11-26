"""Music policy. Talk / comedy / reels: never any added music. Music edits: only (a) a beat this code synthesises itself,
or (b) a file in library/music/licensed/ that has a <file>.license.json. Anything in which the audio classifier hears
music that we did not put there is refused (has_music)."""
import json, os, subprocess, zipfile
from pathlib import Path

import numpy as np

from .ff import LIB, ROOT

MODEL = ROOT / "assets" / "models" / "yamnet.tflite"  # Google YAMNet, Apache-2.0
LICENSED = LIB / "music" / "licensed"
MUSICAL = {"Music", "Singing", "Song", "Musical instrument", "Hip hop music", "Pop music", "Electronic music", "Rapping"}
SR = 16000
_net = None


def _classifier():
    global _net
    if _net is None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        it = tf.lite.Interpreter(str(MODEL))
        it.allocate_tensors()
        with zipfile.ZipFile(MODEL) as z:
            labels = z.read("yamnet_label_list.txt").decode().splitlines()
        _net = it, labels
    return _net


def hear(path, names, hop=None):
    """How likely any of these sounds is (0-1), per ~1s of a file's audio or every hop seconds. Empty if there is no audio."""
    it, labels = _classifier()
    idx = [i for i, l in enumerate(labels) if l in names]
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True).stdout
    x, n, scores = np.frombuffer(pcm, np.float32), int(np.prod(inp["shape"])), []
    if hop:  # windows centred on 0, hop, 2 * hop...
        x = np.pad(x, (n // 2, n))
    for k in range(0, len(x) - n + 1, int(hop * SR) if hop else n):
        it.set_tensor(inp["index"], x[k:k + n].reshape(inp["shape"]))
        it.invoke()
        scores.append(float(it.get_tensor(out["index"]).reshape(-1)[idx].max()))
    return scores


def music_track(path):
    return hear(path, MUSICAL)


def is_musical(scores, thr=0.3, share=0.10, run=2):
    """Music if it is heard in >10% of the file or for 2+ seconds in a row."""
    longest = cur = 0
    for s in scores:
        cur = cur + 1 if s > thr else 0
        longest = max(longest, cur)
    return bool(scores) and (sum(s > thr for s in scores) / len(scores) > share or longest >= run)


def has_music(path):
    """Cached next to the file. Limit: very quiet music under loud speech can still be missed."""
    cache = Path(f"{path}.music.json")
    if cache.exists() and cache.stat().st_mtime >= Path(path).stat().st_mtime:
        return json.loads(cache.read_text())
    hit = is_musical(music_track(path))
    cache.write_text(json.dumps(hit))
    return hit


def licence_of(path):
    """The licence record of a music file, or a refusal. Required: source, license, url."""
    rec = Path(f"{path}.license.json")
    data = json.loads(rec.read_text(encoding="utf-8")) if rec.exists() else {}
    missing = [k for k in ("source", "license", "url") if not data.get(k)]
    if missing:
        raise RuntimeError(f"refused: {Path(path).name} has no licence record ({rec.name} needs {', '.join(missing)})")
    return data


def original_beat(out, seconds, bpm=100, seed=0):
    """Synthesise an original dark drum beat (kick, snare, hats, 808, bell line). Nothing sampled, so nobody owns it.
    -> {start, beats, strong, tempo, licence}; beat times are exact because we placed them."""
    sr, rng = 44100, np.random.default_rng(seed)
    n, step = int(seconds * sr), 60 / bpm
    mix = np.zeros(n + sr * 2)

    def put(t, s, gain):
        i = int(t * sr)
        if i < n:
            mix[i:i + len(s)] += gain * s[:len(mix) - i]

    def tone(freq, dur, decay, harm=(1,)):
        t = np.arange(int(dur * sr)) / sr
        return sum(np.sin(2 * np.pi * freq * (k + 1) * t) * a for k, a in enumerate(harm)) * np.exp(-t * decay)

    t = np.arange(int(0.35 * sr)) / sr
    kick = np.sin(2 * np.pi * (45 * t + 22 * (1 - np.exp(-t * 28)))) * np.exp(-t * 9)
    noise = rng.standard_normal(int(0.25 * sr))
    snare = noise * np.exp(-np.arange(len(noise)) / sr * 22) * 0.6 + tone(190, 0.25, 20) * 0.5
    hat = np.diff(rng.standard_normal(int(0.06 * sr) + 1)) * np.exp(-np.arange(int(0.06 * sr)) / sr * 70)
    scale = 55 * 2 ** (np.array([0, 3, 5, 7, 10, 12]) / 12)  # A minor pentatonic
    root = [scale[i] for i in rng.choice([0, 0, 1, 2, 3], 4)]
    riff = rng.choice(len(scale), 8)
    beats = [k * step for k in range(int(np.ceil(seconds / step))) if k * step < seconds]
    for k, b in enumerate(beats):
        bar, pos, full = k // 4, k % 4, k >= 4  # first bar: bell + hats only, then the drop
        put(b, hat, 0.16)
        put(b + step / 2, hat, 0.12)
        put(b + step * 0.75, hat, 0.06)
        for h in (0, 0.5):
            put(b + h * step, tone(scale[riff[(2 * k + int(h * 2)) % 8]] * 8, 0.3, 9, (1, 0.5, 0.25)), 0.10)
        if full:
            if pos in (0, 2):
                put(b, kick, 0.95)
            if pos == 2:
                put(b + step * 0.5, kick, 0.6)
            if pos in (1, 3):
                put(b, snare, 0.55)
            if pos == 0:
                put(b, tone(root[bar % 4], step * 3.5, 1.2, (1, 0.35)), 0.45)
    mix = np.tanh(mix[:n] * 1.2)
    fade = int(0.8 * sr)
    mix[-fade:] *= np.linspace(1, 0, fade)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", "-c:a", "pcm_s16le",
                    str(out)], input=(mix * 0.9).astype(np.float32).tobytes(), check=True)
    strong = [b for k, b in enumerate(beats) if k >= 4 and k % 8 == 4]  # every second bar, starting on the drop
    return {"start": 0.0, "beats": beats, "strong": strong, "tempo": float(bpm),
            "licence": {"source": "synthesised by this pipeline", "license": "original work", "url": "-"}}
