"""AI services: LLM and vision (Groq / any OpenAI-compatible), transcription with repair, TTS, Hinglish."""
import asyncio, base64, difflib, hashlib, json, re, subprocess, time
from pathlib import Path

import requests

from .ff import JOBS, Stop, env, probe, run

GROQ = "https://api.groq.com/openai/v1"


def key():
    return env("LLM_API_KEY") or env("GROQ_API_KEY")


def daily_limit(r):
    if "per day" in r.text:
        raise Stop(f"the free daily Groq limit is used up ({r.json()['error']['message'].split('Please try again in ')[-1].split('.')[0]} "
                   "until the next call fits). Everything so far is cached: run the same command again later")


def llm(prompt, temperature=0.3, effort="medium", model=None):
    """JSON-mode chat call. None when there is no key or the call keeps failing (callers then use rules)."""
    if not key():
        return None
    model = not env("LLM_BASE_URL") and model or env("LLM_MODEL", "openai/gpt-oss-120b")
    cache = JOBS / "llm_cache" / (hashlib.sha256((model + prompt).encode()).hexdigest()[:24] + ".json")
    if temperature == 0 and cache.exists():  # a run stopped by the daily limit must not pay for the same answer again
        return json.loads(cache.read_text(encoding="utf-8"))
    for attempt in range(3):
        try:
            r = requests.post(env("LLM_BASE_URL", GROQ) + "/chat/completions", timeout=180,
                              headers={"Authorization": f"Bearer {key()}"},
                              json={"model": model, "temperature": temperature, "max_completion_tokens": 8000,
                                    "response_format": {"type": "json_object"},
                                    **({"reasoning_effort": effort} if "gpt-oss" in model else {}),
                                    "messages": [{"role": "user", "content": prompt}]})
            if r.status_code == 429:
                daily_limit(r)
            if r.status_code == 429 and attempt < 2:  # free-tier rate limit: wait and retry
                time.sleep(min(90, float(r.headers.get("retry-after", 15))))
                continue
            r.raise_for_status()
            answer = json.loads(r.json()["choices"][0]["message"]["content"])
            if temperature == 0:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(answer, ensure_ascii=False), encoding="utf-8")
            return answer
        except requests.ConnectionError:
            print("  no internet, waiting 20s")
            time.sleep(20)
        except (requests.RequestException, KeyError, ValueError) as e:
            print(f"LLM call failed ({str(e)[:90]})" + (", retrying" if attempt < 2 else ", using fallback"))
    return None


def look(frames, question, batch=2):
    """Ask the vision model the same question about every frame (JPEG paths) -> list of dicts ({} where it failed)."""
    out = []
    for i in range(0, len(frames), batch):
        part, got = frames[i:i + batch], None
        content = [{"type": "text", "text": f"{len(part)} image(s) follow. For EACH image, in order: {question}\n"
                    f'Answer with only JSON: {{"images": [ exactly {len(part)} object(s) ]}}'}]
        content += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                     base64.b64encode(Path(f).read_bytes()).decode()}} for f in part]
        for attempt in range(5 if env("GROQ_API_KEY") else 0):
            try:  # no JSON mode: it 400s on this model; the object is cut out of the text instead
                r = requests.post(f"{GROQ}/chat/completions", timeout=120,
                                  headers={"Authorization": f"Bearer {env('GROQ_API_KEY')}"},
                                  json={"model": env("VISION_MODEL", "qwen/qwen3.8-27b"), "temperature": 0.1 * attempt,
                                        "max_completion_tokens": 1500, "reasoning_effort": "none",
                                        "messages": [{"role": "user", "content": content}]})
                if r.status_code == 429:
                    daily_limit(r)
                    time.sleep(min(60, float(r.headers.get("retry-after", 15))))
                    continue
                r.raise_for_status()
                t = r.json()["choices"][0]["message"]["content"]
                got = json.loads(t[t.index("{"):t.rindex("}") + 1]).get("images")
                if isinstance(got, list) and len(got) == len(part):
                    break
            except requests.ConnectionError:
                print("  no internet, waiting 20s")
                time.sleep(20)
            except (requests.RequestException, KeyError, ValueError) as e:
                print(f"  vision call failed: {str(e)[:90]}")
        ok = isinstance(got, list) and len(got) == len(part)
        out += [g if isinstance(g, dict) else {} for g in got] if ok else [{} for _ in part]
    return out


# ---------- transcription ----------
MAX_WORD = 0.9  # seconds: nobody holds one word longer than this on screen


def _groq_whisper(path, language=None, af=""):
    mp3 = Path(f"{path}.16k.mp3")  # 16k mono, 64k keeps ~50 min of audio under Groq's 25MB limit
    run(["ffmpeg", "-y", "-i", path, "-vn", *(["-af", "aformat=channel_layouts=mono," + af] if af else []), "-ac", "1", "-ar", "16000", "-b:a", "64k", mp3])
    data = {"model": "whisper-large-v3", "response_format": "verbose_json", "timestamp_granularities[]": ["word", "segment"]}
    if language:
        data["language"] = language
    try:
        for attempt in range(3):
            with open(mp3, "rb") as f:
                r = requests.post(f"{GROQ}/audio/transcriptions", headers={"Authorization": f"Bearer {env('GROQ_API_KEY')}"},
                                  data=data, files={"file": (mp3.name, f, "audio/mpeg")}, timeout=300)
            if r.status_code == 429 and attempt < 2:
                time.sleep(min(60, float(r.headers.get("retry-after", 15))))
                continue
            break
    finally:
        mp3.unlink(missing_ok=True)
    r.raise_for_status()
    return r.json()


def in_order(words):
    """Spoken order is the recogniser's order; a word whose time jumps out of line (Whisper does that) is put between its
    neighbours. Sorting by time instead would swap words."""
    best = []  # longest run of non-decreasing starts: (length, previous index)
    for i, w in enumerate(words):
        prev = max((k for k in range(i) if words[k][0] <= w[0]), key=lambda k: best[k][0], default=None)
        best.append((1 + (best[prev][0] if prev is not None else 0), prev))
    i, keep = max(range(len(words)), key=lambda k: best[k][0], default=None), set()
    while i is not None:
        keep.add(i)
        i = best[i][1]
    for i, w in enumerate(words):
        if i not in keep:
            a = words[i - 1][1] if i else 0.0
            b = next((words[k][0] for k in range(i + 1, len(words)) if k in keep), a + 0.2)
            w[0], w[1] = min(a, b), max(b, min(a, b) + 0.05)
    return words


def _words(j, offset=0.0):
    return in_order([[w["start"] + offset, w["end"] + offset, w["word"].strip()] for w in j.get("words") or [] if w["word"].strip()])


HEARINGS = (("", 1.0), ("atempo=0.9", 0.9), ("highpass=f=120,volume=0.7", 1.0))  # (audio filter, tempo)
MIN_CLEAR = 0.8  # clear talk 0.85-0.95, word salad 0.61-0.69 (measured)


def bones(words):
    """Consonants only, in Roman: what survives two spellings of the same speech."""
    return re.sub(r"[^a-z0-9]|[aeiouyh]", "", "".join(romanise(w) if is_devanagari(w) else w for w in words).lower())


SOUNDS = str.maketrans("wfzqctpg", "vvjkkdbk")


def sounds_like(a, b):
    """Could b be what was said where a was heard? Same consonants, roughly; a word may not move to another's place."""
    x, y = (bones([(romanise(w) if is_devanagari(w) else w.lower()).replace("ph", "f")]).translate(SOUNDS) for w in (a, b))
    return len(x + y) < 2 or difflib.SequenceMatcher(None, x, y, autojunk=False).ratio() >= 0.4


def consensus(hearings, handicap=()):
    """The hearing (a list of words) that agrees most with the others -> (its index, how much: 0-1). Real speech comes out the
    same each time; misheard speech comes out different each time."""
    b = [bones(h) for h in hearings]
    agree = [sum(difflib.SequenceMatcher(None, x, y, autojunk=False).ratio() for k, y in enumerate(b) if k != i) / (len(b) - 1)
             for i, x in enumerate(b)]
    best = max(range(len(b)), key=lambda i: agree[i] - (handicap[i] if handicap else 0))
    return best, round(agree[best], 3)


def clarity(path):
    """How surely the words of a short clip were understood (0-1), or None when it was heard only once."""
    f = Path(f"{path}.clear.json")
    return json.loads(f.read_text()) if f.exists() else None


def heard(path):
    """Everything the recogniser heard in a short clip, one text per hearing."""
    f = Path(f"{path}.heard.json")
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def check_clear(path):
    if (c := clarity(path)) is not None and c < MIN_CLEAR:
        raise RuntimeError(f"speech was not understood clearly (clarity {c:.2f}): its captions would be nonsense")


def transcribe(path):
    """-> [{start, end, text, words: [[start, end, word], ...]}]. Groq whisper-large-v3 (free, best Hindi), local fallback.
    A clip up to 150s is heard three ways and the consensus kept (see clarity). Mangled stretches are re-heard and timings made
    sane. Cached next to the media file."""
    cache = Path(f"{path}.words.json")
    if cache.exists() and cache.stat().st_mtime >= Path(path).stat().st_mtime:
        return json.loads(cache.read_text(encoding="utf-8"))
    segs = None
    if env("GROQ_API_KEY"):
        try:
            lang = env("WHISPER_LANG", "hi")
            js = [_groq_whisper(path, lang, af) for af, _ in (HEARINGS if probe(path)[0] <= 150 else HEARINGS[:1])]
            k = 0
            if len(js) > 1:
                k, clear = consensus([[w[2] for w in _words(j)] for j in js],
                                     [0 if t == 1 else 0.06 for _, t in HEARINGS])  # slowed audio: better words, looser times
                Path(f"{path}.clear.json").write_text(json.dumps(clear))
                Path(f"{path}.heard.json").write_text(json.dumps([j.get("text", "") for j in js], ensure_ascii=False), encoding="utf-8")
            tempo = HEARINGS[k][1]
            segs = [{**x, "start": x["start"] * tempo, "end": x["end"] * tempo} for x in js[k].get("segments") or []]
            words = [[a * tempo, b * tempo, w] for a, b, w in _words(js[k])]
        except (requests.RequestException, KeyError, ValueError) as e:
            print("Groq transcription failed, using local whisper:", str(e)[:120])
            segs = None
    if segs is None:
        segs, words = _local_whisper(path)
    words = repair(path, words) if env("GROQ_API_KEY") else tidy(words)
    out = [dict(start=s["start"], end=s["end"], text=s["text"].strip(),
                words=[]) for s in segs]
    for w in words:  # a word in the pause between two segments belongs to the one before it
        if out:
            max((s for s in out if s["start"] <= w[0]), key=lambda s: s["start"], default=out[0])["words"].append(w)
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def holes(words, long_word=1.2, gap=1.5):
    """Stretches Whisper mangled: one word claiming more than long_word seconds, or a suspicious gap between words."""
    out = [(a, b) for a, b, _ in words if b - a > long_word]
    out += [(x[1], y[0]) for x, y in zip(words, words[1:]) if y[0] - x[1] > gap]
    return sorted(out)


def repair(path, words):
    """Re-listen to each mangled stretch on its own (Whisper usually hears it fine in isolation)."""
    words = sorted(words, key=lambda w: w[0])
    for a, b in holes(words)[:12]:
        part, t0 = Path(f"{path}.hole.wav"), max(0.0, a - 0.15)
        try:
            run(["ffmpeg", "-y", "-ss", f"{t0:.2f}", "-t", f"{b - a + 0.3:.2f}", "-i", path, "-vn", "-ac", "1", part])
            new = _words(_groq_whisper(part, env("WHISPER_LANG", "hi")), t0)
        except (requests.RequestException, RuntimeError):
            continue
        finally:
            part.unlink(missing_ok=True)
        new = [w for w in new if a - 0.05 <= w[0] < b - 0.05 and w[1] - w[0] < 1.2]
        if len(new) > 1:  # the single stretched word was really several
            words = [w for w in words if not a - 0.01 <= w[0] < b - 0.01] + new
    return tidy(words)


def tidy(words):
    """In order, none overlapping the next, none longer than MAX_WORD."""
    words = sorted((list(w) for w in words), key=lambda w: w[0])
    for i, w in enumerate(words):
        nxt = words[i + 1][0] if i + 1 < len(words) else w[1]
        w[1] = max(w[0] + 0.05, min(w[1], w[0] + MAX_WORD, nxt if nxt > w[0] else w[1]))
    return words


def meme_says(path):
    """What a meme clip says (any language), for the Director. '' if no key/audio."""
    try:
        return _groq_whisper(path).get("text", "").strip()[:150] if env("GROQ_API_KEY") else ""
    except requests.RequestException:
        return ""


def _local_whisper(path):
    import numpy as np
    from faster_whisper import WhisperModel
    # decode with ffmpeg ourselves: faster-whisper's PyAV decoder breaks on newer `av` versions
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    for device, ctype in (("cuda", "int8_float16"), ("cpu", "int8")):
        try:
            model = WhisperModel(env("WHISPER_MODEL", "large-v3-turbo"), device=device, compute_type=ctype)
            it, _ = model.transcribe(audio, language=env("WHISPER_LANG", "hi"), word_timestamps=True, vad_filter=True)
            segs = list(it)
            return ([dict(start=s.start, end=s.end, text=s.text) for s in segs],
                    [[w.start, w.end, w.word.strip()] for s in segs for w in s.words])
        except (RuntimeError, ValueError) as e:  # e.g. missing CUDA DLLs -> retry on CPU
            print(f"local whisper on {device} failed ({e})")
    raise RuntimeError("whisper failed on cuda and cpu")


# ---------- voice ----------
def tts(text, out):
    """edge-tts voiceover -> words with timestamps [[start, end, word]]."""
    import edge_tts

    async def go():
        words = []
        c = edge_tts.Communicate(text, env("TTS_VOICE", "hi-IN-MadhurNeural"), rate=env("TTS_RATE", "+8%"),
                                 boundary="WordBoundary")
        with open(out, "wb") as f:
            async for ch in c.stream():
                if ch["type"] == "audio":
                    f.write(ch["data"])
                elif ch["type"] == "WordBoundary":
                    words.append([ch["offset"] / 1e7, (ch["offset"] + ch["duration"]) / 1e7, ch["text"]])
        return words
    return asyncio.run(go())


# ---------- Hinglish ----------
HINDI = "qwen/qwen3.8-27b"  # knows Hindi far better than gpt-oss: "The Nun" and "watchman", not "darnan" and "vachchman"
SAID_PROMPT = """Several speech recognisers heard the same Hindi/Hinglish talk. Each is wrong in different places (sounds kept,
words wrong):
{heard}
Write what the speaker really said, word for word, in Roman Hinglish with English words in correct English. Use sound, sense
and the topic of the talk: a phrase that makes no sense was mis-heard, find the similar-sounding one that fits. Add nothing.
Return JSON {{"said": "..."}}"""
HINGLISH_PROMPT = """Transliterate every word below into natural Hinglish (Roman script), spelled the way young Indians type
on WhatsApp/Instagram: "क्या"->"kya", "भाई"->"bhai", "है"->"hai", "मैं"->"main", "नहीं"->"nahi", "बोला"->"bola",
"ज़िंदगी"->"zindagi", "अच्छा"->"accha". English words written in Devanagari go back to normal English:
"रिलेशनशिप"->"relationship", "ऑफिस"->"office", "सीरियसली"->"seriously". Words already in Roman stay unchanged. Keep numbers.
These words come from speech recognition: if a word is clearly mis-heard, write the word the speaker obviously said
(use the surrounding words as context), e.g. "जिंदी" in "आपकी जिंदी खराब" -> "zindagi".
{heard}Every number keeps its own word: do NOT translate, merge, split, move or drop words.
Return JSON {{"words": {{"0": "...", "1": "..."}}}} with all {n} numbers.
Words: {items}"""
NUKTA_SIGN, VIRAMA = "़", "्"
CONS = dict(zip("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ",
                "k kh g gh n ch chh j jh n t th d dh n t th d dh n p ph b bh m y r l v sh sh s h l".split()))
NUKTA = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "फ": "f", "ड": "d", "ढ": "dh"}
VOWELS = dict(zip("अआइईउऊएऐओऔऋऑऍ", "a aa i ee u oo e ai o au ri o e".split()))
MATRA = dict(zip("ािीुूेैोौृॉॅ", "a i i u u e ai o au ri o e".split()))
PRECOMPOSED = {"क़": "क", "ख़": "ख", "ग़": "ग", "ज़": "ज", "ड़": "ड", "ढ़": "ढ", "फ़": "फ"}
ABUSE = ("bhenchod", "behenchod", "bhainchan", "bhanchan", "bhenchan", "madarchod", "chutiya", "chutiye", "gaand", "gand",
         "lauda", "lavde", "lodu", "bhosdi", "randi", "fuck", "shit", "bitch", "nigga")


def is_devanagari(s):
    return any("ऀ" <= c <= "ॿ" for c in s)


def romanise(word):
    """Letter-by-letter Devanagari -> Roman, for when the LLM is unavailable. Plain but always readable; Devanagari must
    never reach the caption font, which has no Devanagari glyphs."""
    for k, v in PRECOMPOSED.items():
        word = word.replace(k, v + NUKTA_SIGN)
    out, i = [], 0
    while i < len(word):
        c = word[i]
        if c in CONS:
            r = CONS[c]
            if word[i + 1:i + 2] == NUKTA_SIGN:
                r, i = NUKTA.get(c, r), i + 1
            nxt = word[i + 1:i + 2]
            if nxt in MATRA:
                r, i = r + MATRA[nxt], i + 1
            elif nxt == VIRAMA:
                i += 1
            elif nxt:  # inherent 'a', except at the end of the word
                r += "a"
            out.append(r)
        elif c in VOWELS:
            out.append(VOWELS[c])
        elif c in "ँं":
            out.append("n")
        elif "०" <= c <= "९":
            out.append(str(ord(c) - 0x0966))
        elif not "ऀ" <= c <= "ॿ":
            out.append(c)
        i += 1
    return "".join(out) or "..."


def clean_word(w):
    """Caption-safe word: abuses masked (they cost reach and ads)."""
    bare = "".join(ch for ch in w.lower() if ch.isalpha())
    return w[0] + "***" if any(bare.startswith(a) if len(a) > 5 else bare == a for a in ABUSE) else w


def hinglish(words, heard=(), batch=40):
    """Devanagari words -> Roman Hinglish, 1:1 so every word keeps its timestamp. Never returns Devanagari.
    heard: other hearings of the same speech, to correct mis-heard words with."""
    hint = ""
    if heard and any(map(is_devanagari, words)):
        lines = "\n".join(f"- {t}" for t in [*heard, " ".join(words)])
        said = (llm(SAID_PROMPT.format(heard=lines), 0, "low", HINDI) or {}).get("said", "")
        hint = f"Several hearings together say it most likely was: {said}\nUse it for mis-heard words.\n" if said else ""
    out = list(words)
    for i in range(0, len(words), batch):
        chunk = words[i:i + batch]
        if not any(is_devanagari(w) for w in chunk):
            continue
        todo = dict(enumerate(chunk))  # numbered: a skipped word costs that word, not the whole batch
        for h in (hint, "", ""):
            items = json.dumps({str(k): w for k, w in todo.items()}, ensure_ascii=False)
            got = (llm(HINGLISH_PROMPT.format(n=len(todo), items=items, heard=h), 0, "low", HINDI) or {}).get("words")
            for k in list(todo) if isinstance(got, dict) else []:
                g = str(got.get(str(k)) or "").strip()
                if g and not is_devanagari(g) and (not h or sounds_like(todo[k], g)):
                    out[i + k] = g
                    del todo[k]
            if not todo:
                break
    return [clean_word(romanise(w) if is_devanagari(w) else w) for w in out]
