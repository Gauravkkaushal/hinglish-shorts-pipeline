"""python pipeline.py doctor: is everything installed? One line per check, with the fix next to what is missing."""
import importlib.util, re, shutil, subprocess, sys
from pathlib import Path

from .ff import LIB, ROOT, VID_EXT, env

PACKAGES = {"cv2": "opencv-python", "PIL": "pillow", "dotenv": "python-dotenv", "google_auth_oauthlib": "google-auth-oauthlib",
            **{m: m for m in ("requests", "numpy", "librosa", "soundfile", "tensorflow", "edge_tts", "rapidocr", "onnxruntime")}}


def out_of(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def key_works():
    import requests
    try:
        return requests.get("https://api.groq.com/openai/v1/models", timeout=20, headers={"Authorization": f"Bearer {env('GROQ_API_KEY')}"}).ok
    except requests.RequestException:
        return False


def checks():
    """-> [(what, ok, how to fix it, needed)]"""
    ff = re.search(r"ffmpeg version \D*(\d+)", out_of("ffmpeg", "-version"))
    missing = [p for m, p in PACKAGES.items() if not importlib.util.find_spec(m)]
    memes = sum(1 for d in LIB.glob("*") if d.name not in ("sfx", "music", "broll", "emoji", "images") and not d.name.startswith("_")
                for f in (d.iterdir() if d.is_dir() else ()) if f.suffix.lower() in VID_EXT and not f.name.startswith("_demo_"))
    return [
        ("Python 3.11 or newer", sys.version_info >= (3, 11), "winget install Python.Python.3.13", True),
        ("ffmpeg 7 or newer", bool(ff) and int(ff[1]) >= 7, "winget install Gyan.FFmpeg, then open a new terminal", True),
        ("yt-dlp", bool(shutil.which("yt-dlp")), "run setup.bat (or: pip install -U yt-dlp)", True),
        ("Deno (yt-dlp needs it for YouTube)", bool(shutil.which("deno")), "winget install DenoLand.Deno, then open a new terminal", True),
        ("Python packages", not missing, f"run setup.bat (missing: {', '.join(missing)})", True),
        ("models in assets/models", all((ROOT / "assets" / "models" / m).exists() for m in ("yamnet.tflite", "yunet.onnx")),
         "copy assets/models from the project", True),
        ("caption font Segoe UI Black", Path("C:/Windows/Fonts/seguibl.ttf").exists(), "it comes with Windows 10/11; other systems are not supported", True),
        ("sound effects", (LIB / "sfx" / "whoosh.wav").exists(), "python make_demo_assets.py", True),
        ("Groq key in .env", bool(env("GROQ_API_KEY")) and key_works(), "put GROQ_API_KEY=... in .env (free: console.groq.com/keys)", True),
        (f"reaction memes ({memes})", memes >= 10, "python fetch_memes.py (without them videos simply have no memes)", False),
        ("YouTube upload", (ROOT / "client_secret.json").exists(), "only for posting: README, \"Posting\"", False),
        ("Instagram upload", bool(env("IG_USER_ID") and env("IG_TOKEN")), "only for posting: README, \"Posting\"", False),
    ]


def doctor():
    """Prints the report. -> True when nothing needed is missing"""
    found = checks()
    for what, ok, fix, needed in found:
        print(f"  {'OK     ' if ok else 'MISSING' if needed else 'optional'}  {what}" + ("" if ok else f"  ->  {fix}"))
    ready = all(ok for _, ok, _, needed in found if needed)
    print("Ready." if ready else "Fix the MISSING lines, then run this again.")
    return ready
