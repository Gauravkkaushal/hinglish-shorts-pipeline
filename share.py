"""python share.py -> shorts-pipeline.zip for somebody else: the code, docs, models and fonts. Only what is listed here goes in, so
no key, token, sign-in, video or meme clip can slip through."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
PUBLIC = ["README.md", "LICENSE", "requirements.txt", "setup.bat", "run.bat", "share.py", "creators.txt", ".env.example", ".gitignore",
          "pipeline.py", "fetch_memes.py", "make_demo_assets.py", "test_pipeline.py", "shorts/*.py", "assets/models/*", "assets/fonts/*", "assets/NOTICE.md"]


def files():
    return sorted({f for p in PUBLIC for f in ROOT.glob(p) if f.is_file()})


if __name__ == "__main__":
    out = ROOT / "shorts-pipeline.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files():
            z.write(f, Path("shorts-pipeline") / f.relative_to(ROOT))
    print(f"{out.name}: {len(files())} files, {out.stat().st_size / 1e6:.1f} MB")
