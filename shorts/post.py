"""Upload through the official APIs (YouTube Data API, Instagram Graph API). Only videos with a passing QA certificate
(qa.certified) go out, and each only once per site.

  python pipeline.py post --to youtube out/x.mp4              private, for a look in YouTube Studio
  python pipeline.py post --to youtube --public out/x.mp4     published
  python pipeline.py post --to instagram --public out/x.mp4   published (Instagram's API has no private upload)

Google sign-in happens in your own browser; this code only keeps the upload permission in youtube_token.json."""
import json, re, time
from pathlib import Path

import requests

from .ff import ROOT, VID_EXT, env
from .qa import certified

SECRET, TOKEN = ROOT / "client_secret.json", ROOT / "youtube_token.json"
SCOPE = ["https://www.googleapis.com/auth/youtube.upload"]
YOUTUBE = "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status"
GRAPH, RUPLOAD = "https://graph.facebook.com/v27.0", "https://rupload.facebook.com/ig-api-upload/v27.0"


def post_text(meta, site):
    """What goes out. YouTube: title <=100 chars + description; Instagram: caption <=2200 chars, max 5 hashtags."""
    tags = [t for t in meta.get("hashtags", []) if t][:5]
    body = (meta.get("caption") or meta.get("description") or meta.get("title", "")).split("\n\n#")[0].strip()
    credit = meta.get("credits") or ([meta["credit"]] if meta.get("credit") else [])
    credit = [c for c in credit if Path(c).suffix.lower() not in VID_EXT]  # a file name on this laptop credits nobody
    tail = (("\n\nCredits: " + "; ".join(credit)) if credit else "") + "\n\n" + " ".join("#" + t for t in tags)
    if site == "youtube":
        title = re.sub(r"[<>]", "", meta.get("title") or body.split("\n")[0])[:90].strip() + " #shorts"
        return title, re.sub(r"[<>]", "", body + tail).encode()[:4900].decode(errors="ignore")  # YouTube counts bytes
    return "", (body + tail)[:2200]


def ok(r):
    if not r.ok:
        try:
            why = r.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            why = r.text[:200]
        raise RuntimeError(f"upload refused ({r.status_code}): {why}")
    return r


def google_token(interactive=True):
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPE) if TOKEN.exists() else None
    if creds and not creds.valid:
        try:
            creds.refresh(Request())
        except RefreshError:  # Google ends it after 7 days while the app is in "Testing"
            creds = None
    if not creds and not interactive:  # at 2 a.m. nobody can sign in: say so instead of waiting for it
        raise RuntimeError("Google needs you to sign in again: run  run post --to youtube <video>  once by hand (README, Posting)")
    if not creds:
        if not SECRET.exists():
            raise RuntimeError("client_secret.json is missing (README: Upload, YouTube)")
        creds = InstalledAppFlow.from_client_secrets_file(str(SECRET), SCOPE).run_local_server(port=0)
    TOKEN.write_text(creds.to_json())
    return creds.token


def youtube(video, title, text, public, interactive=True):
    auth, size = {"Authorization": f"Bearer {google_token(interactive)}"}, video.stat().st_size
    r = ok(requests.post(YOUTUBE, timeout=60, headers={**auth, "X-Upload-Content-Length": str(size), "X-Upload-Content-Type": "video/mp4"},
                         json={"snippet": {"title": title, "description": text, "categoryId": "24"},  # 24 = Entertainment
                               "status": {"privacyStatus": "public" if public else "private", "selfDeclaredMadeForKids": False}}))
    url = r.headers["Location"]

    def put(span, data=b""):  # 308 = "send the rest", not a redirect
        return requests.put(url, headers={**auth, "Content-Type": "video/mp4", "Content-Range": f"bytes {span}/{size}"},
                            data=data, timeout=300, allow_redirects=False)

    for attempt in range(6):
        sent = 0
        try:
            if attempt:  # ask what arrived: a lost answer must not become a second copy of the video
                time.sleep(2 ** attempt)
                r = put("*")
                if r.status_code == 308 and "Range" in r.headers:
                    sent = int(r.headers["Range"].split("-")[1]) + 1
            if not attempt or r.status_code == 308:
                with open(video, "rb") as f:
                    f.seek(sent)
                    r = put(f"{sent}-{size - 1}", f)
        except (requests.ConnectionError, requests.Timeout):
            continue
        if r.status_code != 308 and r.status_code < 500:
            v = ok(r).json()
            return v["id"], f"https://youtube.com/shorts/{v['id']} ({v['status']['privacyStatus']})"
    raise requests.ConnectionError("YouTube did not answer")


def instagram(video, caption):
    user, token = env("IG_USER_ID"), env("IG_TOKEN")
    if not user or not token:
        raise RuntimeError("IG_USER_ID / IG_TOKEN are missing in .env (README: Upload, Instagram)")
    auth = {"Authorization": f"OAuth {token}"}
    box = ok(requests.post(f"{GRAPH}/{user}/media", headers=auth, timeout=60,
                           data={"media_type": "REELS", "upload_type": "resumable", "caption": caption})).json()["id"]
    with open(video, "rb") as f:
        ok(requests.post(f"{RUPLOAD}/{box}", headers={**auth, "offset": "0", "file_size": str(video.stat().st_size)}, data=f, timeout=600))
    for _ in range(20):
        s = ok(requests.get(f"{GRAPH}/{box}", params={"fields": "status_code,status"}, headers=auth, timeout=60)).json()
        if s["status_code"] != "IN_PROGRESS":
            break
        time.sleep(15)
    if s["status_code"] != "FINISHED":
        raise RuntimeError(f"Instagram did not accept the video: {s['status_code']} {s.get('status', '')}")
    media = ok(requests.post(f"{GRAPH}/{user}/media_publish", headers=auth, data={"creation_id": box}, timeout=120)).json()["id"]
    return media, f"published (id {media})"


def post(video, site, public=False, interactive=True):
    video = Path(video).resolve()
    meta = certified(video)
    if site in meta.get("posted", {}):
        raise RuntimeError(f"{video.name} is already on {site} ({meta['posted'][site]})")
    if site == "instagram" and not public:
        raise RuntimeError("Instagram's API publishes straight away, there is no private upload. Add --public to do it")
    for k in range(1, meta.get("parts", 1) + 1) if meta.get("parts", 1) > 1 else ():  # no "Part 2" that never comes
        certified(video.with_name(re.sub(r"_part\d+$", f"_part{k}", video.stem) + video.suffix))
    title, text = post_text(meta, site)
    try:
        media, where = youtube(video, title, text, public, interactive) if site == "youtube" else instagram(video, text)
    except requests.RequestException as e:
        raise RuntimeError(f"connection to {site} broke ({type(e).__name__}). Look on {site} whether the video arrived before trying again") from None
    meta.setdefault("posted", {})[site] = media
    video.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {site}: {where}")
