"""Whose videos are used: the --creator options, else creators.txt."""
import random, re
from urllib.parse import quote_plus

from .ff import ROOT

LIST = ROOT / "creators.txt"


def listed(style, topic=""):
    """The creators of creators.txt whose section is "[style: label]"; only the sections about the topic, if any is."""
    sections, cur = [], None
    for line in LIST.read_text(encoding="utf-8").splitlines() if LIST.exists() else []:
        line = line.split("#")[0].strip()
        if m := re.fullmatch(r"\[(\w+):(.*)\]", line):
            cur = (m[1].lower(), m[2].lower().split(), [])
            sections.append(cur)
        elif line and cur:
            cur[2].append(line)
    mine = [s for s in sections if s[0] == style]
    about = [s for s in mine if any(w in topic.lower().replace("-", "") for w in s[1])]
    return [c for s in (about or (mine if style == "talk" else [])) for c in s[2]]


def channel(who):
    """@handle, link or plain name -> the channel's address. A name that several channels carry means the biggest of them."""
    from .sources import ytjson
    if who.startswith("http"):
        return re.sub(r"/(videos|shorts|featured|streams|playlists|about)?/?$", "", who)
    if who.startswith("@"):
        return f"https://www.youtube.com/{who}"
    squeeze = lambda t: re.sub(r"\W", "", t or "").lower()
    found = ytjson("--flat-playlist", "--playlist-end", "8", f"https://www.youtube.com/results?search_query={quote_plus(who)}&sp=EgIQAg%3D%3D")
    found = [e for e in found.get("entries") or [] if e.get("url")]
    same = [e for e in found if squeeze(who) in squeeze(e.get("title"))]
    if not same:
        raise RuntimeError(f"no channel is called '{who}'" + (f" (found: {', '.join(e['title'] for e in found[:3])})" if found else "")
                           + ". Give its @handle or link instead")
    best = max(same, key=lambda e: e.get("channel_follower_count") or 0)
    print(f"  creator '{who}' -> {best['title']} {best.get('uploader_id') or ''} ({best.get('channel_follower_count') or '?'} subscribers)")
    return best["url"]


def chosen(given, style, topic="", most=6):
    """-> channel addresses: exactly the given ones, else a few random ones from the list."""
    if given:
        return [channel(c) for c in given]
    names = listed(style, topic)
    names = random.sample(names, min(most, len(names)))
    if names:
        print("  creators (from creators.txt):", ", ".join(names))
    return [channel(c) for c in names]


def about(url, topic):
    """The address of a channel's own search for the topic."""
    return f"{url}/search?query={quote_plus(topic)}"
