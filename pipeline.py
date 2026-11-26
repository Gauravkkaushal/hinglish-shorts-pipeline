"""Hinglish Shorts pipeline. Every video passes shorts/qa.py or lands in out/rejected/ with the reasons.

  python pipeline.py viral --domain "GTA 6" --domain "Hindi podcast"   # everything from just a domain
  python pipeline.py auto --count 3                                    # best story of the creators in creators.txt
  python pipeline.py auto --creator @ZakirKhan                         # only from this creator (repeat for more)
  python pipeline.py comedy --topic "GTA 6" --reels 2                  # funniest 30-50s of several creators
  python pipeline.py montage --topic "GTA 6"                           # beat edit on an original, synthesised beat
  python pipeline.py clip --src podcast.mp4 --topic "funny moments"    # your own file(s)
  python pipeline.py story --topic "mummy ne sabzi lene bheja"         # AI script + AI voice
  python pipeline.py schedule --at 02:00 --count 2 --public           # every day by itself (daily: make + upload)
  python pipeline.py doctor                                            # is everything installed?
  python pipeline.py check out/x.mp4                                   # run the QA gate on any file
  python pipeline.py post --to youtube out/x.mp4                       # upload by API: private, or --public

Music: never added to auto / comedy / clip / story. montage uses a beat synthesised here, or --music with a file from
library/music/licensed that has a licence record. Anything with foreign music in it is refused.
"""
import argparse, json, sys, time
from pathlib import Path

try:
    import dotenv, numpy, requests
except ImportError as e:
    sys.exit(f"{e}. Run setup.bat once, then use run.bat (README, Install)")

from shorts import creators, qa, sources
from shorts.ai import check_clear, clean_word, heard, hinglish, llm, transcribe
from shorts.director import direct, library
from shorts.edit import build_pieces, finish, camera, keep_ranges, length, listen, meme_starts, out_time, render_edit, snap
from shorts.ff import IMG_EXT, JOBS, OUT, Stop, probe, safe_name


def fitted(words, dur, cuts, sound=None, until=None, hook=0.0):
    """The edit's pieces, inside 30-50s (a hook of this length comes on top): longer pauses and laughs when it gets too short,
    shorter ones and then the weakest meme dropped when it gets too long."""
    cuts = sorted(cuts, key=lambda c: -c.get("score", 0))
    while True:
        short = [c["t"] for c in cuts]
        cut = lambda pause, laugh: build_pieces(keep_ranges(words, dur, sound, pause, laugh, short, until), cuts, tail=0.35)
        for pace in ((0.3, 1.1), (0.5, 2.0), (1.0, 3.0), (dur, dur)):
            pieces = cut(*pace)
            if (total := hook + sum(length(p) for p in pieces)) >= qa.LO:
                break
        if total > qa.HI:
            pieces = cut(0.24, 0.8)
            total = hook + sum(length(p) for p in pieces)
        if total <= qa.HI or not cuts:
            return pieces
        cuts = cuts[:-1]


def story_end(words, until):
    """How many words belong to the story that ends around `until`: up to the longest pause near it, so that no half sentence
    hangs on at the end."""
    near = [(words[i + 1][0] - words[i][1] if i + 1 < len(words) else 1.0, i) for i in range(len(words)) if until - 2.5 <= words[i][1] <= until + 0.3]
    gap, i = max(near, default=(0.0, 0))
    return i + 1 if gap >= 0.15 else sum(w[0] < until for w in words)


def story_start(words):
    """The first word to keep: when the file opens inside a running sentence, the one after the longest pause of the first 2s."""
    if not words or words[0][0] > 0.15:  # it opens on a pause: that is a beginning
        return 0
    gap, i = max(((words[i][0] - words[i - 1][1], i) for i in range(1, min(6, len(words))) if words[i][0] <= 2.0), default=(0, 0))
    return i if gap >= 0.2 else 0


def make(name, src, job, title=""):
    """Talk / story source -> finished, checked video. -> (path, meta, problems)"""
    t0, words, dur = time.time(), src["words"], src["dur"]
    clip, why = src["kind"] == "clip", ""
    if clip:  # the finished 30-50s selection, heard properly
        talk = src.get("talk", src["video"])  # a part was already heard inside its whole story, which is understood better
        if talk == src["video"]:
            words = [w for s in transcribe(talk) for w in s["words"]]
        check_clear(talk)
        end = dur
        if until := src.get("until"):  # the sentence after the story is only in the file so that the story's laugh is
            n = story_end(words, until)
            end = min([w[0] - 0.1 for w in words[n:]] + [dur])
            words = words[story_start(words[:n]):n]
        why = sources.refuse(src["video"])
    print(f"[1/4] source: {dur:.1f}s, {len(words)} words")
    roman = hinglish([w[2] for w in words], heard(talk) if clip else ())
    plan = direct(words, dur, library())
    for c in plan["cuts"]:  # cut-in length: the meme's famous moment, max 3.5s
        c["d"] = 2.0 if c["path"].suffix.lower() in IMG_EXT else min(probe(c["path"])[0], 3.5)
    (job / "plan.json").write_text(json.dumps({**plan, "roman": roman}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[2/4] director: {len(plan['cuts'])} meme cut(s), {len(plan['emphasis'])} emphasis words")

    opening = []
    if clip and plan["hook"] and (src.get("part") or (1, 1))[0] == 1:
        i, j = plan["hook"]
        opening = [{"kind": "main", "a": snap(words[i][0] - 0.08), "b": snap(words[j][1] + 0.15), "fade": 0.1, "hook": True}]
    lead = sum(length(p) for p in opening)
    pieces = fitted(words, dur, plan["cuts"], listen(src["video"]), end, lead) if clip else build_pieces([[0.0, dur]], plan["cuts"])
    spans = camera(src["video"], dur) if clip else None
    render_edit(src["video"], opening + pieces, spans, job)
    total = lead + sum(length(p) for p in pieces)
    print(f"[3/4] edit: {dur:.1f}s -> {total:.1f}s, {len(opening)} hook, {sum(p['kind'] == 'main' for p in pieces)} cuts, {len(spans or [])} camera shot(s)")

    emph = set(plan["emphasis"])
    at = lambda t: lead + out_time(t, pieces)
    caps = [[at(s), at(s) + (e - s), roman[i], i in emph] for i, (s, e, _) in enumerate(words)]
    punch = [i for i in plan["punch"] if i < len(words)]
    stops = [lead + t for t in meme_starts(pieces)]
    ends = lambda i: min([at(words[i][1]) + 1.0] + [s for s in stops if s > at(words[i][0])])  # held through the laugh, not into a meme
    zooms = [[at(words[i][0]), ends(i), 0.2] for i in punch] + [[c[0], c[1] + 0.2, 0.07] for k, c in enumerate(caps) if c[3] and k not in punch]
    hits = [at(words[i][1]) for i in punch]
    if opening:
        i, j = plan["hook"]
        a = opening[0]["a"]
        caps += [[s - a, e - a, roman[k], k == j] for k, (s, e, _) in enumerate(words) if i <= k <= j]
        stops = [lead, *stops]
    OUT.mkdir(exist_ok=True)
    out = OUT / f"{name}.mp4"
    title = " ".join(map(clean_word, (title or src["title"] or plan["title"]).split()))
    k, n = src.get("part") or (1, 1)
    more = "" if n == 1 else f"Follow & subscribe for {'more' if k == n else f'Part {k + 1}'}"
    finish(job, total, caps, title + (rf"\NPART {k}/{n}" if n > 1 else ""), stops, zooms, out.resolve(),
           more.upper().replace(" FOR ", r"\NFOR "), hits)
    tags = plan["hashtags"][:5] or ["shorts", "hindi"]
    head = title + (f" (Part {k})" if n > 1 else "")
    meta = {"title": head, "caption": "\n".join(x for x in (head, more) if x) + "\n\n" + " ".join("#" + h for h in tags),
            "hashtags": tags, "credits": src["credits"], "name": title, "parts": n}
    out, problems = qa.deliver(out, meta, src["kind"], [job / "captions.ass"], length=total, extra=[*src.get("problems", ()), *([f"the source {why}"] if why else [])])
    print(f"[4/4] {time.time() - t0:.0f}s -> {out}")
    return out, meta, problems


DOMAIN_PROMPT = """Domain: "{domain}". We turn viral YouTube content of this domain into Indian Instagram Reels.
- "style": "talk" if the content is people talking (podcasts, interviews, stand-up, motivation, debates); "visual" if it is
  footage that works as a beat-synced music edit (video games, cars, bikes, travel, nature).
- "queries": 3 YouTube searches that find the most viral, recent videos of this domain for an Indian audience.
Return JSON {{"style": "talk|visual", "queries": ["...", "...", "..."]}}"""
TALK_WORDS = ("podcast", "interview", "stand-up", "standup", "comedy", "motivation", "speech", "debate", "talk")


def plan_domain(domain):
    """Domain -> (style, search queries). LLM decides; keyword rule if there is no LLM."""
    r = llm(DOMAIN_PROMPT.format(domain=domain)) or {}
    style = r.get("style") if r.get("style") in ("talk", "visual") else \
        ("talk" if any(w in domain.lower() for w in TALK_WORDS) else "visual")
    return style, [q for q in r.get("queries") or [] if isinstance(q, str)][:3] or [domain]


def viral(domain, count, given=()):
    style, queries = plan_domain(domain)
    who = creators.chosen(given, style if style == "talk" else "footage", domain)
    print(f"=== {domain}: {style} | {who or queries}")
    if style == "talk":
        return sources.auto_stories(who or queries, count, make)
    from shorts.montage import montage
    return [montage(domain, folder("montage", domain), creators=who, only=bool(given)) for _ in range(count)]


def folder(kind, topic):
    job = JOBS / f"{kind}_{safe_name(topic, 30)}"
    job.mkdir(parents=True, exist_ok=True)
    return job


def main():
    sys.stdout.reconfigure(errors="replace")  # Hindi text must not crash a run whose output goes to a file
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("auto")
    a.add_argument("--query", action="append", help="search text instead of creators")
    a.add_argument("--count", type=int, default=1)
    a.add_argument("--min-score", type=float, default=7, help="skip videos whose best story scores lower (1-10)")
    c = sub.add_parser("clip")
    c.add_argument("--src", action="append", required=True, help="file or URL; repeat for multiple sources")
    c.add_argument("--topic")
    s = sub.add_parser("story")
    s.add_argument("--topic")
    s.add_argument("--script", help="your own script file (Devanagari)")
    m = sub.add_parser("montage")
    m.add_argument("--topic", required=True)
    m.add_argument("--music", help="a file in library/music/licensed with a licence record; default: an original beat")
    m.add_argument("--length", type=float, default=32, help="seconds, 30-50")
    cm = sub.add_parser("comedy")
    cm.add_argument("--topic", required=True)
    cm.add_argument("--reels", type=int, default=2)
    v = sub.add_parser("viral")
    v.add_argument("--domain", action="append", required=True)
    v.add_argument("--count", type=int, default=1)
    k = sub.add_parser("check")
    k.add_argument("video", nargs="+")
    k.add_argument("--kind", default="clip", choices=["clip", "reel", "montage", "story"])
    pp = sub.add_parser("post")
    pp.add_argument("video")
    pp.add_argument("--to", required=True, choices=["youtube", "instagram"])
    pp.add_argument("--public", action="store_true", help="publish; without it YouTube gets a private upload")
    sub.add_parser("doctor")
    dl = sub.add_parser("daily", help="what the daily task runs: auto mode, then upload what passed")
    dl.add_argument("--count", type=int, default=2)
    dl.add_argument("--public", action="store_true", help="publish at once; without it YouTube gets private uploads")
    sc = sub.add_parser("schedule", help="run daily by itself every day (Windows Task Scheduler)")
    sc.add_argument("--at", default="02:00", help="time of day, 24h")
    sc.add_argument("--count", type=int, default=2)
    sc.add_argument("--public", action="store_true")
    sc.add_argument("--off", action="store_true", help="remove the daily task")
    for p in (a, m, cm, v, dl, sc):
        p.add_argument("--creator", action="append", help="take clips from this creator: @handle, link or name; repeat for several. Default: creators.txt")
    for p in (c, s):
        p.add_argument("--name")
    args = ap.parse_args()

    try:
        done = dispatch(args)
    except Stop as e:
        sys.exit(f"STOPPED: {e}")
    except RuntimeError as e:
        sys.exit(f"FAILED: {str(e)[-600:]}")
    for out, meta, problems in done or []:
        print(("READY  " if not problems else "REJECTED ") + str(out) + ("" if not problems else " -> " + "; ".join(problems)))
    if done is not None and not any(not p for _, _, p in done):
        sys.exit("no video passed")


def dispatch(args):
    if args.mode == "doctor":
        from shorts.doctor import doctor
        sys.exit(not doctor())
    if args.mode == "schedule":
        from shorts.schedule import schedule
        extra = [f"--count {args.count}", *(["--public"] if args.public else []), *(f'--creator "{c}"' for c in args.creator or [])]
        sys.exit(not schedule(args.at, extra, args.off))
    if args.mode == "daily":
        from shorts.schedule import daily
        try:
            made = sources.auto_stories(creators.chosen(args.creator, "talk"), args.count, make)
        except Stop as e:  # e.g. the daily limit: what was finished still goes out
            print(f"stopped: {e}")
            made = []
        daily(made, args.public)
        return None
    if args.mode == "post":
        from shorts.post import post
        post(args.video, args.to, args.public)
        return None
    if args.mode == "check":
        for f in args.video:
            if qa.finish_check(Path(f)) is not None:
                continue
            bad = qa.media_problems(Path(f), None) + (["music heard"] if args.kind != "montage" and qa.has_music(f) else []) + qa.eye_problems(Path(f), args.kind)
            print(f"{Path(f).name}: {'ok' if not bad else '; '.join(bad)}")
        return None
    if args.mode == "viral":
        out = []
        for d in args.domain:
            try:
                out += viral(d, args.count, args.creator)
            except RuntimeError as e:  # one domain failing must not stop the next
                print(f"=== {d} failed: {str(e)[-300:]}")
        return out
    if args.mode == "comedy":
        from shorts.comedy import build
        return build(args.topic, folder("comedy", args.topic), args.reels, creators=creators.chosen(args.creator, "funny", args.topic))
    if args.mode == "montage":
        from shorts.montage import montage
        return [montage(args.topic, folder("montage", args.topic), args.music, args.length,
                        creators=creators.chosen(args.creator, "footage", args.topic), only=bool(args.creator))]
    if args.mode == "auto":
        return sources.auto_stories((not args.creator and args.query) or creators.chosen(args.creator, "talk"), args.count, make, args.min_score)
    name = args.name or time.strftime("%Y%m%d_%H%M%S")
    job = JOBS / name
    job.mkdir(parents=True, exist_ok=True)
    if args.mode == "clip":
        return sources.own_clips(name, job, args.src, args.topic, make)
    return [make(name, sources.story(job, args.topic, args.script), job)]


if __name__ == "__main__":
    main()
