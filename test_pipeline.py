import json, types
from pathlib import Path

import numpy as np
import pytest

import pipeline
from fetch_memes import loudest_start
from shorts import ai, captions, comedy, creators, director, edit, eyes, ff, montage, post, qa, rights, sources

W = [[0.0, 0.3, "भाई"], [0.3, 0.6, "हँस"], [0.6, 0.9, "हँस"], [0.9, 1.2, "के"], [2.0, 2.4, "तभी"], [5.0, 5.3, "रो"]]


# ---------- music policy ----------
def test_no_code_path_adds_music_outside_rights():
    src = {p.name: p.read_text(encoding="utf-8") for p in list(Path("shorts").glob("*.py")) + [Path("pipeline.py")]}
    for name in ("edit.py", "reel.py", "comedy.py", "sources.py", "director.py"):  # talk/comedy never touch a music file
        assert '"music"' not in src[name].replace('{"sfx", "music"', ""), name
    assert not any("get_song" in t or "songs.txt" in t or "cookies-from-browser" in t for t in src.values())


def test_unlicensed_track_is_refused(tmp_path):
    f = tmp_path / "hit.mp3"
    f.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="no licence"):
        rights.licence_of(f)
    Path(f"{f}.license.json").write_text(json.dumps({"source": "Pixabay", "license": "", "url": "u"}))
    with pytest.raises(RuntimeError, match="license"):
        rights.licence_of(f)
    Path(f"{f}.license.json").write_text(json.dumps({"source": "Pixabay", "license": "Pixabay Content License", "url": "u"}))
    assert rights.licence_of(f)["source"] == "Pixabay"


def test_music_verdict():
    assert rights.is_musical([0.9] * 10) and rights.is_musical([0.0] * 30 + [0.6, 0.6])
    assert not rights.is_musical([0.05] * 20 + [0.5]) and not rights.is_musical([])


def test_synthesised_beat_is_heard_as_music_and_has_exact_beats(tmp_path):
    p = rights.original_beat(tmp_path / "b.wav", 12, bpm=120, seed=1)
    assert p["beats"][:3] == [0, 0.5, 1.0] and abs(ff.probe(tmp_path / "b.wav")[0] - 12) < 0.05
    assert p["strong"][0] == 2.0 and rights.has_music(tmp_path / "b.wav")


# ---------- transcription / Hinglish ----------
def test_stretched_words_are_found_and_capped():
    words = [[0, 0.9, "तो"], [0.9, 6.7, "तुम"], [6.7, 7.0, "बना"]]
    assert ai.holes(words) == [(0.9, 6.7)]
    assert max(b - a for a, b, _ in ai.tidy(words)) <= ai.MAX_WORD
    t = ai.tidy([[1.0, 2.0, "b"], [0.5, 1.6, "a"]])
    assert [w[2] for w in t] == ["a", "b"] and t[0][1] <= t[1][0]


def test_captions_never_devanagari_even_without_llm(monkeypatch):
    monkeypatch.setattr(ai, "llm", lambda *a, **k: None)
    out = ai.hinglish(["क्या", "भाई", "देख", "OK", "ज़िंदगी", "१२"])
    assert out[0] == "kya" and out[2:4] == ["dekh", "OK"] and not any(map(ai.is_devanagari, out))


def test_abuse_masked_but_not_lookalikes():
    assert ai.clean_word("bhainchan") == "b***" and [ai.clean_word(w) for w in ("gandhi", "shirt", "random")] == ["gandhi", "shirt", "random"]


# ---------- director ----------
def test_rule_plan_keeps_matras_and_cuts_after_word():
    cuts = director.rule_plan(W)["cuts"]
    assert [c["tag"] for c in cuts] == ["laugh", "laugh", "shock", "cry"] and cuts[0]["t"] == 0.6


def test_from_llm_maps_word_indexes():
    r = {"cuts": [{"after_word": 4, "tag": "shock", "score": 9}, {"after_word": 99, "tag": "laugh", "score": 10},
                  {"after_word": "x", "tag": "laugh", "score": "8/10"}], "emphasis": [1, "bad", 50, 1], "title": "t"}
    plan = director.from_llm(r, W)
    assert [(c["t"], c["tag"], c["score"]) for c in plan["cuts"]] == [(2.4, "shock", 9.0)] and plan["emphasis"] == [1]


def test_clean_plan_keeps_only_strong_spaced_cuts():
    lib = {"laugh": [Path("a.mp4"), Path("b.mp4")]}
    cuts = [{"t": t, "tag": g, "score": s, **f} for t, g, s, f in
            [(1.0, "laugh", 9, {}), (3, "laugh", 9, {}), (5, "laugh", 9, {}), (9, "nope", 9, {}), (9.5, "laugh", 6, {}),
             (10, "laugh", 8, {"file": "b"}), (12, "laugh", 10, {})]]
    out = director.clean_plan({"cuts": cuts, "emphasis": [], "title": "x", "hashtags": ["#a b"]}, lib, dur=12)
    assert [c["t"] for c in out["cuts"]] == [3, 10] and out["cuts"][1]["path"] == Path("b.mp4") and out["hashtags"] == ["ab"]


# ---------- timeline ----------
def sounds(dur, *loud, talk=()):
    """Silence with loud stretches; talk: the stretches in which somebody speaks."""
    db, sp = np.full(int(dur * 100) + 1, -60.0), [0.0] * (int(dur * 4) + 2)
    for x, y in loud:
        db[int(x * 100):int(y * 100)] = -20.0
    for x, y in talk:
        sp[int(x * 4):int(y * 4) + 1] = [0.9] * (int(y * 4) + 1 - int(x * 4))
    return db, sp


def test_cuts_silence_keeps_the_laugh_and_never_cuts_talk():
    w = [[0.5, 1.0, "a"], [1.2, 1.5, "b"], [4.5, 5.0, "c"]]
    quiet = edit.keep_ranges(w, 6, sounds(6, (0.5, 1.5), (4.5, 5.0)))
    assert len(quiet) == 2 and quiet[0][1] < 1.8 and quiet[1][0] > 4.3
    laugh = edit.keep_ranges(w, 6, sounds(6, (0.5, 4.4), (4.5, 5.0)))
    assert len(laugh) == 2 and 2.5 <= laugh[0][1] <= 2.7 and laugh[0][2] == 0.3  # 1.1s of laughter, faded out
    meme = edit.keep_ranges(w, 6, sounds(6, (0.5, 4.4), (4.5, 5.0)), short=[1.5])
    assert meme[0][1] < 2.0
    missed = edit.keep_ranges(w, 6, sounds(6, (0.5, 5.0), talk=[(0.5, 5.0)]))
    assert len(missed) == 1
    late = edit.keep_ranges(w, 6, sounds(6, (0.5, 1.5), (4.2, 5.0), talk=[(4.0, 5.0)]))
    assert late[1][0] <= 4.22  # the word really began earlier than the recogniser said
    end = edit.keep_ranges(w, 9, sounds(9, (0.5, 1.5), (4.5, 8.8)))
    assert 6.0 <= end[-1][1] <= 6.2 and end[-1][2] > 0.02
    assert edit.keep_ranges(w, 9, sounds(9, (0.5, 1.5), (4.5, 8.8)), until=5.5)[-1][1] <= 5.5


def test_pieces_and_time_mapping_with_meme():
    pieces = edit.build_pieces([[0.0, 2.0], [3.0, 5.0]], [{"t": 1.0, "path": Path("m.mp4"), "d": 2.0}])
    assert [p["kind"] for p in pieces] == ["main", "meme", "main", "main"] and pieces[0]["b"] == edit.snap(1.1)
    assert edit.out_time(0.5, pieces) == 0.5 and abs(edit.out_time(1.5, pieces) - 3.5) < 1e-9
    assert abs(edit.out_time(3.5, pieces) - 4.5) < 1e-9 and edit.out_time(2.5, pieces) == edit.out_time(3.0, pieces)
    assert edit.meme_starts(pieces) == [pieces[0]["b"]]


def test_the_camera_follows_and_ignores_flicker():
    still = [(t / 4, (0.5, 0.3, 0.2)) for t in range(40)]
    still[7] = (1.75, None)
    (a, b, path), = edit.follow(still, 0.0, 10.0)
    assert (a, b) == (0.0, 10.0) and path[0] == 1.0 and len({x for _, x in path[2]}) == 1
    walk = edit.follow([(t / 4, (0.3 + t * 0.01, 0.3, 0.1)) for t in range(40)], 0.0, 10.0)
    assert len(walk) == 1 and walk[0][2][0] == 1.35 and walk[0][2][2][-1][1] > walk[0][2][2][0][1] + 0.2
    two = edit.follow([(t / 4, (0.2 if t < 20 else 0.8, 0.3, 0.2)) for t in range(40)], 0.0, 10.0)
    assert len(two) == 2 and two[0][1] == two[1][0] and abs(two[0][1] - 5.0) < 0.6
    assert edit.follow([(t / 4, None) for t in range(8)], 0.0, 2.0)[0][2] is None and edit.follow([], 0, 1) == [[0, 1, None]]


def test_pan_stays_inside_the_picture():
    e = edit.pan([(0.0, 0.02), (1.0, 0.5), (2.0, 0.99)], 0.5, 0.16)
    assert "if(lt(t,0.50)" in e and "0.0000" in e and "0.6800" in e
    assert edit.pan([(0.0, 0.5), (1.0, 0.5)], 0.0, 0.16) == "0.3400"


def test_creators_from_the_list_or_as_given(tmp_path, monkeypatch):
    f = tmp_path / "creators.txt"
    f.write_text("# x\n[talk: podcast]\n@a # note\nhttps://www.youtube.com/@b/videos\n[talk: standup comedy]\n@c\n[footage: gta]\n@d\n")
    monkeypatch.setattr(creators, "LIST", f)
    assert creators.listed("talk") == ["@a", "https://www.youtube.com/@b/videos", "@c"]
    assert creators.listed("talk", "Hindi stand-up comedy") == ["@c"] and creators.listed("talk", "motivation") == creators.listed("talk")
    assert creators.listed("footage", "GTA 6") == ["@d"] and creators.listed("footage", "Supercars") == []
    assert sorted(creators.chosen((), "talk", "podcast")) == ["https://www.youtube.com/@a", "https://www.youtube.com/@b"]
    assert creators.chosen(["@z"], "talk") == ["https://www.youtube.com/@z"]
    monkeypatch.setattr(creators, "LIST", tmp_path / "none.txt")
    assert creators.chosen((), "talk") == []


def test_a_name_means_the_biggest_channel_of_that_name(monkeypatch):
    found = [{"title": "Prakhar Gupta", "url": "small", "channel_follower_count": 310}, {"title": "Prakhar Gupta", "url": "big", "channel_follower_count": 2900000},
             {"title": "Someone Else", "url": "huge", "channel_follower_count": 9e9}]
    monkeypatch.setattr(sources, "ytjson", lambda *a: {"entries": found})
    assert creators.channel("prakhar gupta") == "big"
    with pytest.raises(RuntimeError, match="no channel"):
        creators.channel("Nobody Atall")


def test_creators_take_turns(monkeypatch):
    v = lambda i, n: {"id": i, "view_count": n, "duration": 900, "title": "t"}
    pages = {"https://www.youtube.com/@a/videos": [v("a1", 9), v("a2", 99)], "https://www.youtube.com/@b/videos": [v("b1", 1), v("seen", 5)]}
    monkeypatch.setattr(sources, "ytjson", lambda *a: {"entries": pages[a[-1]]})
    assert [e["id"] for e in sources.discover(["https://www.youtube.com/@a", "https://www.youtube.com/@b"], {"seen"})] == ["a2", "b1", "a1"]


def test_hook_is_a_late_short_line_found_by_its_words():
    said = "pehle kuch aur baat thi fir usne bola ki sabka sarvanaash hoga dekh lena tum log".split()
    w = [[float(t), t + 0.9, x] for t, x in enumerate(said)]
    assert director.hook_of("sabka sarvanaash hoga", w) == (9, 11) and director.hook_of("Sabka sarvnash hoga!", w) == (9, 11)
    for bad in ("kuch aur baat", "fir usne bola ki sabka sarvanaash hoga dekh", "ye line nahi hai", "hoga", "", None, 5):
        assert director.hook_of(bad, w) is None


def test_a_meme_leaves_no_laugh_behind_even_if_the_next_word_came_early():
    w = [[0.5, 1.0, "a"], [1.05, 1.3, "early"], [4.5, 5.0, "c"]]
    r = edit.keep_ranges(w, 6, sounds(6, (0.5, 4.4), (4.5, 5.0)), short=[1.0])
    assert r[0][1] < 1.8


# ---------- captions ----------
def test_captions_chunks_highlight_and_stop_at_memes():
    words = [w + [i == 1] for i, w in enumerate(W)]
    caps = captions.caption_events(words)
    assert len(caps) == len(words) and captions.GREEN in caps[0][2] and captions.YELLOW in caps[0][2]
    assert abs(captions.caption_events(words[:4], stops=[1.2])[-1][1] - (1.2 - 0.08)) < 1e-9
    hook = captions.caption_events([[0, 1, "hook", False], [1.1, 1.4, "start", False]], stops=[1.05])
    assert not any("HOOK" in t and "START" in t for _, _, t in hook)
    messy = [[0, .5, "a", False], [.4, .45, "b", False], [.3, .9, "c", False], [.3, .9, "c", False]]
    ev = captions.caption_events(messy)
    assert all(a[1] <= b[0] and a[0] < a[1] for a, b in zip(ev, ev[1:]))


# ---------- sources ----------
def test_best_window_around_replay_peak():
    hm = [{"start_time": t, "end_time": t + 10, "value": v} for t, v in [(0, 1.0), (100, 0.4), (500, 0.9), (990, 1.0)]]
    assert sources.best_window({"duration": 1000, "heatmap": hm}) == (355.0, 715.0, 505.0)
    assert sources.best_window({"duration": 1000}) == (60.0, 420.0, 210.0)  # no replay data
    assert sources.best_window({"duration": 90}) is None


def test_length_is_30_to_50_on_sentence_boundaries():
    trans = [{"start": float(s), "end": float(s + 8)} for s in range(0, 120, 8)]
    for a, b in [(40, 48), (0, 112), (112, 120)]:
        x, y = sources.fit_length(trans, a, b)
        assert 30 <= y - x <= 50 and y in [s["end"] for s in trans]


def test_bot_check_stops_the_run(monkeypatch):
    class R:
        returncode, stderr, stdout = 1, "Sign in to confirm you're not a bot", ""
    monkeypatch.setattr(ff.subprocess, "run", lambda *a, **k: R)
    with pytest.raises(ff.YouTubeBlocked):
        ff.run(["yt-dlp", "x"])
    assert not issubclass(ff.YouTubeBlocked, RuntimeError)  # so "skip this video" handlers cannot swallow it


# ---------- montage ----------
def test_cut_points_on_beats_and_frame_grid():
    pts = montage.cut_points([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], length=3.2)
    assert pts == [0.0, 1.0, 2.0, 3.2] and all(abs(p * 30 - round(p * 30)) < 1e-9 for p in pts)


def test_assign_never_reuses_a_shot():
    shots = [{"src": i % 3, "a": 100 * i, "b": 100 * i + 60, "score": i / 10, "cx": 0.5} for i in range(6)]
    out = montage.assign(shots, [1.0] * 6, hits={2})
    assert len({(p["src"], p["a"]) for p in out}) == 6 and out[2]["m"] == 48 and out[0]["m"] == 30
    assert out[2]["shot"]["score"] == 0.5  # the best shot lands on the hit
    with pytest.raises(RuntimeError, match="only 6"):
        montage.assign(shots, [1.0] * 7)


def test_split_shots_separates_pieces_and_penalises_dark_and_empty():
    row = lambda bright, edges: [[3, 0.5, bright, 150, edges]] * 60
    cut = [[80, 0.5, 120, 150, 0.1]]
    s = montage.split_shots(np.array(row(120, 0.1) + cut + row(10, 0.1) + cut + row(150, 0.005)), 0)
    assert len(s) == 3 and s[0]["score"] > s[1]["score"] and s[0]["score"] > s[2]["score"]
    long = montage.split_shots(np.array([[3, 0.5, 120, 150, 0.1]] * 300), 0)
    assert all(b["a"] - a["b"] >= 36 for a, b in zip(long, long[1:]))  # two pieces of one take are far apart


def test_overlays_need_to_stay_put(monkeypatch):
    cam, hero = (0.88, 0.05, 0.08, 0.15), (0.45, 0.3, 0.1, 0.2)
    monkeypatch.setattr(eyes, "faces", lambda f: f["faces"])
    monkeypatch.setattr(eyes, "texts", lambda f: f["text"])
    fr = lambda k, **kw: {"faces": [hero] if k % 2 else [], "text": [(f"SIGN{k}", (0.1 * k, 0.2, 0.1 * k + 0.1, 0.25))], **kw}
    assert eyes.overlays([fr(k) for k in range(6)]) == []
    assert eyes.overlays([fr(k, faces=[cam]) for k in range(6)]) == ["facecam"]
    mark = ("CHAMPS NETWORK", (0.8, 0.9, 0.95, 0.95))
    assert eyes.overlays([fr(k, text=[mark if k else ("CHAMPS NETWOR!", mark[1])]) for k in range(6)]) == ["watermark"]


def test_frame_facts(monkeypatch):
    say = lambda *t: monkeypatch.setattr(eyes, "texts", lambda f: list(t))
    say(("I can't tell, but this is the other half", (0.3, 0.85, 0.7, 0.9)))
    assert eyes.facts(None) == ["subtitles"]
    say(("MAP", (0, 0, .05, .02)), ("BRIEF", (0, 0, .05, .02)))
    assert eyes.facts(None) == ["menu"]
    say(("grand", (0.4, 0.3, 0.6, 0.5)), ("theft", (0.4, 0.5, 0.6, 0.7)))
    assert eyes.facts(None) == ["title"]
    say(("$37,621", (0.9, 0.0, 0.98, 0.04)))
    assert eyes.facts(None) == []


def test_text_never_covers_the_video(tmp_path):
    montage.text_ass(tmp_path / "t.ass", "One two three four five six", "Send this", 32)
    ev = qa.ass_events(tmp_path / "t.ass")
    assert ev[0][3].split() == ["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX"] and ev[0][1] - ev[0][0] > 3
    box_top, box_bottom = (1920 - montage.BOX_H) // 2, (1920 + montage.BOX_H) // 2
    t = (tmp_path / "t.ass").read_text(encoding="utf-8")
    hook, end = (next(l for l in t.splitlines() if l.startswith(f"Style: {n},")).split(",") for n in ("Hook", "End"))
    assert hook[18] == "2" and 1920 - int(hook[21]) < box_top and end[18] == "8" and int(end[21]) > box_bottom


def test_round_robin_and_used_shots():
    pools = [[{"id": "t1"}, {"id": "t2"}], [{"id": "f1"}], [{"id": "t1"}, {"id": "i1"}]]
    assert [e["id"] for e in montage.pick_round_robin(pools, 4)] == ["t1", "f1", "t2", "i1"]
    assert montage.was_used([["w", 100, 160, "s"]], "w", 150, 200) and not montage.was_used([["w", 100, 160, "s"]], "w", 160, 220)


def test_peak_windows_distinct_moments():
    hm = [{"start_time": s, "end_time": s + 2, "value": v} for s, v in [(50, 1.0), (55, 0.9), (120, 0.8), (1, 1.0)]]
    assert montage.peak_windows({"duration": 200, "heatmap": hm}, k=2) == [(46.0, 56.0), (116.0, 126.0)]


# ---------- comedy / memes ----------
def test_loudest_moments_spread_out_and_skip_intro():
    s = np.zeros(1000)
    s[10], s[300], s[320], s[700] = 9, 8, 7, 6
    assert comedy.top_peaks(s, 2, min_dist=150) == [300, 700]
    assert loudest_start(np.array([0, 0, 1, 9, 9, 9, 1, 0], dtype=float), 3) == 3


# ---------- QA gate ----------
def ass(tmp_path, *rows):
    f = tmp_path / "c.ass"
    f.write_text(captions.HEAD + "\n".join(f"Dialogue: 0,{captions.ts(a)},{captions.ts(b)},Cap,,0,0,0,,{t}" for a, b, t in rows), encoding="utf-8")
    return qa.ass_problems(f)


def test_qa_caption_faults(tmp_path):
    assert ass(tmp_path, (0, 0.5, "BHAI KYA"), (0.5, 1.0, r"{\c&H0&}SCENE{\r} HAI")) == []
    assert "Devanagari" in ass(tmp_path, (0, 0.5, "देख"))[0]
    assert "hangs" in ass(tmp_path, (0, 6, "TO TUM BANA"))[0]
    assert "at once" in ass(tmp_path, (0, 1, "A"), (0.5, 1.2, "B"))[0]
    assert "wide" in ass(tmp_path, (0, 1, "W" * 30))[0]


def test_upload_needs_a_valid_certificate(tmp_path):
    v = tmp_path / "x.mp4"
    v.write_bytes(b"abc")
    with pytest.raises(RuntimeError, match="not passed"):
        qa.certified(v)
    import hashlib
    cert = {"passed": True, "problems": [], "sha256": hashlib.sha256(b"abc").hexdigest()}
    v.with_suffix(".json").write_text(json.dumps({"title": "t", "qa": cert}))
    assert qa.certified(v)["title"] == "t"
    v.write_bytes(b"abcd")
    with pytest.raises(RuntimeError, match="changed"):
        qa.certified(v)


def test_post_text_limits():
    meta = {"title": "T<" + "x" * 200, "caption": "line1\nline2?\n\n#a #b", "hashtags": list("abcdefg"), "credits": ["Chan - url"]}
    title, d = post.post_text(meta, "youtube")
    assert len(title) <= 100 and "<" not in title and "Credits: Chan - url" in d and d.count("#") == 5
    assert post.post_text(meta, "instagram")[1].count("#") == 5
    meta["credits"] = ["story_1.mp4"]  # a local file name is not a credit
    assert "story_1" not in post.post_text(meta, "youtube")[1]


@pytest.fixture
def api(monkeypatch, tmp_path):
    """Fake YouTube + Instagram on localhost, and a certified video."""
    import hashlib, threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    S = types.SimpleNamespace(got=b"", calls=[], drop=0, lose_answer=False, ig_status=["IN_PROGRESS", "FINISHED"], refuse=False)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def reply(self, code, body=None, **head):
            self.send_response(code)
            data = json.dumps(body or {}).encode()
            for k, v in {**head, "Content-Length": len(data)}.items():
                self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(data)

        def handle_any(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            S.calls.append((self.command, self.path, self.headers, body))
            video = {"id": "VID", "status": {"privacyStatus": "private"}}
            if S.refuse:
                return self.reply(403, {"error": {"message": "quota exceeded"}})
            if self.path.startswith("/yt"):
                return self.reply(200, Location=f"http://127.0.0.1:{srv.server_port}/session")
            if self.path == "/session":
                span, total = self.headers["Content-Range"][6:].split("/")
                if span != "*":
                    if S.drop:
                        S.got, S.drop = body[:S.drop], 0
                        self.connection.close()
                        return
                    assert int(span.split("-")[0]) == len(S.got)
                    S.got += body
                    if S.lose_answer:
                        S.lose_answer = False
                        self.connection.close()
                        return
                if len(S.got) == int(total):
                    return self.reply(201, video)
                return self.reply(308, **({"Range": f"bytes=0-{len(S.got) - 1}"} if S.got else {}))
            if self.path.startswith("/up/BOX"):
                S.got = body
                return self.reply(200, {"success": True})
            if self.command == "GET":
                return self.reply(200, {"status_code": S.ig_status.pop(0)})
            self.reply(200, {"id": "MEDIA" if "publish" in self.path else "BOX"})
        do_GET = do_POST = do_PUT = handle_any

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    for k, v in {"YOUTUBE": base + "/yt", "GRAPH": base + "/g", "RUPLOAD": base + "/up", "google_token": lambda interactive=True: "tok"}.items():
        monkeypatch.setattr(post, k, v)
    monkeypatch.setattr(post.time, "sleep", lambda s: None)
    monkeypatch.setenv("IG_USER_ID", "7")
    monkeypatch.setenv("IG_TOKEN", "sekret")
    S.video, S.data = tmp_path / "x.mp4", bytes(range(256)) * 4000
    S.video.write_bytes(S.data)
    S.record = S.video.with_suffix(".json")
    S.record.write_text(json.dumps({"title": "Mummy <3", "caption": "c\n\n#a", "hashtags": ["a"], "qa": {
        "passed": True, "sha256": hashlib.sha256(S.data).hexdigest()}}))
    yield S
    srv.shutdown()


@pytest.mark.parametrize("trouble", [{}, {"drop": 100_000}, {"drop": 5}, {"lose_answer": True}])
def test_youtube_upload_survives_a_broken_line(api, trouble):
    vars(api).update(trouble)
    post.post(api.video, "youtube")
    sent = json.loads(api.calls[0][3])
    assert api.got == api.data and sent["status"]["privacyStatus"] == "private" and "<" not in sent["snippet"]["title"]
    assert json.loads(api.record.read_text())["posted"] == {"youtube": "VID"}
    with pytest.raises(RuntimeError, match="already"):
        post.post(api.video, "youtube")
    assert sum(c[0] == "POST" for c in api.calls) == 1


def test_youtube_public_and_refusal(api):
    api.refuse = True
    with pytest.raises(RuntimeError, match="quota exceeded"):
        post.post(api.video, "youtube", public=True)
    assert json.loads(api.calls[0][3])["status"]["privacyStatus"] == "public" and "posted" not in json.loads(api.record.read_text())


def test_instagram_upload(api):
    with pytest.raises(RuntimeError, match="--public"):
        post.post(api.video, "instagram")
    assert not api.calls
    post.post(api.video, "instagram", public=True)
    assert api.got == api.data and api.calls[-1][1].endswith("/7/media_publish") and b"creation_id=BOX" in api.calls[-1][3]
    assert all("sekret" not in c[1] for c in api.calls)  # the token never goes into a URL


def test_instagram_rejected_video_is_not_published(api):
    api.ig_status = ["ERROR"]
    with pytest.raises(RuntimeError, match="did not accept"):
        post.post(api.video, "instagram", public=True)
    assert not any("media_publish" in c[1] for c in api.calls)


def test_viral_domain_fallback_without_llm(monkeypatch):
    monkeypatch.setattr(pipeline, "llm", lambda *a, **k: None)
    assert pipeline.plan_domain("Hindi podcast") == ("talk", ["Hindi podcast"]) and pipeline.plan_domain("Supercars")[0] == "visual"
    assert montage.topic_word("Supercars") == "supercar" and montage.topic_word("GTA 6") == "gta"


def test_vet_refuses_lookalikes_and_one_video_taking_over(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    looks = {k: rng.integers(0, 256, (9, 16, 3), np.uint8).repeat(20, 0).repeat(20, 1) for k in range(12)}
    monkeypatch.setattr(eyes, "facts", lambda img: [])
    monkeypatch.setattr(eyes, "grab_frame", lambda f, t, width=960: looks[int(t) // 10 if "same" not in f.name else 0])
    files = [tmp_path / "win_same_0.mp4", tmp_path / "win_many_0.mp4", tmp_path / "win_other_0.mp4"]
    shots = [{"src": s, "a": k * 300, "b": k * 300 + 30, "score": 1 - k / 100, "cx": 0.5} for s in (0, 1) for k in range(1, 9)]
    good = montage.vet(shots, files, tmp_path, 12)
    assert sum(g["src"] == 0 for g in good) == 1      # the same picture only once
    assert sum(g["src"] == 1 for g in good) <= 4      # at most a third from one video


def test_edit_stays_inside_30_to_50_seconds():
    words = [[t, t + 0.5, "x"] for t in np.arange(0.2, 33, 1.0)]  # half of it is pauses
    assert sum(map(edit.length, pipeline.fitted(words, 33.5, []))) >= 30
    words = [[t, t + 0.5, "x"] for t in np.arange(0.2, 47.5, 0.6)]
    cuts = [{"t": 10.0, "path": Path("a"), "d": 3.0, "score": 9}, {"t": 20.0, "path": Path("b"), "d": 3.0, "score": 7}]
    got = pipeline.fitted(words, 48, cuts)
    assert sum(map(edit.length, got)) <= 50 and not any(p.get("path") == Path("b") for p in got)


def test_a_story_ends_at_the_pause():
    w = [[0, 1, "main"], [1.1, 2, "mar"], [2.1, 3, "gaya"], [4.4, 4.7, "ki"], [4.75, 5.0, "meri"], [5.1, 5.5, "maa"]]
    assert pipeline.story_end(w, 5.0) == 3 and pipeline.story_end(w[:3], 3.2) == 3
    assert pipeline.story_end([[t, t + 1.0, "x"] for t in range(9)], 4.5) == 5


def test_a_story_does_not_open_inside_a_sentence():
    w = [[0.0, 0.3, "toh"], [0.3, 0.6, "leta"], [0.6, 0.9, "hoon"], [1.4, 1.7, "par"], [1.7, 2.1, "tumhari"]]
    assert pipeline.story_start(w) == 3
    assert pipeline.story_start([[0.3, 0.5, "ek"], [1.0, 1.2, "baar"]]) == 0 and pipeline.story_start([]) == 0


def test_what_is_shared_holds_no_secret_and_no_video():
    import share
    names = [f.name for f in share.files()]
    assert {"README.md", "pipeline.py", "setup.bat", "creators.txt", "yamnet.tflite", "doctor.py"} <= set(names)
    assert not any(n in (".env", "client_secret.json", "youtube_token.json") or n.endswith((".mp4", ".mp3")) for n in names)


def test_memes_are_not_repeated_from_video_to_video():
    lib = {"a": [Path(f"a{i}") for i in range(6)], "b": [Path(f"b{i}") for i in range(4)]}
    left = director.fresh(lib, ["a0", "b1"])
    assert not {Path("a0"), Path("b1")} & {f for fs in left.values() for f in fs}
    left = director.fresh(lib, ["a0", "a1", "a2", "b1"])  # too few would remain: only the latest stay out
    assert Path("a0") in left["a"] and Path("b1") not in left["b"] and sum(map(len, left.values())) == 8
    assert director.fresh({}, ["x"]) == {}


def test_the_camera_keeps_up_with_a_quick_step():
    xs = [0.51, 0.43, 0.36, 0.28, 0.27, 0.3, 0.34, 0.36]
    path = edit.follow([(i / 4, (x, 0.3, 0.2)) for i, x in enumerate(xs)], 0.0, 2.0)[0][2]
    assert abs(path[2][0][1] - xs[0]) < 0.12


def test_daily_uploads_only_what_passed_and_never_stops(api, tmp_path, monkeypatch, capsys):
    from shorts import schedule
    token = tmp_path / "token.json"
    token.write_text("{}")
    monkeypatch.setattr(schedule, "TOKEN", token)
    good = api.video
    bad = tmp_path / "bad.mp4"
    schedule.daily([(good, {}, []), (bad, {}, ["too long"])], public=False)
    log = capsys.readouterr().out
    assert "1 of 2" in log and "too long" in log and json.loads(api.record.read_text())["posted"] == {"youtube": "VID"}
    schedule.daily([(good, {}, [])], public=False)  # already up: logged, no exception
    assert "already on youtube" in capsys.readouterr().out
    token.unlink()
    schedule.daily([(good, {}, [])], public=False)
    assert "nothing uploaded" in capsys.readouterr().out


def test_nobody_is_asked_to_sign_in_at_night(tmp_path, monkeypatch):
    monkeypatch.setattr(post, "TOKEN", tmp_path / "none.json")
    with pytest.raises(RuntimeError, match="sign in again"):
        post.google_token(interactive=False)


def talk(*lens):
    """Sentences of these lengths, 0.6s apart."""
    out, t = [], 0.0
    for d in lens:
        out.append({"start": t, "end": t + d, "text": "x"})
        t += d + 0.6
    return out


def test_long_story_is_split_between_sentences(monkeypatch):
    t = talk(*[7.0] * 12)  # 90.6s
    end = t[-1]["end"]
    ends = [s["end"] for s in t]
    for answer in (None, {"breaks": ["?"]}, {"breaks": [1, 2, 3]}):  # no or useless answer: equal halves
        monkeypatch.setattr(sources, "llm", lambda *a, **k: answer)
        (a, m), (m2, b) = sources.split_parts(t, 0.0, end)
        assert (a, b) == (0.0, end) and m == m2 and abs(m - 45) < 4
        assert min(m - e for e in ends if e < m) <= 0.5  # after a sentence and its laugh
    monkeypatch.setattr(sources, "llm", lambda *a, **k: {"breaks": [49.1]})
    parts = sources.split_parts(t, 0.0, end)
    assert abs(parts[0][1] - 49.1) < 8 and all(30 <= b - a <= 50 for a, b in parts)
    monkeypatch.setattr(sources, "llm", lambda *a, **k: None)
    assert len(sources.split_parts(talk(*[6.0] * 20), 0.0, 131.4)) == 3
    assert sources.split_parts(talk(55.0), 0.0, 55.0) == [] and sources.split_parts(talk(80.0, 4.0), 0.0, 84.6) == []


def test_pick_story_may_be_long_but_never_in_between(monkeypatch):
    t = talk(*[7.0] * 30)
    for start, stop, lo, hi in ((0, 38, 30, 50), (0, 53, 30, 50), (0, 58, 62, 150), (0, 91, 62, 150), (0, 220, 30, 50)):
        monkeypatch.setattr(sources, "llm", lambda *a, **k: {"start": start, "end": stop, "score": 8})
        st = sources.pick_story(t, "", 0)
        assert lo <= st["end"] - st["start"] <= hi, (stop, st)


def test_parts_are_named_and_posted_only_together(api, tmp_path, monkeypatch):
    seen = []

    def make(name, src, d, title):
        seen.append((name, src["part"], title))
        return name, {"name": "T"}, []
    monkeypatch.setattr(sources, "clips", lambda d, s, span: {"credits": [s[0]]})
    sources.series("x", tmp_path, "v.mp4", [(0, 40), (40, 80)], make)
    assert seen == [("x_part1", (1, 2), ""), ("x_part2", (2, 2), "T")]
    one = api.video.with_name("x_part1.mp4")
    one.write_bytes(api.data)
    one.with_suffix(".json").write_text(json.dumps({**json.loads(api.record.read_text()), "parts": 2}))
    with pytest.raises(RuntimeError, match="x_part2"):
        post.post(one, "youtube")
    assert not api.calls
    two = api.video.with_name("x_part2.mp4")
    two.write_bytes(api.data)
    two.with_suffix(".json").write_text(one.with_suffix(".json").read_text())
    post.post(one, "youtube")


def test_outro_is_written_at_the_end(tmp_path):
    captions.write_ass(tmp_path / "a.ass", [[0, 1, "hi", False]], "T", (), (30.0, 34.0, r"FOLLOW\NFOR PART 2"))
    row = [r for r in (tmp_path / "a.ass").read_text(encoding="utf-8").splitlines() if "PART 2" in r]
    assert len(row) == 1 and "0:00:30.00,0:00:34.00" in row[0] and qa.ass_problems(tmp_path / "a.ass") == []


def test_no_word_is_lost_between_segments_and_other_hearings_reach_the_speller(tmp_path, monkeypatch):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    j = {"text": "one two three", "segments": [{"start": 0.2, "end": 1, "text": "one"}, {"start": 2, "end": 3, "text": "three"}],
         "words": [{"word": w, "start": t, "end": t + 0.3} for w, t in (("वन", 0.0), ("टू", 1.5), ("थ्री", 2.1))]}
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setattr(ai, "_groq_whisper", lambda *a: j)
    monkeypatch.setattr(ai, "probe", lambda p: (3.0, True, 1, 1))
    assert [w[2] for s in ai.transcribe(f) for w in s["words"]] == ["वन", "टू", "थ्री"]
    asked = []
    monkeypatch.setattr(ai, "llm", lambda p, *a: asked.append(p) or {"said": "van gogh", "words": {"0": "van"}})
    assert ai.hinglish(["वन"], ai.heard(f)) == ["van"] and "one two three" in asked[0] and "van gogh" in asked[1]


def test_a_corrected_word_must_sound_like_what_was_heard():
    for heard, said in (("किट्स", "kids"), ("हैफ", "have"), ("प्रिपेड", "prepared"), ("अंटीन", "canteen"), ("दरनन", "The Nun"),
                        ("वाच्चमान", "watchman"), ("है", "hai"), ("प्लेज", "pledge"), ("स्पूल", "school")):
        assert ai.sounds_like(heard, said), (heard, said)
    for heard, said in (("मुवी", "school"), ("मार", "convent"), ("सोचो", "baar")):
        assert not ai.sounds_like(heard, said), (heard, said)


def test_consensus_prefers_the_hearing_the_others_confirm():
    real, same, off = "मेरे स्कूल में इतनी लड़कियां थी".split(), "mere school mein itni ladkiyan thi".split(), "ऋख नहीं कोई भाग मिला".split()
    k, clear = ai.consensus([off, real, same])
    assert k in (1, 2)
    assert ai.consensus([real, same, real])[1] > ai.MIN_CLEAR > ai.consensus([off, real, "कहाँ चलो बस दो टिकट".split()])[1]


def test_low_subtitles_are_cropped_and_others_refused(tmp_path, monkeypatch):
    v = tmp_path / "a.mp4"
    monkeypatch.setattr(sources, "probe", lambda p: (40.0, True, 1920, 1080))
    monkeypatch.setattr(eyes, "grab_frame", lambda *a: 1)
    monkeypatch.setattr(rights, "has_music", lambda p: False)
    for top, at, why in ((0.91, 0.88, ""), (0.6, 0.57, "subtitles"), (None, 1.0, "")):
        v.write_bytes(b"x")
        (tmp_path / "a.mp4.floor.json").unlink(missing_ok=True)
        monkeypatch.setattr(eyes, "texts", lambda f: [("ki apka nam Ravi Gupta hai", (0.3, top, 0.7, top + 0.04))] if top else [])
        assert abs(sources.floor(v) - at) < 1e-6 and (why in sources.refuse(v)) and bool(why) == bool(sources.refuse(v))


def test_a_logo_is_not_a_subtitle(monkeypatch):
    monkeypatch.setattr(eyes, "texts", lambda f: [("URING OUT", (0.43, 0.93, 0.47, 1.0))])
    assert eyes.facts(None) == []


def test_a_wrong_time_never_swaps_words():
    w = ai.in_order([[4.7, 4.9, "ko"], [4.98, 5.2, "lagta"], [5.22, 5.4, "tha"], [4.9, 5.0, "main"], [5.46, 5.7, "tel"]])
    assert [x[2] for x in w] == ["ko", "lagta", "tha", "main", "tel"]
    assert all(a[0] <= b[0] for a, b in zip(w, w[1:]))
    assert ai.in_order([]) == []


def test_an_unseen_video_is_held_back_until_the_last_look(tmp_path, monkeypatch):
    monkeypatch.setattr(qa, "OUT", tmp_path / "out")
    monkeypatch.setattr(qa, "media_problems", lambda *a, **k: [])
    monkeypatch.setattr(qa, "has_music", lambda v: False)
    def limit(*a):
        raise ff.Stop("limit")
    monkeypatch.setattr(qa, "eye_problems", limit)
    (tmp_path / "out").mkdir()
    v = tmp_path / "out" / "x.mp4"
    v.write_bytes(b"video")
    with pytest.raises(ff.Stop):
        qa.deliver(v, {"title": "t"}, "clip")
    held = tmp_path / "out" / "rejected" / "x.mp4"
    assert held.exists() and not v.exists()
    monkeypatch.setattr(qa, "eye_problems", lambda *a: [])
    assert qa.finish_check(held) == [] and v.exists() and qa.certified(v)["title"] == "t"
    assert qa.finish_check(v) is None
