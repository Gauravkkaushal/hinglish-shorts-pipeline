# Hinglish Shorts Pipeline

Give it a creator. It finds the best 30-50 second story in their videos, cuts it vertically around the speaker, adds
Hinglish captions and reaction memes, checks the result, and can upload it to YouTube and Instagram - every day by itself.

- [1. Install](#1-install)
- [2. First video](#2-first-video)
- [3. Choose the creators](#3-choose-the-creators)
- [4. All commands](#4-all-commands)
- [5. How the edit is made](#5-how-the-edit-is-made)
- [6. Quality check](#6-quality-check)
- [7. Posting](#7-posting)
- [7a. Every day by itself](#7a-every-day-by-itself)
- [8. Rights](#8-rights)
- [9. Problems and fixes](#9-problems-and-fixes)
- [10. For developers](#10-for-developers)

---

## 1. Install

You need **Windows 10 or 11**, about 4 GB of free disk space, 8 GB of memory and an internet connection. An NVIDIA card makes
rendering faster but is not required. Everything used is free.

**Step 1 - programs.** Open PowerShell and run these four lines one by one:

```powershell
winget install Python.Python.3.13
winget install Gyan.FFmpeg
winget install DenoLand.Deno
winget install Git.Git
```

(Deno is what yt-dlp uses to read YouTube.) **Close PowerShell and open it again**, or the new programs are not found.

**Step 2 - the project.** Download it into `C:\shorts-pipeline` and go there:

```powershell
git clone https://github.com/Gauravkkaushal/hinglish-shorts-pipeline.git C:\shorts-pipeline
cd C:\shorts-pipeline
```

(Without Git: on the GitHub page press *Code* > *Download ZIP* and unzip it to `C:\shorts-pipeline`.) Later, `git pull` in
that folder brings updates; your `.env`, videos and memes are not touched by it.

**Step 3 - setup.** This makes a private Python environment in `.venv` and installs everything into it (5-10 minutes):

```powershell
.\setup.bat
```

**Step 4 - the free AI key.** Make an account at [console.groq.com/keys](https://console.groq.com/keys), create a key, open the
file `.env` in Notepad and put it behind `GROQ_API_KEY=`. Never share this file.

```powershell
notepad .env
```

**Step 5 - reaction memes** (once, 10-20 minutes). Clips with music, without a face or without sound are left out by
themselves. You can also drop your own `.mp4` files into `library\laugh`, `library\shock` and so on.

```powershell
.\run fetch-memes
```

**Step 6 - check.** Every line should say `OK` (the `optional` ones may wait):

```powershell
.\run doctor
```

## 2. First video

```powershell
.\run auto --creator @ZakirKhan
```

Takes 5-15 minutes. The result is in `out\`: a video and a `.json` next to it with the title, caption and hashtags. A video
that fails the quality check goes to `out\rejected\` with the reasons in its `.json`.

Or use your own file:

```powershell
.\run clip --src "D:\videos\podcast.mp4" --name mypodcast
```

## 3. Choose the creators

**On the command line:** `--creator` takes an @handle, a channel link, or a name, and can be repeated. Only these creators are
used then.

```powershell
.\run auto --creator @ZakirKhan --creator "Abhishek Upmanyu" --count 2
```

A handle is safest. A name means the channel of that name with the most subscribers; the one that was chosen is printed.

**Without `--creator`** the pipeline picks up to six random creators from **`creators.txt`**. Open it in Notepad and make it yours:

```
[talk: standup comedy]      <- used by: auto, and viral for talk domains
@ZakirKhan
@AnubhavSinghBassi

[funny: gta game gaming]    <- used by: comedy, when one of these words is in --topic
@TechnoGamerzOfficial

[footage: gta]              <- used by: montage, when one of these words is in --topic
@RockstarGames
```

One creator per line; text after `#` is ignored. If no `funny`/`footage` section fits the topic, YouTube is searched instead.
A video that was already used or judged is never used twice (`jobs\done.txt`; delete a line to allow that video again).

## 4. All commands

Put `.\run` in front of each.

| Command | What you get |
|---|---|
| `auto` | the best story of the creators' talks, stand-up or podcasts |
| `auto --creator @x --count 3` | three videos, only from this creator |
| `auto --query "hindi podcast funny"` | from a YouTube search instead of creators |
| `clip --src file.mp4` | one video from your own file or link (the best 30-50 s of it) |
| `clip --src long.mp4` | a file of about 60-150 s becomes **Part 1, Part 2...**, each ending with "FOLLOW & SUBSCRIBE FOR PART 2" |
| `clip --src long.mp4 --topic "exam"` | the best 30-50 s about that topic instead of parts |
| `comedy --topic "GTA 6" --reels 2` | the funniest 30-50 s of gaming creators |
| `montage --topic "GTA 6"` | a beat-synced edit on an original beat |
| `viral --domain "Hindi podcast"` | decides between talk and montage by itself |
| `story --topic "..."` | AI script and voice; needs your own footage in `library\broll` |
| `check out\x.mp4` | the quality check again |
| `post --to youtube out\x.mp4` | upload, see [Posting](#7-posting) |
| `doctor` | is everything installed? |

`--creator` works with `auto`, `comedy`, `montage` and `viral`. Story length is always 30-50 seconds.

## 5. How the edit is made

1. **Find.** Around the most replayed moment of a video an AI looks for a complete story with a payoff and scores it 1-10.
   Below `--min-score` (7) the video is skipped.
2. **Refuse.** Music behind the talk, speech that cannot be understood, or subtitles in the middle of the picture: skipped.
   Black bars and the creator's own subtitles at the bottom are cropped off.
3. **Cut by listening.** Silence goes. The audience's laugh after a line **stays** (about a second) and fades out. Words are
   never cut. The story starts and ends on a pause, not in the middle of a sentence.
4. **Follow the speaker.** The vertical frame pans with the face, goes closer when the speaker is small, and never carries a
   framing over a camera change.
5. **Hook.** One curious line from later in the talk is played first.
6. **Punch.** Zoom and a low hit on the lines where the audience laughs hardest; reaction memes with their own sound right
   after the best ones (score 8 or more, at most one every 6 seconds).
7. **Captions.** The speech is heard three ways, compared, and written in Hinglish.

No music is ever added to talk videos. `montage` uses a beat made by the program itself, or your own song from
`library\music\licensed\` if a file `<song>.license.json` with `source`, `license` and `url` lies next to it.

## 6. Quality check

Every video must pass: 1080x1920 at 30 fps, 30-50 s, loudness, no frozen or black picture, no unexpected music, captions that
are readable, not overlapping and not too wide, no facecam or watermark in montages, and a last look by a vision AI. The
certificate in the `.json` fits exactly that file: an edited or unchecked video cannot be posted.

It does **not** check whether the joke is good, or whether you may use the footage. Watch each video once before posting.

## 7. Posting

Through the official APIs. Each video goes out once per site, and no part of a series before all its parts have passed.

```powershell
.\run post --to youtube out\x.mp4
.\run post --to youtube --public out\x.mp4
.\run post --to instagram --public out\x.mp4
```

Without `--public`, YouTube gets a private upload to look at in YouTube Studio. Instagram's API has no private upload. Videos
are marked "not made for kids".

**YouTube, once:**
1. [console.cloud.google.com](https://console.cloud.google.com): new project, enable **YouTube Data API v3**.
2. Google Auth Platform: app name, your email, **External**. Under *Audience*, add your own Google account as a **test user**.
3. *Clients* > *Create client* > **Desktop app** > *Download JSON*. Save it here as `client_secret.json`.
4. The first `post` opens your browser: sign in, choose the channel, allow. To change the channel, delete `youtube_token.json`.
5. For uploads without you: in Google Auth Platform > *Audience* press **Publish app** (it stays unverified; that is fine for
   your own channel). While it is in "Testing", Google ends the sign-in every 7 days and the daily uploads stop.

**Instagram, once:**
1. The account must be Professional (Creator or Business) and linked to a Facebook Page.
2. [developers.facebook.com](https://developers.facebook.com): create an app with Facebook Login and the Instagram API. For your
   own account no App Review is needed.
3. Graph API Explorer: make a token with `instagram_basic`, `instagram_content_publish`, `pages_show_list`,
   `pages_read_engagement`, then *Extend* it in the Access Token Debugger (otherwise it ends after an hour).
4. With that token open `me/accounts?fields=name,access_token,instagram_business_account`. Put your Page's `access_token` into
   `.env` as `IG_TOKEN` and `instagram_business_account.id` as `IG_USER_ID`. (Instagram-Login tokens cannot upload files.)

`.env`, `client_secret.json` and `youtube_token.json` are secrets: whoever has them can act as you.

## 7a. Every day by itself

The computer must be on or asleep (not shut down) at that time; Windows wakes it, and if it was off the run starts as soon
as it is on again.

```powershell
.\run schedule --at 02:00 --count 2            # every night: 2 videos, private uploads to look at
.\run schedule --at 02:00 --count 2 --public   # every night: published at once (YouTube, and Instagram if set up)
.\run schedule --off                           # stop it
```

Each night it runs `daily`: `auto` with the creators of `creators.txt` (or `--creator ...`), then it uploads every video that
passed the check. What happened is in `out\daily.log`. Try it once by hand first: `.\run daily --count 1`.

`--public` posts without anybody looking at it. The check cannot tell whether the joke works or whether you may use the
footage, and three copyright strikes close a YouTube channel. Start without `--public` for a week.

## 8. Rights

The clips, memes and game footage belong to other people. A credit line is added to the caption, but **credit is not
permission**: creators can file claims or strikes, and platforms may refuse to pay for reused content. Ask the creators, use
official clipping programmes, or use your own recordings.

## 9. Problems and fixes

| Message | What to do |
|---|---|
| `MISSING ...` in doctor | do what stands behind the arrow, open a new terminal, run doctor again |
| `... is not recognized` | open a new PowerShell window; make sure you are in the project folder (`cd C:\shorts-pipeline`) |
| `can't open file 'pipeline.py'` | you are in the wrong folder: use `.\run ...` from the project folder |
| `STOPPED: the free daily Groq limit is used up` | wait the time it says and run the same command again; finished work is kept. About 5-10 videos a day are free |
| `STOPPED: YouTube is refusing automated downloads` | wait a few hours, or use `clip` with your own file. Do not hammer it |
| yt-dlp errors or very slow downloads | update it: `.venv\Scripts\python -m pip install -U "yt-dlp[default]"` |
| `speech was not understood clearly` | that recording is too noisy for good captions; try another video |
| `no unseen video of 8 minutes to 4 hours found` | this creator's recent videos are used up: choose another |
| `no channel is called ...` | give the @handle or the channel link |
| `NVENC unavailable, using CPU` | harmless, only slower |
| `QA REJECTED` | the reason is in the message and in `out\rejected\*.json` |
| Google: "app not verified" | it is your own app: *Continue* |
| Google: "access blocked" | add your account as a test user (Posting, step 2) |
| `Google needs you to sign in again` in daily.log | run `.\run post --to youtube <video>` once by hand; then *Publish app* (Posting, step 5) |
| the daily task did not run | the computer was shut down, or `.env`/sign-in is missing: look at `out\daily.log`, try `.\run daily --count 1` |

## 10. For developers

```powershell
.venv\Scripts\python -m pytest -q test_pipeline.py
```

| File | Job |
|---|---|
| `pipeline.py` | commands, and `make()`: source in, finished video out |
| `creators.txt`, `shorts/creators.py` | whose videos |
| `shorts/sources.py` | finding, downloading, picking the story, splitting into parts |
| `shorts/ai.py` | speech to text, Hinglish, language and vision models |
| `shorts/director.py` | memes, hook, punchlines, title |
| `shorts/edit.py` | cutting, camera, rendering |
| `shorts/captions.py` | captions |
| `shorts/montage.py`, `comedy.py`, `reel.py`, `cover.py` | the other modes |
| `shorts/eyes.py`, `rights.py` | faces and text on screen; music, speech and laughter |
| `shorts/qa.py`, `post.py`, `schedule.py`, `doctor.py` | check, upload, daily task, setup check |

Models: Groq `whisper-large-v3` (speech), `qwen/qwen3.8-27b` (captions, directing, vision), `openai/gpt-oss-120b` (finding stories;
`LLM_MODEL` in `.env` changes it). On the computer: YuNet (faces), RapidOCR (text), YAMNet (sounds).

Never commit `.env`, `client_secret.json` or `youtube_token.json`: `.gitignore` keeps them out, and `share.py` makes a zip that
leaves them out too. Licences of the bundled models and fonts: [assets/NOTICE.md](assets/NOTICE.md).

The code is under the [MIT licence](LICENSE). It covers this code only, not the videos it works with (see [Rights](#8-rights)).
