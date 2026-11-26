"""Runs by itself every day: `daily` makes videos and uploads the ones that passed; `schedule` puts it into Windows Task
Scheduler (wakes the computer, catches up after it was off). Everything it does is written to out/daily.log."""
import subprocess, time

from .ff import ENV_FILE, OUT, ROOT, env
from .post import TOKEN, post

TASK = "Shorts Pipeline"


def sites(public):
    """Where uploads can go without anybody at the keyboard."""
    out = ["youtube"] if TOKEN.exists() else []
    return out + (["instagram"] if public and env("IG_USER_ID") and env("IG_TOKEN") else [])


def daily(made, public):
    """made: what the run produced. Uploads every video that passed; a refusal or a broken upload is logged, never fatal."""
    print(time.strftime("=== %Y-%m-%d %H:%M"), f"{sum(not p for *_, p in made)} of {len(made)} video(s) passed")
    where = sites(public)
    if not where:
        print("  nothing uploaded: no site is set up (README, Posting)")
    for out, meta, problems in made:
        if problems:
            print(f"  not uploaded, failed the check: {out.name}: {'; '.join(problems)}")
            continue
        for site in where:
            try:
                post(out, site, public, interactive=False)
            except RuntimeError as e:
                print(f"  {site}: {out.name} NOT uploaded: {e}")


def schedule(at, args, off=False):
    """Creates (or replaces, or with off removes) the daily task of the current user."""
    if off:
        return subprocess.run(["powershell", "-NoProfile", "-Command", f"Unregister-ScheduledTask -TaskName '{TASK}' -Confirm:$false"]).returncode == 0
    run = ROOT / "run.bat"
    if not (ROOT / ".venv").exists() or not ENV_FILE.exists():
        raise RuntimeError("run setup.bat and put the Groq key into .env first")
    line = f'/c ""{run}" daily {" ".join(args)} >> "{OUT / "daily.log"}" 2>&1"'.replace("'", "''")
    ps = (f"$a = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument '{line}' -WorkingDirectory '{ROOT}';"
          f"$t = New-ScheduledTaskTrigger -Daily -At '{at}';"
          "$s = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
          "-ExecutionTimeLimit (New-TimeSpan -Hours 6);"
          f"Register-ScheduledTask -TaskName '{TASK}' -Action $a -Trigger $t -Settings $s -Force | Out-Null")
    OUT.mkdir(exist_ok=True)
    ok = subprocess.run(["powershell", "-NoProfile", "-Command", ps]).returncode == 0
    if ok:
        print(f"Every day at {at}: run daily {' '.join(args)}. Log: out\\daily.log. Stop it with: run schedule --off")
    return ok
