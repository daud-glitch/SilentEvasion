"""Poll the Kaggle kernels and download whichever output is ready.

    python src/watch_kernels.py            # poll until both finish
    python src/watch_kernels.py --once

Why a Python watcher instead of a shell loop: `kaggle kernels status` returns a
quoted enum string, and a bash `case` on it plus `sleep` in a background job is
the kind of thing that silently dies with the parent shell. This keeps the
mapping (kernel -> output directory) in one place and writes a log line per
poll, so a run that sits QUEUED for an hour is visible rather than mysterious.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE = os.path.join(ROOT, "venv", "Scripts", "kaggle.exe")
if not os.path.exists(KAGGLE):
    KAGGLE = "kaggle"

USER = "sadiasadozai"
# One entry per kernel still worth waiting for. Finished runs are removed rather
# than left to be polled (and re-downloaded) every five minutes.
KERNELS = {
    "slm-robustness-selftransfer-v6": "kaggle_output9",
}
DONE = ("COMPLETE", "ERROR", "CANCEL")
LOG = os.path.join(ROOT, "kaggle_watch.log")


def log(msg):
    line = "%s %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(slug):
    try:
        out = subprocess.run([KAGGLE, "kernels", "status", "%s/%s" % (USER, slug)],
                             capture_output=True, text=True, timeout=90)
    except Exception as exc:                      # noqa: BLE001
        return "UNKNOWN(%s)" % type(exc).__name__
    text = (out.stdout + out.stderr).strip()
    for word in DONE + ("RUNNING", "QUEUED"):
        if word in text:
            return word
    return "UNKNOWN"


def download(slug):
    dest = os.path.join(ROOT, KERNELS[slug])
    os.makedirs(dest, exist_ok=True)
    log("downloading %s -> %s" % (slug, KERNELS[slug]))
    r = subprocess.run([KAGGLE, "kernels", "output", "%s/%s" % (USER, slug),
                        "-p", dest], capture_output=True, text=True, timeout=900)
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    for ln in tail:
        log("  " + ln)
    log("download rc=%s" % r.returncode)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll-seconds", type=int, default=120)
    ap.add_argument("--max-polls", type=int, default=400)
    args = ap.parse_args()

    downloaded = set()
    for poll in range(args.max_polls):
        states = {}
        for slug in KERNELS:
            states[slug] = status(slug)
        log("poll %d: %s" % (poll + 1, ", ".join(
            "%s=%s" % (s.replace("slm-robustness-", ""), st)
            for s, st in states.items())))
        for slug, st in states.items():
            if st in DONE and slug not in downloaded:
                download(slug)
                downloaded.add(slug)
        if len(downloaded) == len(KERNELS) or args.once:
            break
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
