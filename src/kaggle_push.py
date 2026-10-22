"""
Push the generated kernel to Kaggle, wait for it, and download the results.

This is the "no notebook, no cell pasting" path: the kernel runs unattended on a
free Kaggle GPU and the outputs come back to ./kaggle_output/.

Authentication (one time, done by the account owner):

    ./venv/Scripts/kaggle.exe auth login          # opens a browser to approve

or drop an API token into ~/.kaggle/access_token, or export KAGGLE_API_TOKEN.

    python src/kaggle_push.py --wait
    python src/kaggle_push.py --wait --timeout-min 90
    python src/kaggle_push.py --status-only
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KAGGLE_DIR = os.path.join(ROOT, "kaggle")
META = os.path.join(KAGGLE_DIR, "kernel-metadata.json")
OUT_DIR = os.path.join(ROOT, "kaggle_output")
KAGGLE = os.path.join(ROOT, "venv", "Scripts", "kaggle.exe")
if not os.path.exists(KAGGLE):                       # linux/macOS checkout
    KAGGLE = shutil.which("kaggle") or "kaggle"


def cli(*args, check=False):
    proc = subprocess.run([KAGGLE, *args], cwd=ROOT, text=True,
                          capture_output=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{proc.stdout}\n{proc.stderr}")
    return proc


def detect_username():
    """Kaggle has no 'whoami'; the username appears in config or the token file."""
    for args in (("config", "view"),):
        proc = cli(*args)
        m = re.search(r"username[:\s\"']+([A-Za-z0-9_.-]+)", proc.stdout + proc.stderr, re.I)
        if m:
            return m.group(1)
    token = os.path.expanduser("~/.kaggle/access_token")
    if os.path.exists(token):
        return None                                  # token has no username in it
    return None


def prepare(meta, username, slug):
    """Copy kernel.py + metadata into a temp dir with the id resolved."""
    meta = dict(meta)
    meta["id"] = f"{username}/{slug}"
    tmp = tempfile.mkdtemp(prefix="kaggle_push_")
    shutil.copy(os.path.join(KAGGLE_DIR, "kernel.py"), os.path.join(tmp, "kernel.py"))
    with open(os.path.join(tmp, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return tmp, meta["id"]


def status(kernel_id):
    proc = cli("kernels", "status", kernel_id)
    text = (proc.stdout or "") + (proc.stderr or "")
    for state in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if state in text:
            return state
    if "404" in text or "not found" in text.lower():
        return "NOT_FOUND"
    return "UNKNOWN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=None,
                    help="kernel slug (default: the slug in kernel-metadata.json)")
    ap.add_argument("--username", default=None,
                    help="Kaggle username (default: read from kaggle config)")
    ap.add_argument("--wait", action="store_true", help="poll until the kernel finishes")
    ap.add_argument("--timeout-min", type=int, default=120)
    ap.add_argument("--poll-seconds", type=int, default=20)
    ap.add_argument("--status-only", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--out", default=None,
                    help="download directory (default: kaggle_output). Use a "
                         "fresh one per run so an earlier run's results are "
                         "not overwritten")
    args = ap.parse_args()

    if not os.path.exists(META):
        raise SystemExit("missing kaggle/kernel-metadata.json -> run: "
                         "python src/make_kaggle.py")

    probe = cli("kernels", "list", "--mine", "-p", "1")
    if probe.returncode != 0:
        print("Kaggle authentication is not set up.\n")
        print("One-time step (account owner):")
        print("  ./venv/Scripts/kaggle.exe auth login        # opens browser to approve")
        print("or: export KAGGLE_API_TOKEN=<token from kaggle.com/settings/api>")
        print(f"\nCLI said:\n{(probe.stderr or probe.stdout).strip()[:400]}")
        return 2

    with open(META, encoding="utf-8") as f:
        meta = json.load(f)
    username = args.username or detect_username()
    if not username:
        raise SystemExit("could not read the Kaggle username; pass --username <name>")

    if args.status_only:
        print(status(f"{username}/{args.slug}"))
        return 0

    slug = args.slug or str(meta.get("id", "")).split("/")[-1]
    if not slug:
        raise SystemExit("kernel-metadata.json has no id; run src/make_kaggle.py")
    tmp, kernel_id = prepare(meta, username, slug)
    try:
        print(f"pushing {kernel_id} (gpu, internet, {meta['title']})")
        push = cli("kernels", "push", "-p", tmp)
        output = (push.stdout or "") + (push.stderr or "")
        print(output.strip())
        if push.returncode != 0:
            return 1
        # Kaggle derives the real slug from the TITLE, not from metadata.id, so
        # trust the URL it prints or every later status poll would 404.
        m = re.search(r"kaggle\.com/code/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", output)
        if m:
            kernel_id = m.group(1).rstrip(".")
            print(f"kernel id: {kernel_id}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if not args.wait:
        print(f"\nqueued. Check with: python src/kaggle_push.py --status-only")
        return 0

    deadline = time.time() + args.timeout_min * 60
    last = None
    while time.time() < deadline:
        state = status(kernel_id)
        if state != last:
            print(f"[{time.strftime('%H:%M:%S')}] {state}")
            last = state
        if state in ("COMPLETE", "ERROR", "CANCEL", "NOT_FOUND"):
            break
        time.sleep(args.poll_seconds)
    else:
        raise SystemExit(f"timed out after {args.timeout_min} min; kernel may still run")

    if state != "COMPLETE":
        print("\nkernel did not complete. Logs: kaggle kernels logs " + kernel_id)
        return 1

    if args.no_download:
        return 0
    out_dir = os.path.abspath(args.out) if args.out else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    dl = cli("kernels", "output", kernel_id, "-p", out_dir)
    print((dl.stdout or dl.stderr).strip())
    report = os.path.join(out_dir, "REPORT.md")
    if os.path.exists(report):
        print("\n" + "=" * 60)
        print(open(report, encoding="utf-8").read())
    return 0 if dl.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
