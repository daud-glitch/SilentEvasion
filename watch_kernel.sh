#!/usr/bin/env bash
# Poll one Kaggle kernel until it finishes, then download its output.
#
#   ./watch_kernel.sh <owner/slug> <output-dir>
#
# Progress goes to <output-dir>.watch.log so it can be tailed live. Every
# kaggle call is wrapped in `timeout`, because this CLI occasionally opens a
# connection that never answers: without the timeout the watcher blocks forever
# and a finished run looks like a run that is still going.
set -u
REF=${1:?usage: watch_kernel.sh <owner/slug> <output-dir>}
OUT=${2:?usage: watch_kernel.sh <owner/slug> <output-dir>}
LOG="$OUT.watch.log"
PYTHONIOENCODING=utf-8

stamp() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
kaggle_call() { PYTHONIOENCODING=utf-8 timeout 120 ./venv/Scripts/kaggle.exe "$@" 2>&1; }
# The download is the one call that legitimately takes many minutes: a
# three-model run is several hundred MB. Timing it out at the poll timeout
# truncates the output silently (rc=124) and leaves a half-copied results/.
kaggle_download() { PYTHONIOENCODING=utf-8 timeout 3600 ./venv/Scripts/kaggle.exe "$@" 2>&1; }

for _ in $(seq 1 900); do
  ST=$(kaggle_call kernels status "$REF" | tail -1)
  case "$ST" in
    *COMPLETE*)
      stamp "COMPLETE - downloading -> $OUT"
      kaggle_download kernels output "$REF" -p "$OUT" >> "$LOG" 2>&1
      stamp "download rc=$? done"
      exit 0;;
    *ERROR*|*CANCEL*)
      stamp "run ended: $ST"
      kaggle_call kernels logs -f "$REF" > "${OUT}.failure.log" 2>&1 &
      exit 1;;
    *)
      stamp "$ST";;
  esac
  sleep 60
done
stamp "watcher timed out"
