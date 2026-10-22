#!/bin/bash
# Poll the v2 kernel until it leaves RUNNING/QUEUED, then download its output.
# Separate from watch_kaggle.sh (v1) so both runs' logs stay distinguishable.
ID=sadiasadozai/slm-robustness-attack-v2
K=./venv/Scripts/kaggle.exe
OUT=kaggle_output5
mkdir -p "$OUT"
for i in $(seq 1 200); do
  out=$($K kernels status "$ID" 2>&1)
  echo "$(date +%H:%M:%S) $out"
  case "$out" in
    *COMPLETE*) echo "DONE_COMPLETE"; break;;
    *ERROR*|*CANCEL*) echo "DONE_FAILED"; break;;
  esac
  sleep 60
done
# the results dir is large (the raw corpus comes back too); data/ is not needed
rm -rf "$OUT/data"
$K kernels output "$ID" -p "$OUT" 2>&1 | tail -5
echo "DOWNLOAD_EXIT=$?"
rm -rf "$OUT/data"
ls -la "$OUT" 2>&1 | head -20
