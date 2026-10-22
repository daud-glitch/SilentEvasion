#!/bin/bash
# Poll the v3 (transfer) kernel until it finishes, then download its output.
ID=sadiasadozai/slm-robustness-transfer-v3
K=./venv/Scripts/kaggle.exe
OUT=kaggle_output6
mkdir -p "$OUT"
for i in $(seq 1 220); do
  out=$($K kernels status "$ID" 2>&1)
  echo "$(date +%H:%M:%S) $out"
  case "$out" in
    *COMPLETE*) echo "DONE_COMPLETE"; break;;
    *ERROR*|*CANCEL*) echo "DONE_FAILED"; break;;
  esac
  sleep 60
done
rm -rf "$OUT/data" 2>/dev/null
$K kernels output "$ID" -p "$OUT" 2>&1 | tail -5
echo "DOWNLOAD_EXIT=$?"
rm -rf "$OUT/data" 2>/dev/null
ls "$OUT" 2>&1 | head -20
