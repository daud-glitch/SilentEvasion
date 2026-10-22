#!/bin/bash
# poll until the kernel leaves RUNNING, then download its output into kaggle_output4
ID=sadiasadozai/slm-robustness-attack-v1
K=./venv/Scripts/kaggle.exe
for i in $(seq 1 240); do
  out=$($K kernels status "$ID" 2>&1)
  echo "$(date +%H:%M:%S) $out"
  case "$out" in
    *COMPLETE*) echo "DONE_COMPLETE"; break;;
    *ERROR*|*CANCEL*) echo "DONE_FAILED"; break;;
  esac
  sleep 60
done
$K kernels output "$ID" -p kaggle_output4 2>&1 | tail -3
echo "DOWNLOAD_EXIT=$?"
