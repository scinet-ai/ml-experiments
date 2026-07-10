#!/bin/bash
# Stability sweep driver (resume-safe). 1.5B first (banks a full scale early), then 7B.
set -e
cd "$(dirname "$0")"
export HF_HOME=/Users/alexroman/research/scinet_seeding/track-c/cache/hf
export TOKENIZERS_PARALLELISM=false
PY=./.venv/bin/python
VARIANTS="para1 para2 para3 fmt1 fmt2 fmt3"

echo "===== [$(date +%H:%M:%S)] selftest ====="
$PY stability_eval.py selftest

for M in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct; do
  echo "===== [$(date +%H:%M:%S)] MODEL $M ====="
  for V in $VARIANTS; do
    echo "----- [$(date +%H:%M:%S)] $M variant $V -----"
    $PY stability_eval.py judge --model "$M" --variant "$V"
  done
  echo "----- [$(date +%H:%M:%S)] $M SAMPLING (T=0.7 n=5) -----"
  $PY stability_eval.py sample --model "$M"
  echo "===== [$(date +%H:%M:%S)] DONE $M ====="
done
echo "===== [$(date +%H:%M:%S)] ALL SWEEPS DONE ====="
