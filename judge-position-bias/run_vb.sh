#!/bin/bash
cd "$(dirname "$0")"
export HF_HOME=/Users/alexroman/research/scinet_seeding/track-c/cache/hf
PY=.venv/bin/python
for M in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct; do
  for ARM in noinstr antiverb; do
    echo "===== $ARM $M  $(date) ====="
    $PY verbosity_eval.py judge --model "$M" --arm "$ARM" || echo "FAILED $ARM $M"
  done
done
echo "===== VB LADDER COMPLETE $(date) ====="
