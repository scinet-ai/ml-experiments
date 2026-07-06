#!/bin/bash
# Full ladder run: judge all pairs x both orders per model, then free-gen robustness check.
cd "$(dirname "$0")"
export HF_HOME=/Users/alexroman/research/scinet_seeding/track-c/cache/hf
PY=.venv/bin/python
for M in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct; do
  echo "===== JUDGE $M  $(date) ====="
  $PY judge_eval.py judge --model "$M" || echo "FAILED judge $M"
  echo "===== GENCHK $M  $(date) ====="
  $PY judge_eval.py genchk --model "$M" || echo "FAILED genchk $M"
done
echo "===== LADDER COMPLETE $(date) ====="
