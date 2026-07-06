#!/usr/bin/env bash
# Sequential full runs (one model in memory at a time). Logs to results/run_all.log.
set -uo pipefail
cd "$(dirname "$0")"
source /Users/alexroman/research/scinet_trackc/work/mcq-env/.venv/bin/activate
export HF_HOME=/Users/alexroman/research/scinet_trackc/cache/hf
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

run() {
  local model=$1 rev=$2 dtype=$3 bs=$4 out=$5
  echo "===== $(date +%H:%M:%S) START $model ($dtype bs=$bs) ====="
  python eval.py --model "$model" --revision "$rev" --data data/mmlu_subset.jsonl \
    --out "$out" --dtype "$dtype" --batch-size "$bs"
  echo "===== $(date +%H:%M:%S) END $model rc=$? ====="
}

run EleutherAI/pythia-160m-deduped  step143000 float32 32 results/scores_pythia-160m.csv
run EleutherAI/pythia-410m-deduped  step143000 float32 32 results/scores_pythia-410m.csv
run EleutherAI/pythia-1.4b-deduped  step143000 float16 16 results/scores_pythia-1.4b.csv
run EleutherAI/pythia-2.8b-deduped  step143000 float16 8  results/scores_pythia-2.8b.csv
echo "===== $(date +%H:%M:%S) ALL DONE ====="
