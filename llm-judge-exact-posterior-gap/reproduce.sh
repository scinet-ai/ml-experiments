#!/usr/bin/env bash
# Reproduce INV-E1 results WITHOUT any API access.
#
#   ./reproduce.sh analysis   # (default) regenerate all metrics + plots from
#                             #   results/calls.jsonl in seconds. No API keys.
#   ./reproduce.sh smoke      # end-to-end pipeline validation with a MockJudge
#                             #   (exact posterior + noise): builds 1 graph, runs
#                             #   the runner offline, then analyzes it. No API keys.
#   ./reproduce.sh tests      # substrate unit checks (enumeration/legality/parser).
#
# Third parties can thus validate the whole pipeline (graph gen -> exact inference
# -> reveal sets -> prompt build -> run loop -> analysis) with no model calls.
set -euo pipefail
cd "$(dirname "$0")"

# --- python env: prefer an existing .venv, else uv, else stdlib venv ----------
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv >/dev/null 2>&1 || uv venv .venv
  uv pip install --python .venv/bin/python -r requirements.txt >/dev/null
  PY=".venv/bin/python"
else
  python3 -m venv .venv
  ./.venv/bin/python -m pip install --quiet --upgrade pip
  ./.venv/bin/python -m pip install --quiet -r requirements.txt
  PY=".venv/bin/python"
fi

MODE="${1:-analysis}"
case "$MODE" in
  tests)
    "$PY" tests.py
    ;;
  smoke)
    echo ">> offline smoke: MockJudge, 1 graph, no API"
    "$PY" run_experiment.py --mode smoke
    "$PY" analysis.py --calls results/smoke_calls.jsonl
    echo ">> smoke OK: pipeline runs end-to-end with no API"
    ;;
  analysis)
    if [ ! -s results/calls.jsonl ]; then
      echo "results/calls.jsonl not found — run './reproduce.sh smoke' for an offline demo," >&2
      echo "or 'python run_experiment.py --mode full' (needs API) to regenerate calls." >&2
      exit 1
    fi
    "$PY" analysis.py
    ;;
  *)
    echo "usage: $0 [analysis|smoke|tests]" >&2; exit 2 ;;
esac
