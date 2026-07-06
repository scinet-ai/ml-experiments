#!/usr/bin/env bash
# ZERO-DOWNLOAD smoke reproduction of the headline table from committed score CSVs.
# Uses only numpy/pandas (no model download, no GPU). Runs in seconds.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

echo "== verify.py (assert headline: raw bias + PriDe reduction) =="
$PY verify.py

echo
echo "== analyze.py (regenerate full summary table across scales) =="
$PY analyze.py --scores 'results/scores_pythia-*.csv' --out results/summary.csv --est-frac 0.25 --seed 0
