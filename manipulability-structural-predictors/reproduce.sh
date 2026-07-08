#!/usr/bin/env bash
# Public smoke reproduction for INV-E2.
# Regenerates a 200-graph mini-corpus + the regression on it, from a FRESH uv venv,
# in well under 60s. Installs probability-flow from PyPI (reproducibility depends on
# the public package, NOT any local checkout).
#
# Usage:  ./reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/4] creating fresh venv + installing pinned deps from PyPI ..."
uv venv .venv-repro >/dev/null 2>&1
uv pip install --python .venv-repro/bin/python -q -r requirements.txt
PY=.venv-repro/bin/python

echo "[2/4] smoke test (generate -> compile -> exact posterior_range -> is_exact) ..."
$PY scripts/smoke.py | tail -3

echo "[3/4] mini-corpus: 200 seeded graphs -> results/mini_corpus.csv ..."
time $PY scripts/sweep.py 200 results/mini_corpus.csv | tail -6

echo "[4/4] regression on the mini-corpus (quick) ..."
$PY scripts/analysis.py results/mini_corpus.csv --quick | \
    $PY -c "import sys,json; d=json.load(sys.stdin); \
print('rows=%d  OLS_R2=%.3f  GBM_R2=%.3f' % (d['n_rows'], d['OLS_full_R2_oos'], d['GBM_R2_oos'])); \
print('mass+asym OLS R2 =', d['hypothesis_nested_R2']['mass_plus_asym'][0])"

echo "DONE. Mini-corpus + regression reproduced."
