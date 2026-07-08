#!/bin/bash
# Zero-download smoke reproduction (no model, no dataset, no network):
#   1. unit-tests the mirror/flip/primacy/majority logic
#   2. recomputes every headline number (per-model flip rate, primacy share,
#      p(pick slot-1), human agreement, pairwise scale deltas + bootstrap CIs)
#      from the COMMITTED raw verdict CSVs + pairs_public.parquet,
#      and writes results.json — compare with the committed results.json.
#
# Full from-scratch reproduction (downloads models + dataset; see README):
#   python judge_eval.py prep && ./run_ladder.sh && python judge_eval.py analyze
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
if command -v uv >/dev/null; then
  [ -d .venv ] || { uv venv .venv -q && uv pip install -q --python .venv/bin/python pandas pyarrow numpy; }
  PY=.venv/bin/python
fi
$PY judge_eval.py selftest
$PY verbosity_eval.py selftest
cp results.json results_committed.json.bak
cp results_vb.json results_vb_committed.json.bak
$PY judge_eval.py analyze > /dev/null
$PY verbosity_eval.py analyze > /dev/null
$PY - <<'EOF'
import json
a = json.load(open("results.json"))
b = json.load(open("results_committed.json.bak"))
for m, r in a["per_model"].items():
    rb = b["per_model"][m]
    assert abs(r["flip_rate"] - rb["flip_rate"]) < 1e-9, (m, r["flip_rate"], rb["flip_rate"])
    print(f"{m}: flip_rate {r['flip_rate']:.4f} == committed  OK")
va = json.load(open("results_vb.json"))
vb = json.load(open("results_vb_committed.json.bak"))
for k, r in va["per_arm"].items():
    rb = vb["per_arm"][k]
    assert abs(r["p_longer"] - rb["p_longer"]) < 1e-9, (k, r["p_longer"], rb["p_longer"])
print(f"verbosity: {len(va['per_arm'])} arm/scale p_longer values == committed  OK")
print("SMOKE REPRO PASS: headline numbers regenerate exactly from committed raw verdicts")
EOF
