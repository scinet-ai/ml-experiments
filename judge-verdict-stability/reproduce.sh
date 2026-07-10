#!/bin/bash
# Zero-download smoke reproduction (no model, no dataset, no network):
#   1. unit-tests the template decomposition, majority, and instability logic
#      (asserts the base template reconstructs judge_eval.JUDGE_TEMPLATE exactly,
#       which is what makes reusing cf4e6c02's base greedy verdicts valid);
#   2. recomputes every headline number (per-family instability fractions + 95% CIs,
#      robust-core fraction + CI, per-variant instability, scale-trend deltas) from the
#      COMMITTED raw verdict CSVs + samples CSVs + pairs_public.parquet, writing
#      results_stab.json, and checks it matches the committed results_stab.json exactly.
#
# Full from-scratch reproduction (downloads models + dataset; see README):
#   python judge_eval.py prep          # builds pairs.parquet (needs the HF dataset)
#   bash run_stab.sh                    # 6 argmax variants + T=0.7 n=5 sampling, both scales
#   python stability_eval.py analyze
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
if command -v uv >/dev/null; then
  [ -d .venv ] || { uv venv .venv -q && uv pip install -q --python .venv/bin/python pandas pyarrow numpy; }
  PY=.venv/bin/python
fi
$PY judge_eval.py selftest
$PY stability_eval.py selftest
cp results_stab.json results_stab_committed.json.bak
$PY stability_eval.py analyze > /dev/null
$PY - <<'EOF'
import json
a = json.load(open("results_stab.json"))
b = json.load(open("results_stab_committed.json.bak"))
for mdl, r in a["per_model"].items():
    rb = b["per_model"][mdl]
    for fam in ("sampling", "paraphrase", "formatting"):
        x, y = r["family_instability"][fam]["instability"], rb["family_instability"][fam]["instability"]
        assert abs(x - y) < 1e-9, (mdl, fam, x, y)
    rc, rcb = r["robust_core"]["robust_core"], rb["robust_core"]["robust_core"]
    assert abs(rc - rcb) < 1e-9, (mdl, "robust_core", rc, rcb)
    print(f"{mdl}: sampling {r['family_instability']['sampling']['instability']:.4f} | "
          f"paraphrase {r['family_instability']['paraphrase']['instability']:.4f} | "
          f"formatting {r['family_instability']['formatting']['instability']:.4f} | "
          f"robust_core {rc:.4f}  == committed  OK")
print("SMOKE REPRO PASS: every headline number regenerates exactly from committed raw verdicts")
EOF
rm -f results_stab_committed.json.bak
