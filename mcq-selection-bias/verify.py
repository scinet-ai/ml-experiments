"""Zero-download smoke verification of the headline result.

Reads a COMMITTED raw-scores CSV (results/scores_pythia-160m.csv) and, using only
numpy/pandas via analyze.py (no model, no GPU, no network), recomputes the
selection-bias metric and the PriDe residual, then asserts the headline claims:

  1. Substantial raw selection bias   (RStd well above the ~0 unbiased baseline)
  2. PriDe reduces the residual RStd   (post-debiasing RStd < raw RStd)

Exit code 0 (PASS) reproduces the headline from committed data alone.

Usage: python verify.py
"""
import sys, pathlib
import pandas as pd
from analyze import analyze_model

HERE = pathlib.Path(__file__).parent


def main():
    fp = HERE / "results" / "scores_pythia-160m.csv"
    if not fp.exists():
        print(f"[verify] FAIL: committed scores missing: {fp}")
        return 1
    df = pd.read_csv(fp)
    m = analyze_model(df, est_frac=0.25, seed=0)
    print("[verify] pythia-160m-deduped:")
    for k in ["n_questions", "acc", "recall_A", "recall_B", "recall_C", "recall_D",
              "RStd", "RStd_heldout_raw", "RStd_heldout_pride", "RStd_reduction_pct"]:
        print(f"    {k:22s} = {m[k]}")

    ok = True
    if not (m["RStd"] > 0.15):
        print(f"[verify] FAIL: expected substantial raw RStd>0.15, got {m['RStd']}")
        ok = False
    if not (m["RStd_heldout_pride"] < m["RStd_heldout_raw"]):
        print(f"[verify] FAIL: PriDe did not reduce RStd "
              f"({m['RStd_heldout_raw']} -> {m['RStd_heldout_pride']})")
        ok = False
    if ok:
        print(f"[verify] PASS: raw selection bias RStd={m['RStd']} (huge vs ~0 baseline); "
              f"PriDe cut held-out RStd {m['RStd_heldout_raw']}->{m['RStd_heldout_pride']} "
              f"({m['RStd_reduction_pct']}% reduction).")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
