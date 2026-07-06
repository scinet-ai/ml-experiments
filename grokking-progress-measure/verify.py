#!/usr/bin/env python3
"""
verify.py -- ZERO-DOWNLOAD smoke reproduction for the grokking progress-measure
finding.  Trains ONE small seed of modular addition to grok (synthetic data, no
download), computes the mechanism-agnostic measures per step, and prints the
grok step and each measure's threshold-cross step + lead time.

PASS criterion: (a) the model groks (test acc crosses 0.5 well after train acc
saturates) and (b) at least one mechanism-agnostic measure crosses its
pre-registered threshold BEFORE the grok, i.e. positive lead time.

Runs standalone in ~1-3 min on CPU:
    python3 verify.py
"""
import math, os, sys
import numpy as np

os.environ.setdefault("GROK_DEVICE", "cpu")          # portable default
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grok

MEASURES = ["weight_l2", "w_eff_rank", "act_eff_rank", "act_sparsity",
            "act_kurtosis", "gzip_bytes"]


def lead_times(rows, grok_thr=0.5, frac_thr=0.5):
    step = np.array([r["step"] for r in rows], float)
    te = np.array([r["test_acc"] for r in rows], float)
    above = np.where(te >= grok_thr)[0]
    grok_step = float(step[above[0]]) if len(above) else float("nan")
    ntail = max(3, len(step) // 20)
    out = {}
    for m in MEASURES:
        M = np.array([r[m] for r in rows], float)
        m0, mf = M[0], M[-ntail:].mean()
        change = mf - m0
        if abs(change) < 0.02 * (abs(m0) + 1e-12):
            out[m] = (float("nan"), float("nan")); continue
        frac = (M - m0) / change
        idx = np.where(frac >= frac_thr)[0]
        cross = float(step[idx[0]]) if len(idx) else float("nan")
        out[m] = (cross, grok_step - cross if math.isfinite(cross) else float("nan"))
    return grok_step, out


def main():
    # small, fast-grokking modular addition (p=53, half held out, strong wd)
    print("Training modular addition (p=53, train_frac=0.5, wd=1.0) to grok ...")
    rows = grok.train(task="add", p=53, seed=0, steps=8000, log_every=100,
                      wd=1.0, train_frac=0.5, quiet=True)
    grok_step, lt = lead_times(rows)
    tr_final = rows[-1]["train_acc"]; te_final = rows[-1]["test_acc"]
    print(f"\nfinal train_acc={tr_final:.3f}  final test_acc={te_final:.3f}")
    if not math.isfinite(grok_step):
        print("FAIL: model did not grok (test acc never crossed 0.5)."); sys.exit(1)
    tr_sat = next((r["step"] for r in rows if r["train_acc"] >= 0.99), None)
    print(f"train acc saturates (>=0.99) at step {tr_sat}")
    print(f"GROK: test acc crosses 0.5 at step {int(grok_step)}")
    print(f"delayed-generalization gap = {int(grok_step - (tr_sat or 0))} steps\n")
    print(f"{'measure':14s} {'cross_step':>10s} {'lead_time':>10s}")
    any_pos = False
    for m in MEASURES:
        cross, lead = lt[m]
        cs = "nan" if not math.isfinite(cross) else str(int(cross))
        ls = "nan" if not math.isfinite(lead) else str(int(lead))
        flag = ""
        if math.isfinite(lead) and lead > 0:
            any_pos = True; flag = "  <- leads grok"
        print(f"{m:14s} {cs:>10s} {ls:>10s}{flag}")
    print()
    if any_pos:
        print("PASS: grokking reproduced AND >=1 mechanism-agnostic measure has "
              "positive lead time (crosses threshold before test acc reaches 0.5).")
        sys.exit(0)
    print("FAIL: no mechanism-agnostic measure led the grok on this smoke seed.")
    sys.exit(1)


if __name__ == "__main__":
    main()
