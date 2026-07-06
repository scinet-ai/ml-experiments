"""Recompute MCQ selection-bias metrics + PriDe residual from committed score CSVs.

ZERO-DOWNLOAD, numpy/pandas only (no torch, no model, no GPU). Reads the raw
per-(question, shift) option-ID logits produced by eval.py and computes, per model:

  - accuracy (mean over the 4 cyclic shifts)
  - recall per option ID  = P(model predicts label X)   (unbiased => 0.25 each)
  - RStd                  = std of the 4 recalls          (selection-bias magnitude)
  - acc_std_over_perm     = std of accuracy across the 4 cyclic shifts
  - PriDe: estimate the option-ID log-prior from a held-out estimation subset
    (mean centred log-softmax per label over its cyclic shifts), subtract it in
    log space, re-argmax, and report post-debiasing accuracy, RStd and residual.

Selection-bias model (Zheng et al. 2024): observed label logit = f(content) + b_label.
Over the 4 cyclic shifts each content visits each label once, so averaging the
log-softmax per label marginalises content and isolates the label prior b.

Usage:
  python analyze.py --scores results/scores_*.csv --out results/summary.csv \
      --est-frac 0.25 --seed 0
"""
import argparse, glob, json
import numpy as np
import pandas as pd

LETTERS = ["A", "B", "C", "D"]
LOGIT_COLS = ["logit_A", "logit_B", "logit_C", "logit_D"]


def log_softmax(logits):
    m = logits.max(axis=1, keepdims=True)
    z = logits - m
    lse = np.log(np.exp(z).sum(axis=1, keepdims=True))
    return z - lse


def metrics_from_pred(pred_labels, correct_labels):
    """pred_labels, correct_labels: arrays of ints 0..3."""
    acc = float((pred_labels == correct_labels).mean())
    recalls = np.array([(pred_labels == i).mean() for i in range(4)])
    rstd = float(recalls.std())
    return acc, recalls, rstd


def analyze_model(df, est_frac, seed):
    df = df.sort_values(["qid", "shift"]).reset_index(drop=True)
    logits = df[LOGIT_COLS].to_numpy(dtype=np.float64)
    correct = df["correct_label"].map({L: i for i, L in enumerate(LETTERS)}).to_numpy()
    qid = df["qid"].to_numpy()
    shift = df["shift"].to_numpy()

    logp = log_softmax(logits)  # (N,4)
    pred = logits.argmax(axis=1)

    # --- raw selection bias ---
    acc, recalls, rstd = metrics_from_pred(pred, correct)
    # accuracy variance across the 4 cyclic shift positions
    accs_per_shift = [float((pred[shift == s] == correct[shift == s]).mean()) for s in range(4)]
    acc_std_perm = float(np.std(accs_per_shift))

    # --- PriDe prior estimation on held-out estimation subset ---
    uq = np.unique(qid)
    rng = np.random.RandomState(seed)
    est_q = set(rng.choice(uq, size=max(1, int(round(est_frac * len(uq)))), replace=False).tolist())
    est_mask = np.array([q in est_q for q in qid])

    # prior_i = mean over estimation rows of log-softmax at label i (content marginalised
    # because each est question contributes all 4 cyclic shifts); then centre.
    prior = logp[est_mask].mean(axis=0)
    prior = prior - prior.mean()

    # debias: subtract prior in log space, re-argmax
    logp_deb = logp - prior[None, :]
    pred_deb = logp_deb.argmax(axis=1)

    # residual measured on questions NOT used for estimation (clean held-out)
    ev = ~est_mask
    acc_deb, recalls_deb, rstd_deb = metrics_from_pred(pred_deb[ev], correct[ev])
    # also raw metrics on the same held-out slice for apples-to-apples reduction
    acc_raw_ev, recalls_raw_ev, rstd_raw_ev = metrics_from_pred(pred[ev], correct[ev])
    accs_per_shift_deb = [
        float((pred_deb[ev & (shift == s)] == correct[ev & (shift == s)]).mean()) for s in range(4)
    ]
    acc_std_perm_deb = float(np.std(accs_per_shift_deb))

    return {
        "model": df["model"].iloc[0],
        "revision": df["revision"].iloc[0],
        "n_questions": int(len(uq)),
        "n_est_questions": len(est_q),
        "acc": round(acc, 4),
        "recall_A": round(recalls[0], 4),
        "recall_B": round(recalls[1], 4),
        "recall_C": round(recalls[2], 4),
        "recall_D": round(recalls[3], 4),
        "RStd": round(rstd, 4),
        "acc_std_over_perm": round(acc_std_perm, 4),
        # held-out slice, raw vs PriDe
        "acc_heldout_raw": round(acc_raw_ev, 4),
        "RStd_heldout_raw": round(rstd_raw_ev, 4),
        "acc_heldout_pride": round(acc_deb, 4),
        "RStd_heldout_pride": round(rstd_deb, 4),
        "acc_std_over_perm_pride": round(acc_std_perm_deb, 4),
        "RStd_reduction_pct": round(100 * (rstd_raw_ev - rstd_deb) / rstd_raw_ev, 1) if rstd_raw_ev > 0 else 0.0,
        "prior_A": round(prior[0], 4),
        "prior_B": round(prior[1], 4),
        "prior_C": round(prior[2], 4),
        "prior_D": round(prior[3], 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True, help="one or more score CSV globs")
    ap.add_argument("--out", default="results/summary.csv")
    ap.add_argument("--est-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = []
    for g in args.scores:
        files.extend(sorted(glob.glob(g)))
    if not files:
        raise SystemExit(f"no score files matched {args.scores}")

    rows = []
    for fp in files:
        df = pd.read_csv(fp)
        for _, sub in df.groupby("model"):
            rows.append(analyze_model(sub, args.est_frac, args.seed))

    summ = pd.DataFrame(rows)
    # order by parameter count if inferable from name
    def pcount(name):
        import re
        m = re.search(r"(\d+)m", name.lower())
        if m:
            return int(m.group(1)) * 1e6
        m = re.search(r"(\d+(?:\.\d+)?)b", name.lower())
        if m:
            return float(m.group(1)) * 1e9
        return 0
    summ["_p"] = summ["model"].map(pcount)
    summ = summ.sort_values("_p").drop(columns="_p").reset_index(drop=True)
    summ.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(summ.to_string(index=False))
    print(f"\n[analyze] wrote {args.out}")


if __name__ == "__main__":
    main()
