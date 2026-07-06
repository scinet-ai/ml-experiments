"""
analyze.py — Compute grok step, pre-registered threshold crossing, lead time,
and Spearman correlations for every logged run, then build per-run and summary
tables (and optional curve plots).

Pre-registered, mechanism-agnostic threshold rule (no test-acc / circuit info):
  For measure M(t) with reference m0 = M(step 0) and converged value
  m_final = mean of the last few logged M values, the normalised progress toward
  the converged value is
        frac(t) = (M(t) - m0) / (m_final - m0)
  (This is direction-agnostic: it goes 0 -> 1 whether M rises or falls.)  The
  threshold-cross step is the first t with frac(t) >= 0.5 (measure halfway to its
  converged value).  A measure whose net change |m_final - m0| is below a small
  relative floor is 'uninformative' for that run (cross = NaN).

  grok_step   = first step with test_acc >= 0.5
  lead_time   = grok_step - cross_step        (POSITIVE => measure leads grok)

Spearman:
  sp_conc   = Spearman( M(t), test_acc(t) )                    over all t
  sp_future = Spearman( M(t), test_acc(t + HORIZON_STEPS) )    over valid t
Signs are reported as-is (a measure that falls as test rises has negative rho).
"""
import csv, glob, math, os, sys
import numpy as np

MEASURES = ["weight_l2", "w_eff_rank", "act_eff_rank", "act_sparsity",
            "act_kurtosis", "gzip_bytes"]
GROK_THR = 0.5
# Per-task grok marker = midpoint between chance and perfect test accuracy.
# Modular (p classes, chance ~0.01): 0.5.  Parity (2 classes, chance 0.5): 0.75.
GROK_THR_BY_TASK = {"add": 0.5, "mul": 0.5, "parity": 0.75}
FRAC_THR = 0.5
REL_FLOOR = 0.02           # min |net change| / |m0| for a measure to be informative
HORIZON_STEPS = 1500       # 'future' horizon for predictive Spearman
HERE = os.path.dirname(os.path.abspath(__file__))
CSVDIR = os.path.join(HERE, "csv")


def rankdata(a):
    a = np.asarray(a, float)
    order = a.argsort()
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(len(a))
    # average ties
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    start = csum - cnt
    avg = (start + csum - 1) / 2.0
    return avg[inv]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx, ry = rankdata(x[m]), rankdata(y[m])
    rx -= rx.mean(); ry -= ry.mean()
    denom = math.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    d = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return d


def analyze_run(d, grok_thr=GROK_THR):
    step = d["step"]; te = d["test_acc"]
    above = np.where(te >= grok_thr)[0]
    grok_step = float(step[above[0]]) if len(above) else float("nan")
    ntail = max(3, len(step) // 20)
    out = {}
    for meas in MEASURES:
        M = d[meas]
        m0 = M[0]
        m_final = M[-ntail:].mean()
        change = m_final - m0
        if abs(change) < REL_FLOOR * (abs(m0) + 1e-12):
            cross_step, lead = float("nan"), float("nan")
        else:
            frac = (M - m0) / change
            idx = np.where(frac >= FRAC_THR)[0]
            cross_step = float(step[idx[0]]) if len(idx) else float("nan")
            lead = grok_step - cross_step if math.isfinite(cross_step) else float("nan")
        # cross_frac = cross_step / grok_step: fraction of the pre-grok window
        # elapsed at the threshold crossing.  ~0 => trivially early (dominated by
        # the initial weight-decay transient); ~1 => tight, transition-specific
        # predictor; >1 => the measure LAGS the grok.
        cross_frac = (cross_step / grok_step) if (math.isfinite(cross_step)
                     and math.isfinite(grok_step) and grok_step > 0) else float("nan")
        sp_conc = spearman(M, te)
        # future-horizon spearman: pair M(t) with te at step ~t+HORIZON
        h = max(1, int(round(HORIZON_STEPS / (step[1] - step[0])))) if len(step) > 1 else 1
        sp_future = spearman(M[:-h], te[h:]) if len(step) > h + 3 else float("nan")
        out[meas] = dict(cross_step=cross_step, lead_time=lead, cross_frac=cross_frac,
                         sp_conc=sp_conc, sp_future=sp_future,
                         m0=m0, m_final=m_final)
    return grok_step, out


def parse_name(fn):
    base = os.path.basename(fn)[:-4]
    if not any(base.startswith(t + "_s") for t in ("add", "mul", "parity")):
        return None
    task, srest = base.split("_s", 1)
    try:
        seed = int(srest)
    except ValueError:
        return None
    return task, seed


def plot_curves(tasks):
    """One PNG per task (seed 0): test acc + normalised measures vs step."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plotting skipped: {e})"); return
    for task in tasks:
        fn = os.path.join(CSVDIR, f"{task}_s0.csv")
        if not os.path.exists(fn):
            continue
        d = load(fn)
        step = d["step"]
        thr = GROK_THR_BY_TASK.get(task, GROK_THR)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(step, d["test_acc"], "k-", lw=2.4, label="test acc", zorder=5)
        ax.plot(step, d["train_acc"], color="0.6", lw=1.2, label="train acc")
        ax.axhline(thr, color="k", ls=":", lw=0.8)
        for meas in MEASURES:
            M = d[meas]; m0 = M[0]; mf = M[-max(3, len(M)//20):].mean()
            if abs(mf - m0) < 1e-9:
                continue
            ax.plot(step, (M - m0) / (mf - m0), lw=1.3, alpha=0.85,
                    label=f"{meas} (norm.)")
        above = np.where(d["test_acc"] >= thr)[0]
        if len(above):
            ax.axvline(step[above[0]], color="red", ls="--", lw=1.2,
                       label=f"grok (test>{thr:g}) @ {int(step[above[0]])}")
        ax.set_xlabel("training step"); ax.set_ylabel("acc / normalised measure")
        ax.set_ylim(-0.15, 1.2)
        ax.set_title(f"{task}: grokking & mechanism-agnostic measures (seed 0)")
        ax.legend(fontsize=7, ncol=2, loc="center right")
        fig.tight_layout()
        out = os.path.join(HERE, f"curves_{task}.png")
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"wrote {out}")


def main():
    files = sorted(glob.glob(os.path.join(CSVDIR, "*.csv")))
    per_run = []
    for fn in files:
        pn = parse_name(fn)
        if pn is None:
            continue
        task, seed = pn
        d = load(fn)
        grok_step, res = analyze_run(d, GROK_THR_BY_TASK.get(task, GROK_THR))
        for meas, r in res.items():
            per_run.append(dict(task=task, seed=seed, grok_step=grok_step,
                                grokked=int(math.isfinite(grok_step)),
                                measure=meas, **r))
    if not per_run:
        print("no run CSVs found (expect <task>_s<seed>.csv)"); return

    # write per-run table
    keys = ["task", "seed", "measure", "grokked", "grok_step", "cross_step",
            "lead_time", "cross_frac", "sp_conc", "sp_future", "m0", "m_final"]
    with open(os.path.join(HERE, "results_per_run.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in per_run:
            w.writerow({k: r[k] for k in keys})

    # summary per (task, measure) over grokked seeds
    tasks = sorted({r["task"] for r in per_run})
    summary = []
    for task in tasks:
        for meas in MEASURES:
            rr = [r for r in per_run if r["task"] == task and r["measure"] == meas
                  and r["grokked"]]
            n = len(rr)
            leads = np.array([r["lead_time"] for r in rr], float)
            n_pos = int(np.nansum(leads > 0))
            n_valid = int(np.sum(np.isfinite(leads)))
            summary.append(dict(
                task=task, measure=meas, n_seeds=n, n_valid_lead=n_valid,
                n_pos_lead=n_pos,
                mean_lead=float(np.nanmean(leads)) if n_valid else float("nan"),
                median_lead=float(np.nanmedian(leads)) if n_valid else float("nan"),
                mean_cross_frac=float(np.nanmean([r["cross_frac"] for r in rr])) if n else float("nan"),
                mean_sp_conc=float(np.nanmean([r["sp_conc"] for r in rr])) if n else float("nan"),
                mean_sp_future=float(np.nanmean([r["sp_future"] for r in rr])) if n else float("nan"),
            ))
    with open(os.path.join(HERE, "results_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader()
        w.writerows(summary)

    # verdict: a single measure with positive lead on ALL grokked seeds of ALL 3 tasks
    print("\n=== grok reproduced ===")
    for task in tasks:
        seeds = sorted({r["seed"] for r in per_run if r["task"] == task})
        gk = sorted({r["seed"] for r in per_run if r["task"] == task and r["grokked"]})
        gsteps = sorted({int(r["grok_step"]) for r in per_run
                         if r["task"] == task and r["grokked"] and r["measure"] == MEASURES[0]})
        print(f"  {task:7s}: {len(gk)}/{len(seeds)} seeds grokked; grok steps {gsteps}")

    print("\n=== per-(task,measure): pos-lead/valid  L=mean-lead  cf=cross-frac  r=Spearman ===")
    print("    (cf ~0 => trivially early transient; cf ~1 => tight predictor; cf>1 => lags)")
    hdr = f"{'measure':14s}" + "".join(f"{t:>26s}" for t in tasks)
    print(hdr)
    for meas in MEASURES:
        line = f"{meas:14s}"
        for task in tasks:
            s = next(s for s in summary if s["task"] == task and s["measure"] == meas)
            line += f"  {s['n_pos_lead']}/{s['n_valid_lead']} L={s['mean_lead']:.0f} cf={s['mean_cross_frac']:.2f} r={s['mean_sp_conc']:+.2f}".rjust(26)
        print(line)

    print("\n=== VERDICT ===")
    n_tasks = len(tasks)
    winners = []
    for meas in MEASURES:
        ok = True
        for task in tasks:
            s = next(s for s in summary if s["task"] == task and s["measure"] == meas)
            # require every grokked seed to have a positive, valid lead
            if s["n_seeds"] == 0 or s["n_pos_lead"] < s["n_seeds"] or s["n_valid_lead"] < s["n_seeds"]:
                ok = False; break
        if ok and n_tasks >= 3:
            winners.append(meas)
    if winners:
        print("SUCCESS-candidate measures (positive lead on ALL grokked seeds of all "
              f"{n_tasks} tasks): {winners}")
    else:
        print(f"No single measure gives positive lead on ALL grokked seeds across all "
              f"{n_tasks} task(s) -> PARTIAL/NEGATIVE. See results_summary.csv.")
    print("\nWrote results_per_run.csv and results_summary.csv")
    plot_curves(tasks)


if __name__ == "__main__":
    main()
