"""
INV-E1 analysis. Reads results/calls.jsonl (no API needed) and produces:
  * per-(condition,judge) pointwise metrics: MAE, Pearson r, calibration
    slope/intercept (logit-logit, verdicts clipped to [0.01,0.99]),
    overconfidence index, flip-set fraction, parse/failure accounting;
  * per-graph manipulability: LLM [min,max] vs exact [min,max] over the SAME
    evaluated sets -> per-graph excess, containment fractions, paired bootstrap CI;
  * skeptical shift (B-A) vs reveal-set size; verbal-numeric delta (C-A);
  * plots in results/plots/, machine-readable results/analysis_summary.json.

Usage: python analysis.py
"""

import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PLOTS = os.path.join(RESULTS, "plots")
os.makedirs(PLOTS, exist_ok=True)

WEAK = "claude-haiku-4-5-20251001"
STRONG = "claude-sonnet-5"

COND_LABEL = {
    ("A", WEAK): "A: numeric-known-credulous (weak)",
    ("A", STRONG): "A: numeric-known-credulous (strong)",
    ("B", WEAK): "B: numeric-known-skeptical (weak)",
    ("C", WEAK): "C: verbal-known-credulous (weak)",
}
GROUP_ORDER = [("A", WEAK), ("A", STRONG), ("B", WEAK), ("C", WEAK)]


def load(path=os.path.join(RESULTS, "calls.jsonl")):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clip01(x, lo=0.01, hi=0.99):
    return np.clip(x, lo, hi)


def logit(p):
    p = clip01(np.asarray(p, float))
    return np.log(p / (1 - p))


def group_rows(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["condition"], r["judge_model"])].append(r)
    return g


def ok_pairs(rs):
    """Return arrays (exact, judge) over parsed calls."""
    e = np.array([r["exact_posterior"] for r in rs if r["parsed_verdict"] is not None])
    j = np.array([r["parsed_verdict"] for r in rs if r["parsed_verdict"] is not None])
    return e, j


def boot_ci(values, stat=np.mean, B=10000, seed=0):
    v = np.asarray(values, float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    bs = stat(v[idx], axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def pointwise_metrics(rs):
    e, j = ok_pairs(rs)
    n_total = len(rs)
    n_ok = len(e)
    m = dict(n_calls=n_total, n_parsed=n_ok,
             parse_rate=(n_ok / n_total if n_total else float("nan")))
    if n_ok < 2:
        return m
    mae = float(np.mean(np.abs(j - e)))
    # bootstrap MAE over graphs
    by_graph = defaultdict(list)
    for r in rs:
        if r["parsed_verdict"] is not None:
            by_graph[r["graph_id"]].append(abs(r["parsed_verdict"] - r["exact_posterior"]))
    graph_mae = [np.mean(v) for v in by_graph.values()]
    m["mae"] = mae
    m["mae_ci"] = boot_ci(graph_mae)
    m["rmse"] = float(np.sqrt(np.mean((j - e) ** 2)))
    m["pearson_r"] = float(stats.pearsonr(e, j)[0])
    m["spearman_r"] = float(stats.spearmanr(e, j)[0])
    # calibration on logit scale
    le, lj = logit(e), logit(j)
    sl, ic, r, p, se = stats.linregress(le, lj)
    m["calib_slope"] = float(sl)
    m["calib_intercept"] = float(ic)
    # overconfidence index
    oc = np.abs(j - 0.5) - np.abs(e - 0.5)
    m["overconfidence"] = float(np.mean(oc))
    m["overconfidence_ci"] = boot_ci(
        [np.mean(np.abs(np.array([r["parsed_verdict"] for r in v]) - 0.5)
                 - np.abs(np.array([r["exact_posterior"] for r in v]) - 0.5))
         for v in _by_graph_rows(rs).values()])
    # signed bias
    m["mean_signed_error"] = float(np.mean(j - e))
    # flip-set fraction: opposite sides of 0.5
    flip = np.mean(((e > 0.5) & (j < 0.5)) | ((e < 0.5) & (j > 0.5)))
    m["flip_fraction"] = float(flip)
    return m


def _by_graph_rows(rs):
    d = defaultdict(list)
    for r in rs:
        if r["parsed_verdict"] is not None:
            d[r["graph_id"]].append(r)
    return d


def manipulability(rs, all_legal_exact=None):
    """Per-graph LLM interval vs exact interval over the SAME evaluated sets."""
    per_graph = {}
    excesses, widths_llm, widths_exact = [], [], []
    llm_contains = exact_contains = neither = 0
    eps = 1e-6
    for gid, grows in _by_graph_rows(rs).items():
        j = np.array([r["parsed_verdict"] for r in grows])
        e = np.array([r["exact_posterior"] for r in grows])
        if len(j) < 2:
            continue
        llm_lo, llm_hi = float(j.min()), float(j.max())
        ex_lo, ex_hi = float(e.min()), float(e.max())
        excess = (ex_lo - llm_lo) + (llm_hi - ex_hi)   # >0 => LLM range wider
        excesses.append(excess)
        widths_llm.append(llm_hi - llm_lo)
        widths_exact.append(ex_hi - ex_lo)
        if llm_lo < ex_lo - eps and llm_hi > ex_hi + eps:
            llm_contains += 1
        elif ex_lo < llm_lo - eps and ex_hi > llm_hi + eps:
            exact_contains += 1
        else:
            neither += 1
        per_graph[gid] = dict(llm=[llm_lo, llm_hi], exact=[ex_lo, ex_hi],
                              excess=excess, n_sets=len(j))
    ng = len(excesses)
    out = dict(
        n_graphs=ng,
        mean_excess=float(np.mean(excesses)) if ng else float("nan"),
        mean_excess_ci=boot_ci(excesses),
        median_excess=float(np.median(excesses)) if ng else float("nan"),
        mean_width_llm=float(np.mean(widths_llm)) if ng else float("nan"),
        mean_width_exact=float(np.mean(widths_exact)) if ng else float("nan"),
        frac_llm_strictly_contains_exact=llm_contains / ng if ng else float("nan"),
        frac_exact_strictly_contains_llm=exact_contains / ng if ng else float("nan"),
        frac_neither=neither / ng if ng else float("nan"),
        per_graph=per_graph)
    return out


def matched(rs_a, rs_b):
    """Match rows on (graph_id,set_id); return list of (row_a,row_b) with parsed."""
    idx = {(r["graph_id"], r["set_id"]): r for r in rs_b if r["parsed_verdict"] is not None}
    pairs = []
    for r in rs_a:
        if r["parsed_verdict"] is None:
            continue
        k = (r["graph_id"], r["set_id"])
        if k in idx:
            pairs.append((r, idx[k]))
    return pairs


def skeptical_shift(groups):
    """B - A (weak) on matched sets, vs reveal-set size."""
    A = groups.get(("A", WEAK), [])
    B = groups.get(("B", WEAK), [])
    pairs = matched(B, A)  # (B_row, A_row)
    if not pairs:
        return {}
    shift = np.array([b["parsed_verdict"] - a["parsed_verdict"] for b, a in pairs])
    nrev = np.array([b["n_revealed"] for b, a in pairs])
    out = dict(n=len(pairs),
               mean_shift=float(np.mean(shift)),
               mean_shift_ci=boot_ci(shift),
               shift_sparse=float(np.mean(shift[nrev <= 1])) if np.any(nrev <= 1) else float("nan"),
               n_sparse=int(np.sum(nrev <= 1)),
               shift_dense=float(np.mean(shift[nrev >= 3])) if np.any(nrev >= 3) else float("nan"),
               n_dense=int(np.sum(nrev >= 3)))
    if len(set(nrev.tolist())) > 1:
        sl, ic, r, p, se = stats.linregress(nrev, shift)
        out["slope_vs_nrev"] = float(sl)
        out["slope_p"] = float(p)
    return out, pairs


def verbal_numeric_delta(groups):
    """C - A (weak) on matched sets; also error comparison vs exact."""
    A = groups.get(("A", WEAK), [])
    C = groups.get(("C", WEAK), [])
    pairs = matched(C, A)
    if not pairs:
        return {}, []
    delta = np.array([c["parsed_verdict"] - a["parsed_verdict"] for c, a in pairs])
    err_c = np.array([abs(c["parsed_verdict"] - c["exact_posterior"]) for c, a in pairs])
    err_a = np.array([abs(a["parsed_verdict"] - a["exact_posterior"]) for c, a in pairs])
    out = dict(n=len(pairs),
               mean_abs_delta=float(np.mean(np.abs(delta))),
               mean_signed_delta=float(np.mean(delta)),
               mean_signed_delta_ci=boot_ci(delta),
               mae_verbal=float(np.mean(err_c)),
               mae_numeric_matched=float(np.mean(err_a)),
               mae_increase=float(np.mean(err_c) - np.mean(err_a)),
               mae_increase_ci=boot_ci(err_c - err_a))
    return out, pairs


# ============================ plots ============================

def plot_scatter(groups):
    fig, axes = plt.subplots(1, len(GROUP_ORDER), figsize=(4 * len(GROUP_ORDER), 4))
    for ax, key in zip(axes, GROUP_ORDER):
        rs = groups.get(key, [])
        e, j = ok_pairs(rs)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        if len(e):
            ax.scatter(e, j, s=14, alpha=0.5, color="#2b6cb0")
            mae = np.mean(np.abs(j - e))
            r = stats.pearsonr(e, j)[0] if len(e) > 1 else float("nan")
            ax.set_title(f"{key[0]} ({'weak' if key[1]==WEAK else 'strong'})\n"
                         f"MAE={mae:.3f} r={r:.3f}", fontsize=9)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("exact posterior"); ax.set_ylabel("judge verdict")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "scatter_judge_vs_exact.png"), dpi=130)
    plt.close(fig)


def plot_manipulability(groups):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, key in zip(axes, [("A", WEAK), ("A", STRONG)]):
        rs = groups.get(key, [])
        man = manipulability(rs)
        pg = man["per_graph"]
        gids = sorted(pg.keys())
        for i, gid in enumerate(gids):
            ex = pg[gid]["exact"]; ll = pg[gid]["llm"]
            ax.plot([i - 0.15, i - 0.15], ex, color="#2f855a", lw=5, alpha=0.8,
                    solid_capstyle="butt", label="exact" if i == 0 else "")
            ax.plot([i + 0.15, i + 0.15], ll, color="#c05621", lw=5, alpha=0.8,
                    solid_capstyle="butt", label="LLM" if i == 0 else "")
        ax.set_xticks(range(len(gids))); ax.set_xticklabels(gids, fontsize=7)
        ax.set_xlabel("graph id"); ax.set_ylabel("posterior / verdict range")
        ax.set_title(f"Manipulability intervals — cond {key[0]} "
                     f"({'weak' if key[1]==WEAK else 'strong'})\n"
                     f"mean excess={man['mean_excess']:.3f} "
                     f"CI[{man['mean_excess_ci'][0]:.3f},{man['mean_excess_ci'][1]:.3f}]",
                     fontsize=9)
        ax.legend(fontsize=8); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "manipulability_intervals.png"), dpi=130)
    plt.close(fig)


def plot_overconfidence(groups, summary):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    keys = [k for k in GROUP_ORDER if k in groups]
    labels = [f"{k[0]}/{'wk' if k[1]==WEAK else 'st'}" for k in keys]
    vals = [summary["groups"][COND_LABEL[k]].get("overconfidence", np.nan) for k in keys]
    cis = [summary["groups"][COND_LABEL[k]].get("overconfidence_ci", [np.nan, np.nan]) for k in keys]
    x = range(len(keys))
    err = [[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]]
    ax.bar(x, vals, color="#6b46c1", alpha=0.8)
    ax.errorbar(x, vals, yerr=err, fmt="none", ecolor="k", capsize=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("overconfidence  mean(|j-0.5| - |e-0.5|)")
    ax.set_title("Overconfidence index by condition (>0 = judge more extreme than Bayes)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "overconfidence.png"), dpi=130)
    plt.close(fig)


def plot_skeptical(pairs):
    if not pairs:
        return
    shift = np.array([b["parsed_verdict"] - a["parsed_verdict"] for b, a in pairs])
    nrev = np.array([b["n_revealed"] for b, a in pairs])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    jitter = (np.random.default_rng(0).random(len(nrev)) - 0.5) * 0.25
    ax.scatter(nrev + jitter, shift, s=16, alpha=0.45, color="#b83280")
    ax.axhline(0, color="k", lw=0.8)
    # mean per size
    for k in sorted(set(nrev.tolist())):
        mk = nrev == k
        ax.plot(k, shift[mk].mean(), "D", color="black", ms=7)
    ax.set_xlabel("reveal-set size |S|")
    ax.set_ylabel("skeptical shift  (verdict_B - verdict_A)")
    ax.set_title("Skeptical-instruction shift vs disclosure (black = mean per size)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "skeptical_shift.png"), dpi=130)
    plt.close(fig)


def plot_verbal(pairs):
    if not pairs:
        return
    a = np.array([p[1]["parsed_verdict"] for p in pairs])   # numeric
    c = np.array([p[0]["parsed_verdict"] for p in pairs])   # verbal
    e = np.array([p[0]["exact_posterior"] for p in pairs])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    sc = ax.scatter(a, c, c=e, cmap="viridis", s=22, alpha=0.8)
    ax.set_xlabel("verdict — numeric (A)"); ax.set_ylabel("verdict — verbal (C)")
    ax.set_title("Verbal vs numeric verdicts on matched sets\n(color = exact posterior)",
                 fontsize=10)
    fig.colorbar(sc, ax=ax, label="exact")
    ax.set_aspect("equal"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "verbal_vs_numeric.png"), dpi=130)
    plt.close(fig)


def plot_calibration(groups):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="identity")
    colors = {("A", WEAK): "#2b6cb0", ("A", STRONG): "#2f855a",
              ("B", WEAK): "#c05621", ("C", WEAK): "#b83280"}
    for key in GROUP_ORDER:
        rs = groups.get(key, [])
        e, j = ok_pairs(rs)
        if len(e) < 5:
            continue
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(e, bins) - 1
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 2:
                xs.append(e[m].mean()); ys.append(j[m].mean())
        ax.plot(xs, ys, "-o", ms=4, color=colors[key],
                label=f"{key[0]}/{'wk' if key[1]==WEAK else 'st'}")
    ax.set_xlabel("exact posterior (bin mean)"); ax.set_ylabel("mean judge verdict")
    ax.set_title("Calibration curves", fontsize=10); ax.legend(fontsize=8)
    ax.set_aspect("equal"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "calibration_curves.png"), dpi=130)
    plt.close(fig)


def main(calls_path=None):
    global PLOTS
    rows = load(calls_path) if calls_path else load()
    # keep non-canonical (e.g. smoke) outputs separate from the real results
    base = os.path.basename(calls_path) if calls_path else "calls.jsonl"
    tag = "" if base == "calls.jsonl" else "_" + base.replace("_calls.jsonl", "").replace(".jsonl", "")
    summary_path = os.path.join(RESULTS, f"analysis_summary{tag}.json")
    PLOTS = os.path.join(RESULTS, "plots" + (tag or ""))
    os.makedirs(PLOTS, exist_ok=True)
    groups = group_rows(rows)
    summary = {"n_calls_total": len(rows),
               "n_parsed_total": sum(1 for r in rows if r["parsed_verdict"] is not None),
               "judge_models": {"weak": WEAK, "strong": STRONG},
               "groups": {}, "manipulability": {}, "skeptical": {},
               "verbal_numeric": {}}

    for key in GROUP_ORDER:
        rs = groups.get(key, [])
        if not rs:
            continue
        summary["groups"][COND_LABEL[key]] = pointwise_metrics(rs)

    for key in [("A", WEAK), ("A", STRONG), ("B", WEAK), ("C", WEAK)]:
        rs = groups.get(key, [])
        if rs:
            summary["manipulability"][COND_LABEL[key]] = {
                k: v for k, v in manipulability(rs).items() if k != "per_graph"}

    sk = skeptical_shift(groups)
    if sk:
        summary["skeptical"], sk_pairs = sk
    else:
        sk_pairs = []
    vn = verbal_numeric_delta(groups)
    if vn:
        summary["verbal_numeric"], vn_pairs = vn
    else:
        vn_pairs = []

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # plots
    plot_scatter(groups)
    plot_manipulability(groups)
    plot_overconfidence(groups, summary)
    plot_calibration(groups)
    plot_skeptical(sk_pairs)
    plot_verbal(vn_pairs)

    # console report
    print("=" * 78)
    print(f"INV-E1 ANALYSIS  |  calls={summary['n_calls_total']} "
          f"parsed={summary['n_parsed_total']}")
    print("=" * 78)
    for key in GROUP_ORDER:
        lbl = COND_LABEL[key]
        if lbl not in summary["groups"]:
            continue
        m = summary["groups"][lbl]
        print(f"\n[{lbl}]  n={m['n_calls']} parsed={m['n_parsed']} "
              f"parse_rate={m['parse_rate']:.3f}")
        if "mae" in m:
            print(f"   MAE={m['mae']:.4f} CI{_fmt(m['mae_ci'])}  RMSE={m['rmse']:.4f}  "
                  f"Pearson r={m['pearson_r']:.4f}  Spearman={m['spearman_r']:.4f}")
            print(f"   calib slope={m['calib_slope']:.3f} intercept={m['calib_intercept']:.3f}  "
                  f"overconf={m['overconfidence']:.4f} CI{_fmt(m['overconfidence_ci'])}")
            print(f"   signed err={m['mean_signed_error']:+.4f}  flip_frac={m['flip_fraction']:.4f}")
        if lbl in summary["manipulability"]:
            mn = summary["manipulability"][lbl]
            print(f"   manip: mean_excess={mn['mean_excess']:+.4f} "
                  f"CI{_fmt(mn['mean_excess_ci'])}  "
                  f"widthLLM={mn['mean_width_llm']:.3f} widthExact={mn['mean_width_exact']:.3f}")
            print(f"          LLM⊃exact={mn['frac_llm_strictly_contains_exact']:.2f} "
                  f"exact⊃LLM={mn['frac_exact_strictly_contains_llm']:.2f} "
                  f"neither={mn['frac_neither']:.2f} (n_graphs={mn['n_graphs']})")
    if summary["skeptical"]:
        s = summary["skeptical"]
        print(f"\n[SKEPTICAL SHIFT B-A]  n={s['n']}  mean={s['mean_shift']:+.4f} "
              f"CI{_fmt(s['mean_shift_ci'])}")
        print(f"   sparse|S|<=1: {s['shift_sparse']:+.4f} (n={s['n_sparse']})   "
              f"dense|S|>=3: {s['shift_dense']:+.4f} (n={s['n_dense']})   "
              f"slope/|S|={s.get('slope_vs_nrev', float('nan')):+.4f} "
              f"(p={s.get('slope_p', float('nan')):.3f})")
    if summary["verbal_numeric"]:
        v = summary["verbal_numeric"]
        print(f"\n[VERBAL - NUMERIC C-A]  n={v['n']}  |delta|={v['mean_abs_delta']:.4f}  "
              f"signed={v['mean_signed_delta']:+.4f} CI{_fmt(v['mean_signed_delta_ci'])}")
        print(f"   MAE verbal={v['mae_verbal']:.4f}  MAE numeric(matched)={v['mae_numeric_matched']:.4f}  "
              f"increase={v['mae_increase']:+.4f} CI{_fmt(v['mae_increase_ci'])}")
    print("\nplots ->", PLOTS)
    print("summary ->", summary_path)


def _fmt(ci):
    return f"[{ci[0]:.4f},{ci[1]:.4f}]"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", default=os.path.join(RESULTS, "calls.jsonl"),
                    help="path to a calls.jsonl to analyze")
    args = ap.parse_args()
    main(args.calls)
