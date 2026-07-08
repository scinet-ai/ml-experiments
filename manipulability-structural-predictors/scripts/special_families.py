"""Special families: chains and stars.

STAR:  root + k independent leaf children, each a premise supporting (L>1) or
attacking (L<1) the root. We derive and VERIFY (against the package exact DP) a
closed-form law for star manipulability width, then measure how well the flat-star
law extrapolates as an approximation to the general tree corpus (its standalone R^2).

CHAIN: a single directed path into the root; shows depth attenuation of a fixed
amount of evidence.

Reveal model recap (target prior fixed 0.5): revealing a leaf child i puts input
(leff_i, pp_i=prior_i) into the root's Jeffrey marginal, where the judge may use the
true LR or the direction-only type-default (2.0 if L>1 else 0.5), picking whichever
is more extreme.  So:
    leff^max_i = max(L_i, 2.0)   for pro children (L_i>1)
    leff^min_i = min(L_i, 0.5)   for con children (L_i<1)
    max_post = J(0.5, {(leff^max_i, prior_i): L_i>1})   [reveal only pro]
    min_post = J(0.5, {(leff^min_i, prior_i): L_i<1})   [reveal only con]
    width    = max_post - min_post
where J(p0,S) = sum_{T subseteq S} prod_{i in T} prior_i * prod_{i notin T}(1-prior_i)
                * sigmoid( logit(p0) + sum_{i in T} log leff_i ).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probability_flow.aspic.argument import Conclusion, Premise
from probability_flow.core import IndependentEvidenceCPD
from probability_flow.metrics import posterior_range

PLOTS = "results/plots"
_CLAMP = 1e-3


def _clamp(p):
    return min(1 - _CLAMP, max(_CLAMP, p))


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _J(p0, configs):
    if not configs:
        return p0
    base = math.log(p0 / (1 - p0))
    k = len(configs)
    total = 0.0
    for combo in range(1 << k):
        lo = base
        w = 1.0
        for i, (leff, pp) in enumerate(configs):
            if (combo >> i) & 1:
                lo += math.log(max(leff, 1e-9))
                w *= pp
            else:
                w *= 1.0 - pp
        total += _sigmoid(lo) * w
    return total


def star_width_from_children(children):
    """children: list of (L, prior). Returns closed-form (min_post, max_post, width)."""
    pro = [(max(L, 2.0), _clamp(pi)) for (L, pi) in children if L > 1.0]
    con = [(min(L, 0.5), _clamp(pi)) for (L, pi) in children if L < 1.0]
    hi = _clamp(_J(0.5, pro))
    lo = _clamp(_J(0.5, con))
    return lo, hi, hi - lo


def build_star(children, root_prior=0.5):
    """children: list of (L, prior). L>1 -> support, L<1 -> rebut."""
    root = Conclusion("Root", prior=root_prior)
    for j, (L, pi) in enumerate(children):
        leaf = Premise(f"P{j}", prior=pi)
        if L > 1.0:
            root.support(leaf, lr=L)
        else:
            root.rebut(leaf, lr=L)
    return root.assemble()


def verify_stars():
    rng = np.random.default_rng(20240706)
    rows = []
    max_gap = 0.0
    for t in range(400):
        k = int(rng.integers(1, 6))
        children = []
        for _ in range(k):
            up = rng.random() < 0.6
            if up:
                L = float(np.round(math.exp(rng.uniform(math.log(1.1), math.log(20.0))), 4))
            else:
                L = float(np.round(math.exp(rng.uniform(math.log(1 / 20.0), math.log(1 / 1.1))), 4))
            pi = float(np.round(rng.uniform(0.2, 0.9), 4))
            children.append((L, pi))
        arg = build_star(children, root_prior=float(np.round(rng.uniform(0.3, 0.7), 4)))
        res = posterior_range(arg.bn, arg.target, exact=True)
        assert res.is_exact
        plo, phi = float(res[0]), float(res[1])
        clo, chi, cw = star_width_from_children(children)
        gap = max(abs(plo - clo), abs(phi - chi))
        max_gap = max(max_gap, gap)
        rows.append((k, phi - plo, cw, gap))
    df = pd.DataFrame(rows, columns=["k", "pkg_width", "closed_width", "gap"])
    return df, max_gap


def chain_scan():
    """Fixed per-edge LR, vary chain length; show attenuation of width."""
    rows = []
    for L in (3.0, 8.0):
        for length in range(1, 9):
            # root <- c1 <- c2 <- ... <- leaf, each support with lr L, mid prior 0.5
            nodes = [Conclusion("Root", prior=0.5)]
            for d in range(1, length):
                nodes.append(Conclusion(f"C{d}", prior=0.5))
            leaf = Premise("Leaf", prior=0.7)
            chain = nodes + [leaf]
            for d in range(len(chain) - 1):
                chain[d].support(chain[d + 1], lr=L)
            arg = chain[0].assemble()
            res = posterior_range(arg.bn, arg.target, exact=True)
            rows.append((L, length, float(res[1] - res[0])))
    return pd.DataFrame(rows, columns=["edge_LR", "length", "width"])


def star_law_on_corpus(corpus="results/corpus.csv"):
    """Standalone predictive power of the flat-star law on the tree corpus.

    Two forms, both computed per-graph at extraction time (columns in corpus.csv):
      * star_exact_width -- the EXACT flat-star Jeffrey law applied to the graph's
        real per-edge LRs and source priors, flattened to root children (ignores
        depth composition). Available for graphs with <=14 LR-edges.
      * star_lumped_width -- the cheap logit-additive limit (children always-on):
        width = sigmoid(pro_mass) - sigmoid(-con_mass). Available for all graphs.

    R^2 alone = the closed-form prediction used directly (no fitted params). The gap
    from 1.0 quantifies depth attenuation the flat star ignores.
    """
    df = pd.read_csv(corpus)
    y = df["width"].values.astype(float)

    def r2_direct(pred, yy):
        ss_res = np.sum((yy - pred) ** 2)
        ss_tot = np.sum((yy - yy.mean()) ** 2)
        return 1 - ss_res / ss_tot

    def r2_pearson(pred, yy):
        r = np.corrcoef(pred, yy)[0, 1]
        return r * r

    predL = df["star_lumped_width"].values.astype(float)
    ex = df.dropna(subset=["star_exact_width"])
    predE = ex["star_exact_width"].values.astype(float)
    yE = ex["width"].values.astype(float)

    out = {
        "lumped_logit_additive_all_rows": {
            "n": int(len(df)),
            "R2_direct": round(float(r2_direct(predL, y)), 4),
            "r2_pearson": round(float(r2_pearson(predL, y)), 4)},
        "exact_flat_star_le14_edges": {
            "n": int(len(ex)), "coverage": round(len(ex) / len(df), 3),
            "R2_direct": round(float(r2_direct(predE, yE)), 4),
            "r2_pearson": round(float(r2_pearson(predE, yE)), 4)},
    }
    # also the exact-flat-star R^2 restricted to genuinely shallow graphs (depth<=2),
    # where the flat star should be nearly exact
    shallow = ex[ex["depth"] <= 2]
    if len(shallow) > 20:
        out["exact_flat_star_shallow_depth_le2"] = {
            "n": int(len(shallow)),
            "R2_direct": round(float(r2_direct(
                shallow["star_exact_width"].values.astype(float),
                shallow["width"].values.astype(float))), 4)}
    return out, predL, df["star_lumped_width"].values, y


def main():
    os.makedirs(PLOTS, exist_ok=True)
    out = {}

    sdf, max_gap = verify_stars()
    out["star_closed_form_vs_package"] = {
        "n_stars": int(len(sdf)), "max_abs_gap": float(max_gap),
        "exact_match": bool(max_gap < 1e-6),
        "mean_pkg_width": round(float(sdf["pkg_width"].mean()), 4)}

    cdf = chain_scan()
    out["chain_attenuation"] = {
        f"LR={lr}": [round(float(w), 4) for w in
                     cdf[cdf.edge_LR == lr].sort_values("length")["width"]]
        for lr in sorted(cdf.edge_LR.unique())}

    corpus = sys.argv[1] if len(sys.argv) > 1 else "results/corpus.csv"
    if os.path.exists(corpus):
        law, predL, _pe, y = star_law_on_corpus(corpus)
        out["star_law_as_tree_approx"] = law
        df = pd.read_csv(corpus)
        ex = df.dropna(subset=["star_exact_width"])
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].scatter(df["star_lumped_width"], df["width"], s=6, alpha=0.3, color="#3b7dd8")
        ax[0].plot([0, 1], [0, 1], "k--", lw=1)
        ax[0].set_xlabel("lumped-star predicted width"); ax[0].set_ylabel("true width")
        ax[0].set_title(f"lumped star (all rows)  R2={law['lumped_logit_additive_all_rows']['R2_direct']}")
        ax[1].scatter(ex["star_exact_width"], ex["width"], s=6, alpha=0.3, color="#d8543b")
        ax[1].plot([0, 1], [0, 1], "k--", lw=1)
        ax[1].set_xlabel("exact flat-star predicted width"); ax[1].set_ylabel("true width")
        ax[1].set_title(f"exact flat star (<=14 edges)  R2={law['exact_flat_star_le14_edges']['R2_direct']}")
        fig.tight_layout(); fig.savefig(f"{PLOTS}/star_surrogate.png", dpi=130); plt.close(fig)

    # chain plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for lr in sorted(cdf.edge_LR.unique()):
        sub = cdf[cdf.edge_LR == lr].sort_values("length")
        ax.plot(sub["length"], sub["width"], marker="o", label=f"edge LR={lr}")
    ax.set_xlabel("chain length (edges from leaf to root)")
    ax.set_ylabel("manipulability width")
    ax.set_title("chain: depth attenuation of a single evidence path")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{PLOTS}/chain_attenuation.png", dpi=130)
    plt.close(fig)

    # star width vs k
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sdf["k"] + np.random.default_rng(1).uniform(-0.15, 0.15, len(sdf)),
               sdf["pkg_width"], s=8, alpha=0.4, color="#3b7dd8")
    ax.set_xlabel("number of star children k"); ax.set_ylabel("package exact width")
    ax.set_title(f"star manipulability width (closed form matches to {max_gap:.1e})")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/star_width.png", dpi=130); plt.close(fig)

    with open("results/special_families.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
