"""Predictive analysis of manipulability width from structural features.

- 80/20 train/test split (seeded).
- OLS on standardized features; HistGradientBoosting regressor.
- Out-of-sample R^2 for both; permutation importance for the GBM.
- PI hypothesis test: width ~ total |log LR| mass + pro/con asymmetry, depth via
  attenuation. Fit nested OLS models and compare.
- Phase maps: width over (total_mass, signed asymmetry) and (n_nodes, attack_frac);
  knife-edge (>0.6) and near-unmovable (<0.1) signatures + prevalence.
- Partial-dependence for the top GBM features.

Usage: python scripts/analysis.py [corpus.csv] [--quick]
Writes results/plots/*.png and prints a JSON summary block.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PLOTS = "results/plots"

# structural predictors (exclude outcomes/answer-leakage: width,min_post,max_post,
# true_post, outer_*; exclude circuit_rank which is identically 0 on the kept set)
FEATURES = [
    "n_nodes", "n_edges", "n_reg_edges", "n_bn_nodes", "upstream_size",
    "depth", "claim_depth", "mean_fanin", "max_fanin",
    "n_support", "n_rebut", "n_undermine", "n_strict", "n_undercut",
    "n_axioms", "n_premises", "n_conclusions",
    "attack_fraction", "rebut_fraction",
    "total_mass", "mean_abs_log", "max_abs_log", "pro_mass", "con_mass",
    "signed_mass", "abs_signed_mass", "asymmetry",
    "signed_mass_typesign", "asymmetry_typesign",
    "depth_weighted_mass", "depth_weighted_signed", "root_prior", "n_lr_edges",
]
TARGET = "width"


def ols_r2(X, y, Xte, yte):
    sc = StandardScaler().fit(X)
    m = LinearRegression().fit(sc.transform(X), y)
    return r2_score(yte, m.predict(sc.transform(Xte))), m, sc


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "results/corpus.csv"
    quick = "--quick" in sys.argv
    os.makedirs(PLOTS, exist_ok=True)
    df = pd.read_csv(csv)
    feats = [f for f in FEATURES if f in df.columns and df[f].nunique() > 1]
    df = df.dropna(subset=feats + [TARGET])
    X = df[feats].values.astype(float)
    y = df[TARGET].values.astype(float)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    summary = {"n_rows": int(len(df)), "n_features": len(feats),
               "width_mean": float(y.mean()), "width_std": float(y.std())}

    # --- OLS full ---
    r2_ols, ols_model, ols_scaler = ols_r2(Xtr, ytr, Xte, yte)
    coefs = sorted(zip(feats, ols_model.coef_), key=lambda t: -abs(t[1]))
    summary["OLS_full_R2_oos"] = round(float(r2_ols), 4)
    summary["OLS_top_standardized_coefs"] = [(f, round(float(c), 4)) for f, c in coefs[:10]]

    # --- GBM ---
    gbm = HistGradientBoostingRegressor(
        max_iter=400 if not quick else 120, learning_rate=0.06,
        max_depth=None, max_leaf_nodes=31, l2_regularization=1.0,
        random_state=0, early_stopping=True)
    gbm.fit(Xtr, ytr)
    r2_gbm = r2_score(yte, gbm.predict(Xte))
    summary["GBM_R2_oos"] = round(float(r2_gbm), 4)

    perm = permutation_importance(gbm, Xte, yte, n_repeats=10 if not quick else 4,
                                  random_state=0, scoring="r2")
    imp = sorted(zip(feats, perm.importances_mean, perm.importances_std),
                 key=lambda t: -t[1])
    summary["GBM_permutation_importance_top"] = [
        (f, round(float(m), 4), round(float(s), 4)) for f, m, s in imp[:12]]

    # --- PI hypothesis: nested OLS models ---
    def r2_subset(cols):
        cc = [c for c in cols if c in feats]
        idx = [feats.index(c) for c in cc]
        return round(float(ols_r2(Xtr[:, idx], ytr, Xte[:, idx], yte)[0]), 4), cc

    hyp = {}
    hyp["total_mass_only"] = r2_subset(["total_mass"])
    hyp["asymmetry_only"] = r2_subset(["asymmetry"])
    hyp["mass_plus_asym"] = r2_subset(["total_mass", "asymmetry"])
    hyp["mass_asym_pro_con"] = r2_subset(["total_mass", "asymmetry", "pro_mass", "con_mass"])
    hyp["depth_weighted_mass_only"] = r2_subset(["depth_weighted_mass"])
    hyp["mass_asym_depth"] = r2_subset(["total_mass", "asymmetry", "depth"])
    hyp["mass_asym_depthw"] = r2_subset(["depth_weighted_mass", "asymmetry"])
    hyp["depth_only"] = r2_subset(["depth"])
    hyp["pro_and_con"] = r2_subset(["pro_mass", "con_mass"])
    hyp["full"] = (summary["OLS_full_R2_oos"], "all")
    # also a GBM on the 3-feature hypothesis set
    core = [c for c in ["total_mass", "asymmetry", "depth_weighted_mass"] if c in feats]
    ci = [feats.index(c) for c in core]
    g2 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                       random_state=0, early_stopping=True).fit(Xtr[:, ci], ytr)
    hyp["GBM_core3_total_asym_depthw"] = (round(float(r2_score(yte, g2.predict(Xte[:, ci]))), 4), core)
    summary["hypothesis_nested_R2"] = hyp

    # --- correlations of key features with width ---
    keycorr = {}
    for c in ["total_mass", "asymmetry", "pro_mass", "con_mass", "depth",
              "depth_weighted_mass", "n_nodes", "attack_fraction", "max_abs_log",
              "abs_signed_mass", "root_prior"]:
        if c in df.columns:
            keycorr[c] = round(float(np.corrcoef(df[c], df[TARGET])[0, 1]), 4)
    summary["pearson_corr_with_width"] = keycorr

    # --- regime knife-edge / near-unmovable prevalence ---
    knife = df[df[TARGET] > 0.6]
    unmov = df[df[TARGET] < 0.1]
    def sig(sub):
        if len(sub) == 0:
            return {}
        return {c: round(float(sub[c].mean()), 3) for c in
                ["total_mass", "asymmetry", "pro_mass", "con_mass", "depth",
                 "depth_weighted_mass", "n_nodes", "attack_fraction", "max_abs_log",
                 "mean_fanin", "root_prior", "true_post"] if c in sub.columns}
    summary["knife_edge_width_gt_0.6"] = {
        "n": int(len(knife)), "prevalence": round(len(knife) / len(df), 4),
        "signature_mean": sig(knife)}
    summary["near_unmovable_width_lt_0.1"] = {
        "n": int(len(unmov)), "prevalence": round(len(unmov) / len(df), 4),
        "signature_mean": sig(unmov)}
    summary["overall_signature_mean"] = sig(df)

    # ---------------- PLOTS ----------------
    # 1. phase map: width over (total_mass, asymmetry) with signed asymmetry
    df["signed_asym"] = np.sign(df["signed_mass"]) * df["asymmetry"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    hb = ax[0].hexbin(df["total_mass"], df["signed_asym"], C=df[TARGET],
                      gridsize=32, cmap="viridis", reduce_C_function=np.mean)
    ax[0].set_xlabel("total |log LR| mass"); ax[0].set_ylabel("signed pro/con asymmetry")
    ax[0].set_title("mean manipulability width")
    fig.colorbar(hb, ax=ax[0], label="width")
    hb2 = ax[1].hexbin(df["n_nodes"], df["attack_fraction"], C=df[TARGET],
                       gridsize=28, cmap="viridis", reduce_C_function=np.mean)
    ax[1].set_xlabel("n_nodes"); ax[1].set_ylabel("attack fraction")
    ax[1].set_title("mean manipulability width")
    fig.colorbar(hb2, ax=ax[1], label="width")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/phase_maps.png", dpi=130); plt.close(fig)

    # 2. width vs total_mass colored by asymmetry
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(df["total_mass"], df[TARGET], c=df["asymmetry"], cmap="coolwarm",
                    s=6, alpha=0.4)
    ax.set_xlabel("total |log LR| mass"); ax.set_ylabel("manipulability width")
    ax.set_title("width vs evidence mass (color = asymmetry)")
    fig.colorbar(sc, ax=ax, label="asymmetry"); fig.tight_layout()
    fig.savefig(f"{PLOTS}/width_vs_mass.png", dpi=130); plt.close(fig)

    # 3. GBM permutation importance bar
    fig, ax = plt.subplots(figsize=(8, 6))
    top = imp[:14][::-1]
    ax.barh([t[0] for t in top], [t[1] for t in top],
            xerr=[t[2] for t in top], color="#3b7dd8")
    ax.set_xlabel("permutation importance (drop in OOS R^2)")
    ax.set_title(f"GBM feature importance (OOS R^2={r2_gbm:.3f})")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/gbm_importance.png", dpi=130); plt.close(fig)

    # 4. partial dependence for top-4 GBM features + one 2D pair
    topf = [feats.index(imp[k][0]) for k in range(min(4, len(imp)))]
    try:
        fig, axs = plt.subplots(2, 2, figsize=(11, 8))
        for a, fidx in zip(axs.ravel(), topf):
            pd_res = partial_dependence(gbm, Xte, [fidx], kind="average")
            gx = pd_res["grid_values"][0]; gy = pd_res["average"][0]
            a.plot(gx, gy, color="#d8543b")
            a.set_xlabel(feats[fidx]); a.set_ylabel("PD width")
        fig.suptitle("GBM partial dependence — top features")
        fig.tight_layout(); fig.savefig(f"{PLOTS}/partial_dependence.png", dpi=130); plt.close(fig)
    except Exception as e:
        summary["pd_error"] = str(e)

    # 5. 2D partial dependence: total_mass x asymmetry (the PI's pair)
    try:
        if "total_mass" in feats and "asymmetry" in feats:
            ti, ai = feats.index("total_mass"), feats.index("asymmetry")
            pdr = partial_dependence(gbm, Xte, [(ti, ai)], kind="average", grid_resolution=20)
            gv = pdr["grid_values"]; Z = pdr["average"][0]
            fig, ax = plt.subplots(figsize=(7, 5.5))
            cf = ax.contourf(gv[0], gv[1], Z.T, levels=18, cmap="viridis")
            ax.set_xlabel("total |log LR| mass"); ax.set_ylabel("asymmetry")
            ax.set_title("GBM 2D partial dependence -> width")
            fig.colorbar(cf, ax=ax, label="PD width"); fig.tight_layout()
            fig.savefig(f"{PLOTS}/pd_mass_asymmetry.png", dpi=130); plt.close(fig)
    except Exception as e:
        summary["pd2d_error"] = str(e)

    with open("results/analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
