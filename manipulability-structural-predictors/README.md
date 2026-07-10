# Structural predictors of argument-graph manipulability under partial disclosure

Which structural features of an argument graph predict its **manipulability** — the width
of the exact achievable range `[min, max]` of the root-claim posterior over all legal
(root-connected) reveal subsets, judged by an ideal Bayesian?

SciNet problem: `193e0217` · investigation: `75238f3c` · agent: `tracke-debate-lead`.

## Setup (pinned)

- **Engine:** [`probability-flow==0.4.0`](https://pypi.org/project/probability-flow/)
  (PyPI, MIT). Graphs from its ASPIC generator, compiled to Bayesian networks;
  manipulability via `posterior_range(bn, root, exact=True)` — the linear-time exact
  polytree DP. Reveal semantics are the package's documented ones: a hidden premise
  contributes prior 0.5 / LR 1; a revealed defeasible edge may be presented with its
  **true LR or the direction-only default** (LR>1 → 2.0, LR<1 → 0.5), and the DP takes
  the achievable extremes over legal root-connected reveal sets.
- **Validation:** the DP was cross-checked against an **independent brute-force
  enumeration** written for this study (20 graphs, all reveal sets): max gap **5.6e-13**
  (`results/crossval_report.txt`). Sanity invariants (exact ⊆ outer bound; empty- and
  full-reveal posteriors inside the range) hold 20/20.
- **Corpus:** 7,200 attempted seeded graphs across 6 declared generator regimes
  (baseline / attacks-heavy / deep-chains / large / lr-spread / calibrated-target;
  weights in `results/sweep_manifest.json`), `share_prob=0` → all polytrees by
  construction (reconvergence probe in `results/discard_probe.json`). **7,199 kept**;
  1 graph skipped by a declared 5s per-graph timeout guard (pathological high-fan-in
  outer-bound enumeration; seed recorded in the manifest). Graph sizes 2–~60 nodes,
  root priors 0.3–0.7. Corpus: `results/corpus.csv.gz` (regenerates exactly from seeds).
- **Features:** 33 structural features computed *without* looking at the answer
  (`scripts/features.py`): sizes, depth, fan-in, edge-type counts, attack fractions,
  |log LR| masses (total / pro / con / signed / asymmetry), **depth-weighted mass**
  (each edge's |log LR| attenuated by the product of intervening edge strengths on the
  path to the root), root prior.
- **Models:** OLS (standardized) and HistGradientBoosting, 80/20 split, seeded;
  permutation importances; nested-model hypothesis tests (`scripts/analysis.py`).

## Results

Corpus width: mean **0.633**, sd 0.186.

| model | OOS R² |
|---|---|
| OLS, all 33 features | **0.867** |
| GBM, all 33 features | **0.924** |
| GBM, 3 features (depth-weighted mass, total mass, asymmetry) | **0.869** |

**Nested hypothesis test** (OLS OOS R²): total |log LR| mass alone **0.278**; pro/con
asymmetry alone 0.338; mass+asymmetry 0.500; **depth-weighted mass alone 0.574**;
depth-weighted mass + asymmetry **0.685**. Adding raw depth to (mass, asymmetry) adds
~nothing (0.501) — depth matters **through attenuation**, not as a count.

**Dominant predictor:** depth-weighted evidence mass. GBM permutation importance 1.67 vs
0.066 for the runner-up (25×). Pearson r with width: depth_weighted_mass **+0.757**,
con_mass +0.561, asymmetry −0.561, total_mass +0.546.

**Star closed form (exact):** for a root with k independent evidence edges, the
achievable extremes are (all-and-only pro edges revealed) vs (all-and-only con edges
revealed), each side free to use true-LR or direction-default presentation per edge.
Verified against the package on 400 random stars: max abs gap **4.4e-16**
(`results/special_families.json`). As a *surrogate* for general trees (flattening each
tree to a star of its root-visible masses) it reaches only R² ≈ 0.60 — deep structure
carries real signal beyond lumped mass.

**Chain attenuation:** width along a single support chain converges geometrically with
depth (e.g. LR=3: 0.175 → 0.167 plateau by depth ~5) — evidence far from the root is
capped in influence, which is why depth-weighting works.

**Phase map** (`results/plots/phase_maps.png`), *under this generator's distribution*:
knife-edge graphs (width > 0.6) are **60.2%** of the corpus — signature: high
depth-weighted mass, high fan-in, roughly balanced pro/con mass. Near-unmovable graphs
(width < 0.1) are essentially absent (**0.03%**, only trivial 2-node one-sided graphs):
random argument graphs of any size are, by default, highly manipulable to an ideal judge
under selective disclosure; low manipulability requires extreme one-sidedness
(asymmetry → 1) or near-zero evidence mass.

## Limitations

- Polytrees only (the exact DP's domain); reconvergent DAGs excluded by construction.
- One generator family (6 regimes); "prevalence" numbers are properties of this
  declared distribution, not of argument graphs in the wild.
- Manipulability is the package's reveal semantics (including direction-default edge
  presentation); a true-LR-only variant would give narrower ranges.

## Reproduce

```bash
./reproduce.sh   # fresh uv venv -> smoke test -> 200-graph mini-corpus -> regression, <60s
```

Full corpus: `python scripts/sweep.py 7200 results/corpus.csv` then
`python scripts/sweep_resume.py 7200 results/corpus.csv 5` (timeout-guarded completion)
then `python scripts/analysis.py results/corpus.csv`. Validation:
`python scripts/crossval.py`. Star law/chains: `python scripts/special_families.py`.
