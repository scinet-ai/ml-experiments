"""Reconvergent discard-rate probe.

The exact manipulability DP is exact only on polytrees (circuit_rank==0); it falls
back to the loose outer bound (is_exact=False) on reconvergent DAGs, which we
DISCARD. The main corpus is 100% polytree by construction (share_prob==0), so its
discard rate is 0. This probe measures the discard rate that WOULD arise if the
generator's node-sharing were enabled: it builds graphs directly and checks
circuit_rank -- no ExactSolver, so it is cheap even where generate() would not be.

Run: python scripts/discard_probe.py > results/discard_probe.json
"""
from __future__ import annotations

import json
import random
import sys
import warnings

warnings.simplefilter("ignore")

from probability_flow.aspic.generate import StructuralParams, _build_one
from probability_flow.metrics import circuit_rank

import numpy as np


def probe(share_prob, n=400, seed0=0):
    disc = 0
    kept = 0
    ranks = []
    sizes = []
    for i in range(n):
        rng = random.Random(90000 + seed0 + i)
        params = StructuralParams(
            n_support=rng.choice([2, 3]), n_attack=rng.choice([1, 2]),
            min_depth=1, max_depth=rng.choice([2, 3]), max_fanin=3,
            fanin_prob=rng.choice([0.5, 0.7]), share_prob=share_prob,
            internal_attack_prob=0.15,
        )
        g = _build_one(rng, params)
        arg = g.build()
        cr = circuit_rank(arg.bn, arg.target)
        ranks.append(cr)
        sizes.append(len(arg.bn.nodes))
        if cr > 0:
            disc += 1
        else:
            kept += 1
    return {
        "share_prob": share_prob, "n": n,
        "non_polytree_discarded": disc, "polytree_kept": kept,
        "discard_rate": round(disc / n, 4),
        "mean_circuit_rank": round(float(np.mean(ranks)), 3),
        "max_circuit_rank": int(max(ranks)),
        "mean_bn_nodes": round(float(np.mean(sizes)), 1),
    }


def main():
    out = {"note": ("discard rate = fraction of generated graphs that are "
                    "reconvergent (circuit_rank>0), for which exact=True falls back "
                    "to the outer bound and which are discarded from the exact "
                    "corpus. Main corpus uses share_prob=0 -> discard_rate 0."),
           "corpus_share_prob_0_discard_rate": 0.0,
           "reconvergent_regimes": []}
    for sp in (0.0, 0.2, 0.4, 0.6, 0.8):
        out["reconvergent_regimes"].append(probe(sp))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
