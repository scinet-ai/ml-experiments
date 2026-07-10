"""Main exact manipulability sweep.

Generates a large seeded corpus of ASPIC argument graphs across a DECLARED grid of
parameter regimes, keeps the polytrees (exact=True), records manipulability width +
structural features to results/corpus.csv incrementally.

Reproducibility: graph i uses generation seed = i and draws its per-graph structural
knobs from random.Random(PARAM_SEED_BASE + i), so every row is independent and the
whole corpus regenerates identically. Regime assignment is a fixed weighted schedule.

Usage:
    python scripts/sweep.py [N] [out_csv]
        N        target number of ATTEMPTED graphs (default 7200)
        out_csv  output path (default results/corpus.csv)
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features  # noqa: E402

from probability_flow.aspic.generate import (  # noqa: E402
    StructuralParams, DifficultyTargets, generate,
)

PARAM_SEED_BASE = 8_000_000
warnings.simplefilter("ignore")


def _loguniform(rng, lo, hi):
    import math
    return round(math.exp(rng.uniform(math.log(lo), math.log(hi))), 4)


# --- declared regimes: name -> (weight, sampler) ---------------------------
# each sampler(rng) -> (StructuralParams, DifficultyTargets|None, meta dict)

def s_baseline(rng):
    ns = rng.choice([1, 2, 2, 3]); na = rng.choice([0, 1, 1, 2])
    if ns + na < 1:
        ns = 1
    p = StructuralParams.independent_only(
        n_support=ns, n_attack=na, min_depth=1, max_depth=rng.choice([2, 3, 3, 4]),
        max_fanin=rng.choice([1, 2, 2, 3]), fanin_prob=rng.choice([0.2, 0.3, 0.5]),
        internal_attack_prob=rng.choice([0.0, 0.1, 0.2]),
        root_prior=round(rng.uniform(0.3, 0.7), 4),
        support_lr_range=(rng.choice([2.0, 3.0]), rng.choice([8.0, 15.0])),
        attack_lr_range=(1 / rng.choice([8.0, 15.0]), 1 / rng.choice([2.0, 3.0])),
    )
    return p, None, {}


def s_attacks_heavy(rng):
    p = StructuralParams(
        n_support=rng.choice([1, 2]), n_attack=rng.choice([2, 3]),
        min_depth=1, max_depth=rng.choice([2, 3, 4]), max_fanin=rng.choice([1, 2, 3]),
        fanin_prob=0.4, internal_attack_prob=rng.choice([0.2, 0.35, 0.5]),
        undercut_prob=0.0, strict_prob=0.0, undermine_prob=rng.choice([0.0, 0.2]),
        axiom_prob=0.0, root_prior=round(rng.uniform(0.3, 0.7), 4),
    )
    return p, None, {}


def s_rich_defeaters(rng):
    p = StructuralParams(
        n_support=rng.choice([2, 3]), n_attack=rng.choice([1, 2]),
        min_depth=1, max_depth=rng.choice([2, 3, 4]), max_fanin=rng.choice([1, 2, 3]),
        fanin_prob=0.4, internal_attack_prob=rng.choice([0.1, 0.2]),
        undercut_prob=rng.choice([0.15, 0.3, 0.45]),
        strict_prob=rng.choice([0.1, 0.25, 0.4]),
        undermine_prob=rng.choice([0.15, 0.3]),
        axiom_prob=rng.choice([0.1, 0.2, 0.35]),
        attacker_depth_range=rng.choice([(1, 1), (1, 2)]),
        root_prior=round(rng.uniform(0.3, 0.7), 4),
    )
    return p, None, {}


def s_deep_chains(rng):
    p = StructuralParams.independent_only(
        n_support=1, n_attack=rng.choice([0, 1]),
        min_depth=rng.choice([3, 4]), max_depth=rng.choice([4, 5, 6]),
        max_fanin=rng.choice([0, 1]), fanin_prob=0.2,
        internal_attack_prob=rng.choice([0.0, 0.15]),
        root_prior=round(rng.uniform(0.35, 0.65), 4),
    )
    return p, None, {}


def s_wide(rng):
    # shallow + high fan-out at the root -> star-like
    p = StructuralParams.independent_only(
        n_support=rng.choice([2, 3, 4]), n_attack=rng.choice([1, 2, 3]),
        min_depth=1, max_depth=1, max_fanin=0,
        internal_attack_prob=0.0, root_prior=round(rng.uniform(0.3, 0.7), 4),
        support_lr_range=(rng.choice([2.0, 3.0]), rng.choice([6.0, 12.0, 20.0])),
        attack_lr_range=(1 / rng.choice([6.0, 12.0, 20.0]), 1 / rng.choice([2.0, 3.0])),
    )
    return p, None, {}


def s_large(rng):
    lo = rng.choice([22, 28, 34])
    p = StructuralParams(
        n_support=rng.choice([3, 4]), n_attack=2, min_depth=1,
        max_depth=rng.choice([4, 5]), max_fanin=3, fanin_prob=0.6,
        internal_attack_prob=rng.choice([0.1, 0.2]),
        undercut_prob=rng.choice([0.0, 0.2]), strict_prob=rng.choice([0.0, 0.15]),
        undermine_prob=rng.choice([0.0, 0.15]), axiom_prob=rng.choice([0.0, 0.15]),
        n_claims_range=(lo, 9999), root_prior=round(rng.uniform(0.35, 0.65), 4),
    )
    return p, None, {}


def s_lr_spread(rng):
    hi = rng.choice([5.0, 12.0, 25.0, 50.0])
    p = StructuralParams.independent_only(
        n_support=rng.choice([2, 3]), n_attack=rng.choice([1, 2]),
        min_depth=1, max_depth=rng.choice([2, 3]), max_fanin=rng.choice([1, 2]),
        fanin_prob=0.3, internal_attack_prob=rng.choice([0.0, 0.15]),
        support_lr_range=(1.2, hi), internal_lr_range=(1.2, max(2.0, hi * 0.8)),
        attack_lr_range=(1 / hi, 1 / 1.2), root_prior=round(rng.uniform(0.3, 0.7), 4),
    )
    return p, None, {}


def s_calibrated(rng):
    tp = rng.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    p = StructuralParams(
        n_support=rng.choice([2, 3]), n_attack=rng.choice([1, 2]),
        min_depth=1, max_depth=rng.choice([2, 3]), max_fanin=rng.choice([1, 2]),
        fanin_prob=0.3, internal_attack_prob=rng.choice([0.0, 0.15]),
        undercut_prob=rng.choice([0.0, 0.2]), strict_prob=rng.choice([0.0, 0.15]),
    )
    t = DifficultyTargets(target_posterior=tp, posterior_tol=0.04)
    return p, t, {"target_posterior": tp}


# NOTE on reconvergent graphs: the exact DP is exact only on polytrees
# (circuit_rank==0). The generator produces polytrees whenever share_prob==0 (all
# regimes below), so the corpus is 100% exact by construction and its own discard
# rate is 0. Reconvergence requires share_prob>0, which also forces generate()'s
# acceptance screen through the 2**n ExactSolver -- too costly to fold into the main
# sweep. The reconvergent DISCARD RATE is measured separately and cheaply by
# scripts/discard_probe.py (direct construction + circuit_rank, no solving).

REGIMES = [
    ("baseline", 28, s_baseline),
    ("attacks_heavy", 13, s_attacks_heavy),
    ("rich_defeaters", 17, s_rich_defeaters),
    ("deep_chains", 9, s_deep_chains),
    ("wide", 9, s_wide),
    ("large", 8, s_large),
    ("lr_spread", 8, s_lr_spread),
    ("calibrated", 8, s_calibrated),
]


def build_schedule(n):
    """Deterministic weighted round-robin regime assignment for indices 0..n-1."""
    names = [r[0] for r in REGIMES]
    weights = [r[1] for r in REGIMES]
    total = sum(weights)
    # expand into a repeating pattern proportional to weights, shuffled once
    pattern = []
    for name, w in zip(names, weights):
        pattern += [name] * w
    rng = random.Random(12345)
    rng.shuffle(pattern)
    return [pattern[i % total] for i in range(n)]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 7200
    out = sys.argv[2] if len(sys.argv) > 2 else "results/corpus.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    samplers = {name: fn for name, _w, fn in REGIMES}
    schedule = build_schedule(n)

    kept = 0
    discarded = 0        # is_exact False (reconvergent DAG)
    failed = 0           # generate() raised (targets unreachable)
    per_regime = {name: {"kept": 0, "disc": 0, "fail": 0} for name, _w, _f in REGIMES}

    writer = None
    fh = None
    t0 = time.time()
    fieldnames = None

    for i in range(n):
        regime = schedule[i]
        prng = random.Random(PARAM_SEED_BASE + i)
        params, targets, meta = samplers[regime](prng)
        try:
            arg = generate(seed=i, structural=params, targets=targets,
                           max_attempts=params.__dict__.get("_max_attempts", 200))
        except Exception:
            failed += 1
            per_regime[regime]["fail"] += 1
            continue
        row = extract_features(arg)
        if row is None:
            discarded += 1
            per_regime[regime]["disc"] += 1
            continue
        row["seed"] = i
        row["regime"] = regime
        row["target_posterior"] = meta.get("target_posterior", "")
        if writer is None:
            fieldnames = (["seed", "regime", "target_posterior"]
                          + [k for k in row.keys()
                             if k not in ("seed", "regime", "target_posterior")])
            fh = open(out, "w", newline="")
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
        writer.writerow(row)
        kept += 1
        per_regime[regime]["kept"] += 1

        if (i + 1) % 250 == 0:
            fh.flush()
            el = time.time() - t0
            print(f"[{i+1}/{n}] kept={kept} disc={discarded} fail={failed} "
                  f"{el:.1f}s ({1000*el/(i+1):.1f} ms/attempt)", flush=True)

    if fh:
        fh.close()
    el = time.time() - t0
    manifest = {
        "attempted": n, "kept": kept, "discarded_nonpolytree": discarded,
        "failed_generation": failed,
        "discard_rate_of_generated": round(discarded / max(kept + discarded, 1), 4),
        "wall_seconds": round(el, 1), "ms_per_attempt": round(1000 * el / n, 2),
        "per_regime": per_regime, "param_seed_base": PARAM_SEED_BASE,
        "regime_weights": {name: w for name, w, _f in REGIMES},
    }
    mpath = os.path.join(os.path.dirname(out) or ".", "sweep_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print("\n=== SWEEP DONE ===")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
