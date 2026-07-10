"""Cross-validate posterior_range(exact=True) against an INDEPENDENT brute-force
reimplementation of the documented reveal model.

The package does not expose reveal-set enumeration, so we reimplement the reveal
semantics from the manipulability.py docstring and enumerate the full achievable
set of root posteriors over every connected-from-root reveal configuration:

  * root always included, its own prior fixed at 0.5;
  * each edge (input) may be HIDDEN (LR 1, child contributes p0) or REVEALED with
    the true LR *or* the direction-only type-default (2.0 if lr>1 else 0.5);
  * a leaf premise revealed contributes its true prior; an interior node included
    picks its own p0 in {0.5, true prior};
  * connectivity is enforced structurally (an edge can be revealed only inside its
    parent's own reveal choice).

We take min/max over the FULL achievable set (no greedy per-child shortcut), so a
match with the DP validates both the DP's endpoint-summarisation and its
independent-child optimisation. Restricted to independent-evidence polytrees where
these semantics are exactly reimplementable. We also record the weaker package-only
checks (a) exact-in-outer-bound, (b) empty-reveal 0.5 in range, (c) full-reveal
posterior in range -- honestly, for every graph.

Run:  .venv/bin/python scripts/crossval.py
"""
from __future__ import annotations

import math
import sys
from itertools import product

from probability_flow.aspic.generate import StructuralParams, generate
from probability_flow.core import IndependentEvidenceCPD, LoopySolver
from probability_flow.metrics import posterior_range

_CLAMP = 1e-3
_RND = 12


def _clamp(p):
    return min(1 - _CLAMP, max(_CLAMP, p))


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _jeffrey(p0, configs):
    """Independent reimplementation of the exact Jeffrey marginal for one
    IndependentEvidence node given a list of (effective_LR, on_prob) inputs."""
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


def brute_range(bn, target):
    """Full achievable [min,max] root posterior by enumerating every reveal config.
    Returns (lo, hi). Independent-evidence polytrees only."""
    memo: dict[int, set] = {}

    def included_set(node):
        if id(node) in memo:
            return memo[id(node)]
        cc = bn.compiled_cpd(node)
        cpd, inputs = cc.cpd, cc.inputs
        assert isinstance(cpd, IndependentEvidenceCPD), "independent-evidence only"
        if not inputs:                                   # leaf
            if node.prior >= 1.0 - 1e-12:
                s = {1.0}
            elif node is target:
                s = {_clamp(0.5)}
            else:
                s = {_clamp(node.prior)}
            memo[id(node)] = s
            return s
        # interior node
        if node is target:
            p0s = [0.5]
        elif node.prior >= 1.0 - 1e-12:
            p0s = [1.0]
        else:
            p0s = [0.5, _clamp(node.prior)]
        per_input = []
        for inp, lr in zip(inputs, cpd.lrs):
            child = included_set(inp)
            assumed = 2.0 if lr > 1.0 else 0.5
            opts = [(1.0, 0.5)]                          # hidden -> evaluates to p0
            for leff in (lr, assumed):
                for pp in child:
                    opts.append((leff, pp))
            per_input.append(opts)
        vals = set()
        for p0 in p0s:
            for combo in product(*per_input):
                vals.add(round(_clamp(_jeffrey(p0, list(combo))), _RND))
        memo[id(node)] = vals
        return vals

    s = included_set(target)
    return min(s), max(s)


def main():
    n_graphs = 20
    rows = []
    brute_pass = 0
    contain_pass = 0
    empty_pass = 0
    full_pass = 0
    max_brute_gap = 0.0

    i = 0
    made = 0
    while made < n_graphs:
        pr_seed = i
        i += 1
        # small independent-evidence polytrees (<=10 nodes)
        import random
        pr = random.Random(500 + pr_seed)
        params = StructuralParams.independent_only(
            n_support=pr.choice([1, 2]),
            n_attack=pr.choice([0, 1]),
            min_depth=1, max_depth=pr.choice([1, 2, 2, 3]),
            max_fanin=pr.choice([0, 1, 1, 2]), fanin_prob=0.4,
            internal_attack_prob=pr.choice([0.0, 0.2]),
        )
        if params.n_support + params.n_attack < 1:
            params.n_support = 1
        try:
            arg = generate(seed=pr_seed, structural=params, max_attempts=200)
        except Exception:
            continue
        bn = arg.bn
        n_nodes = len(bn.nodes)
        if n_nodes > 10:
            continue
        made += 1

        res = posterior_range(bn, arg.target, exact=True)
        assert res.is_exact
        lo, hi = float(res[0]), float(res[1])
        olo, ohi = posterior_range(bn, arg.target, exact=False)

        blo, bhi = brute_range(bn, arg.target)
        gap = max(abs(blo - lo), abs(bhi - hi))
        max_brute_gap = max(max_brute_gap, gap)
        ok_brute = gap < 1e-6

        # (a) exact contained in outer bound
        ok_contain = (olo <= lo + 1e-9) and (hi <= ohi + 1e-9)
        # (b) empty reveal (target prior 0.5, nothing else) -> 0.5 in range
        ok_empty = (lo - 1e-6) <= 0.5 <= (hi + 1e-6)
        # (c) full reveal at true LRs with reveal-model root prior 0.5, in range
        saved = arg.target.prior
        arg.target.prior = 0.5
        fresh = arg.target.compile()
        p_full = LoopySolver(fresh).prob(arg.target, 1)
        arg.target.prior = saved
        ok_full = (lo - 1e-6) <= p_full <= (hi + 1e-6)

        brute_pass += ok_brute
        contain_pass += ok_contain
        empty_pass += ok_empty
        full_pass += ok_full
        rows.append((made, n_nodes, lo, hi, blo, bhi, gap,
                     ok_brute, ok_contain, ok_empty, ok_full))

    print(f"{'#':>2} {'nodes':>5} {'DP_lo':>8} {'DP_hi':>8} {'BF_lo':>8} {'BF_hi':>8} "
          f"{'gap':>10}  brute contain empty full")
    for (idx, nn, lo, hi, blo, bhi, gap, ob, oc, oe, of) in rows:
        print(f"{idx:>2} {nn:>5} {lo:>8.4f} {hi:>8.4f} {blo:>8.4f} {bhi:>8.4f} "
              f"{gap:>10.2e}   {int(ob)}     {int(oc)}     {int(oe)}   {int(of)}")

    n = len(rows)
    print(f"\nGraphs validated: {n}")
    print(f"(brute-force exact match, gap<1e-6): {brute_pass}/{n}   max gap {max_brute_gap:.2e}")
    print(f"(a) exact inside outer bound       : {contain_pass}/{n}")
    print(f"(b) empty-reveal 0.5 inside range  : {empty_pass}/{n}")
    print(f"(c) full-reveal posterior inside   : {full_pass}/{n}")
    all_ok = (brute_pass == n and contain_pass == n and empty_pass == n and full_pass == n)
    print("\nVALIDATION:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
