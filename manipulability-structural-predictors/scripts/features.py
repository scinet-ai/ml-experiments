"""Feature extraction for one compiled Argument.

Works off the raw claim graph (`_edges` / `_strict` / `_undercuts` on each claim,
reached via `_reachable`) so we call the exact posterior_range DP exactly once per
graph. Orientation-toward-root signs are computed by a BFS from the root over the
reverse (upstream) links, multiplying an edge's base polarity along the unique
downstream path (well-defined on a polytree). Undercut linkages carry no LR but do
flip polarity (undercutting a support removes support; undercutting an attack
removes the attack), so an undercutter's own sub-argument mass is oriented through
the gated edge's sign.
"""
from __future__ import annotations

import math
import warnings

from probability_flow.aspic.argument import Axiom, Conclusion, Premise, _Claim
from probability_flow.aspic.compile import _reachable
from probability_flow.core import LoopySolver
from probability_flow.metrics import (
    circuit_rank, max_depth, posterior_range, upstream_size,
)

GAMMA = 0.5  # depth discount for depth-weighted mass


_CLAMP = 1e-3
_STAR_EXACT_MAX_CHILDREN = 14   # 2**14 cap for the exact flat-star Jeffrey sum


def _base_sign(kind: str) -> int:
    # support / strict push the conclusion up; rebut / undermine push it down
    return +1 if kind in ("support", "strict") else -1


def _clamp(p):
    return min(1 - _CLAMP, max(_CLAMP, p))


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _jeffrey_star(configs):
    """Exact Jeffrey marginal at root prior 0.5 for independent children
    (leff, on_prob). Used for the flat-star law prediction on a tree."""
    if not configs:
        return 0.5
    k = len(configs)
    total = 0.0
    for combo in range(1 << k):
        lo = 0.0
        w = 1.0
        for i, (leff, pp) in enumerate(configs):
            if (combo >> i) & 1:
                lo += math.log(max(leff, 1e-9))
                w *= pp
            else:
                w *= 1.0 - pp
        total += _sigmoid(lo) * w
    return total


def extract_features(arg, gamma: float = GAMMA) -> dict | None:
    """Return a feature+outcome row for `arg`, or None if the exact DP falls back
    (reconvergent DAG -> is_exact False; caller records the discard)."""
    target = arg.target
    bn = arg.bn

    # --- exact manipulability range (single DP call) -----------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ManipulabilityWarning handled via is_exact
        res = posterior_range(bn, target, exact=True)
    if not res.is_exact:
        return None
    lo, hi = float(res[0]), float(res[1])
    width = hi - lo

    olo, ohi = posterior_range(bn, target, exact=False)  # outer bound (for slack)

    # --- claim graph: nodes, edges, links ----------------------------------
    claims = [c for c in _reachable(target) if isinstance(c, _Claim)]

    # regular edges: (src, tgt, kind, lr)   ;  undercut links: (by, tgt, sign, lr=None)
    reg_edges = []          # (src, tgt, kind, lr)
    uc_links = []           # (by, tgt, sign)
    for c in claims:
        for src, lr, kind in c._edges:
            reg_edges.append((src, c, kind, lr))
        for src in c._strict:
            reg_edges.append((src, c, "strict", None))
        # undercut: linkage sign is the NEGATION of the gated edge's sign
        gated_sign = {}
        for src, lr, kind in c._edges:
            gated_sign[id(src)] = _base_sign(kind)
        for src in c._strict:
            gated_sign[id(src)] = +1
        for source, by in c._undercuts:
            uc_links.append((by, c, -gated_sign.get(id(source), +1)))

    # reverse adjacency for BFS from root: from a node -> its upstream neighbours,
    # each tagged with the base sign of the link.
    up_adj: dict[int, list[tuple[_Claim, int]]] = {}
    for src, tgt, kind, lr in reg_edges:
        up_adj.setdefault(id(tgt), []).append((src, _base_sign(kind)))
    for by, tgt, sign in uc_links:
        up_adj.setdefault(id(tgt), []).append((by, sign))

    # BFS from root -> depth and orientation sign per claim (polytree: unique).
    depth = {id(target): 0}
    orient = {id(target): +1}
    frontier = [target]
    while frontier:
        nxt = []
        for node in frontier:
            for up, sign in up_adj.get(id(node), []):
                if id(up) not in depth:
                    depth[id(up)] = depth[id(node)] + 1
                    orient[id(up)] = orient[id(node)] * sign
                    nxt.append(up)
        frontier = nxt

    # --- counts by edge kind ----------------------------------------------
    n_support = sum(1 for _s, _t, k, _l in reg_edges if k == "support")
    n_rebut = sum(1 for _s, _t, k, _l in reg_edges if k == "rebut")
    n_undermine = sum(1 for _s, _t, k, _l in reg_edges if k == "undermine")
    n_strict = sum(1 for _s, _t, k, _l in reg_edges if k == "strict")
    n_undercut = len(uc_links)
    n_reg = len(reg_edges)
    n_edges = n_reg + n_undercut

    # --- node-type counts --------------------------------------------------
    n_axioms = sum(1 for c in claims if isinstance(c, Axiom))
    n_premises = sum(1 for c in claims if isinstance(c, Premise) and not isinstance(c, Axiom))
    n_conclusions = sum(1 for c in claims if isinstance(c, Conclusion))
    n_nodes = len(claims)

    # --- fan-in per node (regular incoming edges) --------------------------
    fanin: dict[int, int] = {}
    for _s, tgt, _k, _l in reg_edges:
        fanin[id(tgt)] = fanin.get(id(tgt), 0) + 1
    fanins = list(fanin.values()) if fanin else [0]
    mean_fanin = sum(fanins) / len(fanins)
    max_fanin = max(fanins)

    # --- LR mass features --------------------------------------------------
    abs_logs = []
    pro_mass = 0.0
    con_mass = 0.0
    typesign_mass = 0.0
    depth_w_mass = 0.0
    depth_w_signed = 0.0
    pro_children = []   # (leff_up, pp) toward-root-increasing, for the flat-star law
    con_children = []   # (leff_down, pp) toward-root-decreasing
    for src, tgt, kind, lr in reg_edges:
        if lr is None:          # strict / pending carry no LR
            continue
        al = abs(math.log(lr))
        abs_logs.append(al)
        s_orient = orient.get(id(src), _base_sign(kind))  # fallback: own polarity
        pp = _clamp(float(src.prior))
        if s_orient >= 0:
            pro_mass += al
            pro_children.append((max(math.exp(al), 2.0), pp))   # judge picks larger LR
        else:
            con_mass += al
            con_children.append((min(math.exp(-al), 0.5), pp))  # judge picks smaller LR
        typesign_mass += _base_sign(kind) * al
        d = depth.get(id(tgt), 0)          # edge distance from root ~ downstream depth
        w = gamma ** d
        depth_w_mass += w * al
        depth_w_signed += w * (s_orient if s_orient else 1) * al

    total_mass = sum(abs_logs)
    mean_abs_log = (total_mass / len(abs_logs)) if abs_logs else 0.0
    max_abs_log = max(abs_logs) if abs_logs else 0.0
    signed_mass = pro_mass - con_mass
    denom = pro_mass + con_mass
    asymmetry = (abs(pro_mass - con_mass) / denom) if denom > 0 else 0.0
    asymmetry_typesign = (abs(typesign_mass) / denom) if denom > 0 else 0.0

    attack_edges = n_rebut + n_undermine + n_undercut
    attack_fraction = attack_edges / n_edges if n_edges else 0.0
    rebut_fraction = n_rebut / n_reg if n_reg else 0.0

    # --- flat-star-law prediction (treat every LR-edge as a direct root child) ---
    # lumped logit-additive form (children always-on, pp->1): the cheap closed form.
    star_lumped_width = _clamp(_sigmoid(pro_mass)) - _clamp(_sigmoid(-con_mass))
    # exact flat-star Jeffrey (true per-edge on-probabilities) when child count small
    n_children = len(pro_children) + len(con_children)
    if n_children <= _STAR_EXACT_MAX_CHILDREN:
        star_exact_max = _clamp(_jeffrey_star(pro_children))
        star_exact_min = _clamp(_jeffrey_star(con_children))
        star_exact_width = star_exact_max - star_exact_min
    else:
        star_exact_width = ""    # NaN -> analysis reports coverage separately

    # --- posteriors --------------------------------------------------------
    true_post = float(arg.posterior(target))       # calibrated prior, all revealed
    root_prior = float(target.prior)

    return {
        # outcomes
        "width": width, "min_post": lo, "max_post": hi,
        "true_post": true_post,
        "outer_lo": float(olo), "outer_hi": float(ohi),
        "outer_slack": float((ohi - olo) - width),
        # structure
        "n_nodes": n_nodes, "n_edges": n_edges, "n_reg_edges": n_reg,
        "n_bn_nodes": len(bn.nodes), "upstream_size": upstream_size(bn, target),
        "depth": max_depth(bn, target),
        "claim_depth": max(depth.values()) if depth else 0,
        "mean_fanin": mean_fanin, "max_fanin": max_fanin,
        "circuit_rank": circuit_rank(bn, target),
        # counts
        "n_support": n_support, "n_rebut": n_rebut, "n_undermine": n_undermine,
        "n_strict": n_strict, "n_undercut": n_undercut,
        "n_axioms": n_axioms, "n_premises": n_premises, "n_conclusions": n_conclusions,
        "attack_fraction": attack_fraction, "rebut_fraction": rebut_fraction,
        # LR mass
        "total_mass": total_mass, "mean_abs_log": mean_abs_log, "max_abs_log": max_abs_log,
        "pro_mass": pro_mass, "con_mass": con_mass,
        "signed_mass": signed_mass, "abs_signed_mass": abs(signed_mass),
        "asymmetry": asymmetry,
        "signed_mass_typesign": typesign_mass, "asymmetry_typesign": asymmetry_typesign,
        "depth_weighted_mass": depth_w_mass, "depth_weighted_signed": depth_w_signed,
        "root_prior": root_prior,
        # flat-star-law predictions of width (for the star-approx R^2 analysis)
        "star_lumped_width": star_lumped_width,
        "star_exact_width": star_exact_width,
        "n_lr_edges": len(abs_logs),
    }
