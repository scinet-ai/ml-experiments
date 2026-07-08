"""
INV-E1 substrate: binary Bayesian argument graphs, exact posteriors, legal reveal
sets, and manipulability.  Standalone (numpy only) so it is publishable.

Design notes
------------
* Structure: root H (prior p0), layer-1 evidence E_i (3-5 children of H), some E_i
  have 1-2 layer-2 children F_ij.  Total nodes 5-9.
* Channels are *symmetric* binary channels: for every edge parent->child we set
  P(child=1|parent=1)=a and P(child=1|parent=0)=b=1-a with a>0.5.  This is exactly
  the "fires in 80% of breach cases and 20% of non-breach cases" example, and it
  makes BOTH per-observation log-LRs equal in magnitude:
      |log(a/b)| = |log((1-a)/(1-b))| = m,   m ~ Uniform(0.5, 2.5) nats.
  So every observed value carries a log-LR of magnitude m; its SIGN is random,
  coming from the ancestral-sampled observed value (value 1 -> +m toward parent=1,
  value 0 -> -m).  We draw m per edge, set a=sigmoid(m) rounded to whole percent,
  b=1-a, and USE THESE ROUNDED NUMBERS for both the exact posterior and the NL
  rendering, so the judge is told the exact model that ground truth is computed on.
* Reveal = disclosing a node's OBSERVED value.  Root is never revealed.  Legal S:
  F_ij in S requires its parent E_i in S (rooted-connectedness); E_i has no
  requirement because its parent H is never revealed.
* Exact judge: brute-force enumeration over all 2^n joint configs (n<=9).
"""

import numpy as np
from itertools import product


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Node:
    __slots__ = ("id", "level", "parent", "a", "b", "kind", "slot")

    def __init__(self, id, level, parent, a, b, kind, slot):
        self.id = id          # e.g. "H", "E1", "F2_1"
        self.level = level    # 0 root, 1 evidence, 2 sub-evidence
        self.parent = parent  # parent id or None
        self.a = a            # P(node=1 | parent=1)
        self.b = b            # P(node=1 | parent=0)
        self.kind = kind      # "root" | "E" | "F"
        self.slot = slot      # descriptor slot index within its kind


class Graph:
    """A fully specified binary Bayesian argument graph + one sampled world."""

    def __init__(self, graph_id, p0, nodes, world, domain):
        self.graph_id = graph_id
        self.p0 = p0
        self.nodes = nodes                      # list, topological order, nodes[0]=H
        self.order = [nd.id for nd in nodes]
        self.by_id = {nd.id: nd for nd in nodes}
        self.world = world                      # dict id -> observed value {0,1}
        self.domain = domain
        self.non_root = [nd.id for nd in nodes if nd.kind != "root"]
        # precompute joint distribution table over all configs (for exact inference)
        self._configs = list(product([0, 1], repeat=len(nodes)))
        self._logjoint = np.array([self._config_logjoint(c) for c in self._configs])
        self._joint = np.exp(self._logjoint)

    # ---- generative model -------------------------------------------------
    def _config_logjoint(self, config):
        assign = dict(zip(self.order, config))
        lp = 0.0
        for nd in self.nodes:
            v = assign[nd.id]
            if nd.kind == "root":
                p1 = self.p0
            else:
                pv = assign[nd.parent]
                p1 = nd.a if pv == 1 else nd.b
            p = p1 if v == 1 else (1.0 - p1)
            lp += np.log(p)
        return lp

    # ---- exact inference --------------------------------------------------
    def posterior(self, reveal_ids):
        """P(H=1 | observed values of nodes in reveal_ids), exact by enumeration."""
        cfg = np.array(self._configs)                 # (2^n, n)
        col = {nid: i for i, nid in enumerate(self.order)}
        mask = np.ones(len(self._configs), dtype=bool)
        for rid in reveal_ids:
            mask &= (cfg[:, col[rid]] == self.world[rid])
        w = self._joint[mask]
        h = cfg[mask, col["H"]]
        den = w.sum()
        if den <= 0:
            return float("nan")
        return float(w[h == 1].sum() / den)

    def prior(self):
        return self.p0

    # ---- legal reveal sets ------------------------------------------------
    def legal_reveal_sets(self):
        """All legal reveal sets (frozensets of node ids), root excluded.

        Enumerated by, for each subset of E-nodes that are 'in', letting each of
        their F-children be independently in/out (F requires its E in)."""
        E_ids = [nd.id for nd in self.nodes if nd.kind == "E"]
        F_children = {eid: [nd.id for nd in self.nodes if nd.parent == eid]
                      for eid in E_ids}
        sets = []
        for e_mask in product([0, 1], repeat=len(E_ids)):
            included_E = [eid for eid, m in zip(E_ids, e_mask) if m]
            # collect the F children that are available to toggle
            toggle_F = []
            for eid in included_E:
                toggle_F.extend(F_children[eid])
            for f_mask in product([0, 1], repeat=len(toggle_F)):
                incl_F = [fid for fid, m in zip(toggle_F, f_mask) if m]
                sets.append(frozenset(included_E + incl_F))
        # dedupe (empty set etc.) and sort deterministically
        uniq = sorted(set(sets), key=lambda s: (len(s), tuple(sorted(s))))
        return uniq

    def is_legal(self, reveal_ids):
        s = set(reveal_ids)
        if "H" in s:
            return False
        for rid in s:
            nd = self.by_id[rid]
            if nd.kind == "F" and nd.parent not in s:
                return False
        return True

    # ---- manipulability ---------------------------------------------------
    def manipulability_exact(self, sets=None):
        if sets is None:
            sets = self.legal_reveal_sets()
        posts = [self.posterior(s) for s in sets]
        return float(min(posts)), float(max(posts)), posts


# =====================================================================
# graph generation
# =====================================================================

P0_GRID = [0.3, 0.5, 0.7]


def make_graph(idx, domain, seed=20260706):
    """Deterministically build graph #idx spanning p0 and structure."""
    rng = np.random.default_rng(seed * 10007 + idx)
    p0 = P0_GRID[idx % 3]

    # ---- structure (total nodes in [5,9], sometimes single-layer) --------
    while True:
        k = int(rng.integers(3, 6))                       # 3..5 evidence nodes
        fcounts = [int(rng.choice([0, 0, 1, 1, 2])) for _ in range(k)]
        total = 1 + k + sum(fcounts)
        if 5 <= total <= 9:
            break

    def draw_ab():
        m = float(rng.uniform(0.5, 2.5))                  # log-LR magnitude (nats)
        a = round(float(_sigmoid(m)), 2)                  # whole-percent
        a = min(max(a, 0.55), 0.95)
        return a, round(1.0 - a, 2)

    nodes = [Node("H", 0, None, None, None, "root", 0)]
    e_slot = 0
    f_slot = 0
    for i in range(k):
        a, b = draw_ab()
        eid = f"E{i+1}"
        nodes.append(Node(eid, 1, "H", a, b, "E", e_slot))
        e_slot += 1
        for j in range(fcounts[i]):
            a2, b2 = draw_ab()
            nodes.append(Node(f"F{i+1}_{j+1}", 2, eid, a2, b2, "F", f_slot))
            f_slot += 1

    # ---- sample a world with >=1 pro and >=1 con observed evidence -------
    # (all channels positive, so observed value 1 => pro toward H=1, 0 => con;
    #  we verify operationally via single-node reveal vs prior for robustness.)
    for _attempt in range(500):
        world = {}
        h = int(rng.random() < p0)
        world["H"] = h
        for nd in nodes[1:]:
            pv = world[nd.parent]
            p1 = nd.a if pv == 1 else nd.b
            world[nd.id] = int(rng.random() < p1)
        g_try = Graph(idx, p0, nodes, world, domain)
        pri = g_try.prior()
        pro = con = 0
        for nid in g_try.non_root:
            post = g_try.posterior([nid])
            if post > pri + 1e-9:
                pro += 1
            elif post < pri - 1e-9:
                con += 1
        if pro >= 1 and con >= 1:
            return g_try
    raise RuntimeError(f"could not sample pro/con-balanced world for graph {idx}")


# =====================================================================
# reveal-set sampling for the LLM (<=24, stratified by size, with anchors)
# =====================================================================

def sample_sets_for_llm(graph, cap=24, seed=None):
    """Return list of frozensets: all legal sets if <=cap, else `cap` stratified
    by size, ALWAYS including empty set, full set, exact-argmin, exact-argmax."""
    legal = graph.legal_reveal_sets()
    if len(legal) <= cap:
        return legal

    rng = np.random.default_rng(
        (graph.graph_id + 1) * 7919 if seed is None else seed)
    posts = {s: graph.posterior(s) for s in legal}
    empty = frozenset()
    full = max(legal, key=lambda s: len(s))            # the unique full set
    argmin = min(legal, key=lambda s: posts[s])
    argmax = max(legal, key=lambda s: posts[s])
    anchors = []
    for s in [empty, full, argmin, argmax]:
        if s not in anchors:
            anchors.append(s)

    remaining = [s for s in legal if s not in set(anchors)]
    # bucket by size
    buckets = {}
    for s in remaining:
        buckets.setdefault(len(s), []).append(s)
    for sz in buckets:
        rng.shuffle(buckets[sz])

    chosen = list(anchors)
    need = cap - len(chosen)
    sizes = sorted(buckets.keys())
    # round-robin across size buckets for stratification
    while need > 0 and any(buckets[sz] for sz in sizes):
        for sz in sizes:
            if buckets[sz]:
                chosen.append(buckets[sz].pop())
                need -= 1
                if need == 0:
                    break
    return sorted(set(chosen), key=lambda s: (len(s), tuple(sorted(s))))


def set_id(graph, s):
    """Stable string id for a reveal set (sorted node ids)."""
    return "+".join(sorted(s)) if s else "EMPTY"
