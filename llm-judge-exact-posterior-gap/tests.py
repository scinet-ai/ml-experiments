"""Unit checks: enumeration exactness on hand-computable graphs, reveal-set
legality, and the FINAL: parser.  Run: python tests.py"""

import math
from substrate import Node, Graph, make_graph, sample_sets_for_llm, set_id
from nl_render import DOMAIN_NAMES, build_prompt, kent
from judge import parse_final


def approx(x, y, tol=1e-9):
    return abs(x - y) <= tol


def test_enumeration_2node():
    # H -> E1, H -> E2 ; p0=0.5 ; E1 a=.8 b=.2 ; E2 a=.7 b=.3 ; world H=1,E1=1,E2=0
    nodes = [
        Node("H", 0, None, None, None, "root", 0),
        Node("E1", 1, "H", 0.8, 0.2, "E", 0),
        Node("E2", 1, "H", 0.7, 0.3, "E", 1),
    ]
    g = Graph(0, 0.5, nodes, {"H": 1, "E1": 1, "E2": 0}, "ship")
    assert approx(g.posterior([]), 0.5), g.posterior([])
    assert approx(g.posterior(["E1"]), 0.8), g.posterior(["E1"])
    # P(H=1|E1=1,E2=0) = 0.12/0.19
    assert approx(g.posterior(["E1", "E2"]), 0.12 / 0.19), g.posterior(["E1", "E2"])
    print("  ok: 2-node hand computation")


def test_enumeration_chain():
    # H -> E1 -> F1 ; p0=0.5 ; E1 a=.8 b=.2 ; F1 a=.9 b=.1 ; world all 1
    nodes = [
        Node("H", 0, None, None, None, "root", 0),
        Node("E1", 1, "H", 0.8, 0.2, "E", 0),
        Node("F1_1", 2, "E1", 0.9, 0.1, "F", 0),
    ]
    g = Graph(0, 0.5, nodes, {"H": 1, "E1": 1, "F1_1": 1}, "ship")
    # P(H=1|F1=1) = 0.37/0.50 = 0.74
    assert approx(g.posterior(["F1_1"]), 0.74), g.posterior(["F1_1"])
    # revealing E1=1 too: P(H=1|E1=1)=0.8 (F1 conditionally independent of H given E1)
    assert approx(g.posterior(["E1", "F1_1"]), 0.8), g.posterior(["E1", "F1_1"])
    print("  ok: 2-layer chain hand computation")


def test_reveal_legality():
    nodes = [
        Node("H", 0, None, None, None, "root", 0),
        Node("E1", 1, "H", 0.8, 0.2, "E", 0),
        Node("F1_1", 2, "E1", 0.9, 0.1, "F", 0),
        Node("E2", 1, "H", 0.7, 0.3, "E", 1),
    ]
    g = Graph(0, 0.5, nodes, {"H": 1, "E1": 1, "F1_1": 1, "E2": 0}, "ship")
    legal = g.legal_reveal_sets()
    # F1_1 present implies E1 present in every legal set
    for s in legal:
        if "F1_1" in s:
            assert "E1" in s, s
        assert "H" not in s
    # illegal: F without parent
    assert not g.is_legal({"F1_1"})
    assert g.is_legal({"E1", "F1_1"})
    assert g.is_legal(set())
    # count: E-subsets 2^2=4; E1 in -> F1 toggle (x2). sets = for each (E1,E2) mask:
    #   E1 out: F1 out -> 1 ; E1 in: F1 in/out -> 2. times E2 in/out (2).
    #   -> (1*2)+(2*2) = 6 legal sets
    assert len(legal) == 6, len(legal)
    print(f"  ok: reveal-set legality ({len(legal)} legal sets)")


def test_parser():
    cases = [
        ("FINAL: 0.42", 0.42),
        ("blah blah\nFINAL: 0.7\n", 0.7),
        ("I think about 0.3 chance.\nFINAL: 0.55", 0.55),
        ("FINAL:0.9", 0.9),
        ("final: .25", 0.25),
        ("The odds are 0.62 in my view.", 0.62),          # fallback
        ("prob 0.1 then reconsider 0.88 finally", 0.88),  # last in-range
        ("no numbers here", None),
        ("FINAL: 1.5 (out of range) but 0.4 works", 0.4), # skip OOR, take fallback
    ]
    for text, want in cases:
        got = parse_final(text)
        if want is None:
            assert got is None, (text, got)
        else:
            assert got is not None and approx(got, want, 1e-9), (text, got, want)
    print(f"  ok: parser ({len(cases)} cases)")


def test_graph_generation():
    for i in range(20):
        dom = DOMAIN_NAMES[i % len(DOMAIN_NAMES)]
        g = make_graph(i, dom)
        n = len(g.nodes)
        assert 5 <= n <= 9, (i, n)
        # pro/con guarantee
        pri = g.prior()
        pro = sum(1 for nid in g.non_root if g.posterior([nid]) > pri + 1e-9)
        con = sum(1 for nid in g.non_root if g.posterior([nid]) < pri - 1e-9)
        assert pro >= 1 and con >= 1, (i, pro, con)
        # legality of sampled LLM sets, and anchors present
        legal = g.legal_reveal_sets()
        sel = sample_sets_for_llm(g, cap=24)
        assert all(g.is_legal(s) for s in sel)
        if len(legal) > 24:
            assert len(sel) == 24, (i, len(sel))
            assert frozenset() in sel
            full = max(legal, key=len)
            assert full in sel
        # prompt builds without error and contains FINAL sentinel
        p = build_prompt(g, sel[-1], "numeric", skeptical=False)
        assert "FINAL:" in p
        pv = build_prompt(g, sel[-1], "verbal", skeptical=False)
        assert "FINAL:" in pv
    print("  ok: 20-graph generation (node bounds, pro/con, legality, prompts)")


def test_kent_bins():
    assert kent(0.05) == "almost never"
    assert kent(0.2) == "rarely"
    assert kent(0.4) == "sometimes"
    assert kent(0.6) == "often"
    assert kent(0.8) == "very often"
    assert kent(0.95) == "almost always"
    print("  ok: Sherman-Kent bins")


if __name__ == "__main__":
    print("Running unit checks...")
    test_enumeration_2node()
    test_enumeration_chain()
    test_reveal_legality()
    test_parser()
    test_kent_bins()
    test_graph_generation()
    print("ALL UNIT CHECKS PASSED")
