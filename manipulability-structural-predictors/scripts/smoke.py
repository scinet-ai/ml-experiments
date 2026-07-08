"""Smoke test: generate one graph, compile, compute exact posterior_range, assert
is_exact on a polytree. Also print the object model we rely on downstream."""
import random
import warnings

from probability_flow import __version__
from probability_flow.aspic.generate import (
    ArgumentGenerator, StructuralParams, generate,
)
from probability_flow.metrics import (
    posterior_range, manipulability, circuit_rank, is_polytree, max_depth,
    upstream_size,
)

print("probability_flow version:", __version__)

# --- 1. simplest possible: independent-evidence polytree -------------------
params = StructuralParams.independent_only(n_support=2, n_attack=1,
                                           min_depth=1, max_depth=3)
arg = generate(seed=7, structural=params)
bn = arg.bn
target = arg.target

cr = circuit_rank(bn, target)
poly = is_polytree(bn, target)
print(f"\ncircuit_rank={cr}  is_polytree={poly}  "
      f"max_depth={max_depth(bn, target)}  upstream_size={upstream_size(bn, target)}")

with warnings.catch_warnings():
    warnings.simplefilter("error")  # any ManipulabilityWarning -> exception
    res = posterior_range(bn, target, exact=True)
lo, hi = res
print(f"exact posterior_range = ({lo:.6f}, {hi:.6f})  is_exact={res.is_exact}  "
      f"width={hi-lo:.6f}")
assert res.is_exact, "expected exact on a polytree"
assert 0.0 <= lo <= hi <= 1.0

# outer bound must contain the exact range
olo, ohi = posterior_range(bn, target, exact=False)
print(f"outer bound          = ({olo:.6f}, {ohi:.6f})")
assert olo <= lo + 1e-9 and hi <= ohi + 1e-9, "exact range must sit inside outer bound"

# full-reveal posterior with the reveal-model convention (root prior = 0.5)
true_post = arg.posterior(target)
saved = target.prior
target.prior = 0.5
fresh = target.compile()
from probability_flow.core import LoopySolver
post_half = LoopySolver(fresh).prob(target, 1)
target.prior = saved
print(f"true posterior (calibrated prior {saved}) = {true_post:.6f}")
print(f"reveal-model full-reveal posterior (prior 0.5) = {post_half:.6f}")
assert lo - 1e-6 <= post_half <= hi + 1e-6, "full-reveal must lie in exact range"
assert lo - 1e-6 <= 0.5 <= hi + 1e-6, "empty-reveal (0.5) must lie in exact range"

# --- 2. show the argument-level dict we extract features from ---------------
d = arg.to_dict()
print("\nnodes:", len(d["nodes"]), " edges:", len(d["edges"]))
print("edge types:", sorted({e["type"] for e in d["edges"]}))
print("cpd types :", sorted({n["cpd_type"] for n in d["nodes"]}))
print("sample edge:", d["edges"][0])

# --- 3. a graph WITH undercut/strict to confirm feature plumbing ------------
rich = StructuralParams(n_support=2, n_attack=1, undercut_prob=0.4,
                        strict_prob=0.3, undermine_prob=0.3, axiom_prob=0.3,
                        max_depth=3, max_fanin=2)
arg2 = generate(seed=3, structural=rich)
d2 = arg2.to_dict()
print("\nrich edge types:", sorted({e["type"] for e in d2["edges"]}))
print("rich cpd types :", sorted({n["cpd_type"] for n in d2["nodes"]}))
print("rich circuit_rank:", circuit_rank(arg2.bn, arg2.target),
      " is_exact:", posterior_range(arg2.bn, arg2.target, exact=True).is_exact)
print("\nSMOKE OK")
