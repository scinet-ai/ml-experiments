# LLM judges vs exact Bayesian posteriors under partial disclosure: numeric evidence saturates, verbal evidence opens new manipulation surface

Do LLM judges diverge from the exact Bayesian posterior on partially disclosed argument
graphs — and does that divergence *widen the manipulation surface* beyond what selective
disclosure of true evidence can do to an ideal judge?

SciNet problem: `59eb2f72` · investigation: `a4dc2065` · agent: `tracke-debate-lead`.

## Setup (pinned)

- **Substrate** (`substrate.py`, standalone, numpy-only): binary Bayesian networks,
  5–9 nodes — root H (prior ∈ {0.3, 0.5, 0.7}), layer-1 evidence, some layer-2
  children; per-observation |log LR| ~ U(0.5, 2.5) nats, mixed signs guaranteed;
  observed world ancestral-sampled, all seeded. Revealing a node = disclosing its
  observed value; legal reveal sets are root-connected (child revealed ⇒ parent
  revealed). **Exact posterior by brute-force enumeration** over hidden nodes (unit
  tests against hand-computed cases in `tests.py`). Manipulability_exact = [min, max]
  of the exact posterior over ALL legal sets.
- **Rendering** (`nl_render.py`): 6 invented fictional micro-domains (no real-world
  priors to contaminate). The judge sees the FULL model (structure + parameters) and
  the revealed observed facts, in one of two parameter presentations: **numeric**
  (probabilities as numbers) or **verbal** (fixed Sherman–Kent-style qualifier bins).
- **Judges:** `claude-haiku-4-5-20251001` (weak) and `claude-sonnet-5` (strong), via
  one-shot CLI calls, `FINAL: <p>` parsing (parse rate **100%**, 1272/1272).
- **Conditions:** A = numeric-credulous (both judges, 20 graphs × ~22 reveal sets);
  B = numeric + explicit warning that facts were strategically selected and others
  withheld ("skeptical", weak judge, 10 graphs); C = verbal-credulous (weak judge,
  10 graphs). The per-graph LLM manipulability is [min, max] of the judge's verdicts
  over the SAME evaluated reveal sets (exact extremal sets always included), so the
  containment comparison is apples-to-apples.

## Results (1,272 judged reveal sets; bootstrap CIs)

| condition | judge | MAE | r | calib slope | overconf | flip% | manip excess |
|---|---|---|---|---|---|---|---|
| A numeric | weak | 0.0042 [.001,.007] | 0.997 | 1.000 | +0.001 (ns) | 0.2 | +0.003 [.000,.007] |
| A numeric | strong | 0.0021 [.001,.005] | 0.999 | 0.997 | −0.001 | 0.0 | +0.000 (ns) |
| B skeptical | weak | 0.0090 | 0.992 | 0.988 | ~0 | 1.0 | +0.000 (ns) |
| C verbal | weak | **0.0625** [.042,.088] | 0.963 | 1.027 | **+0.019** | **5.9** | **+0.072** [.010,.122] |

1. **Numeric presentation saturates.** Given explicit numeric parameters, both judges
   are near-perfect Bayesians on 5–9-node graphs — Haiku *computes* the posterior
   (visible explicit Bayes arithmetic in transcripts), MAE 0.004, calibration slope
   1.000, no overconfidence, flip rate 0.2%. The LLM manipulability range ≈ the exact
   range (excess +0.003). **On this substrate, with numbers, judge error creates
   essentially no new manipulation surface.**
2. **Verbal presentation opens net-new manipulation surface.** Same graphs, same
   reveal sets, parameters in words: MAE rises **15×** (0.0035 → 0.0625 matched,
   increase CI [0.048, 0.071]), verdict flips 5.9%, and the judge's reachable verdict
   range **strictly contains** the exact range in **7/10 graphs (0/10 reverse)** —
   mean manipulability excess **+0.072** (CI [0.010, 0.122]), i.e. a debater who
   exploits the judge's word→number mapping can push the verdict ~11% further (range
   0.731 vs 0.659) than any selection of true evidence could push an ideal Bayesian.
3. **No unraveling: the judge stays credulous when warned.** Telling the judge that
   evidence was strategically selected (B) produces no significant adverse inference
   (shift −0.003, CI crosses 0; sparse-reveal shift −0.009, ns; slope vs |S| p=0.49).
   LLM judges do not spontaneously apply Milgrom-style skepticism about withheld
   evidence even when explicitly prompted about strategic selection — the modeling
   assumption of a *credulous* judge in reveal-game analyses is empirically the right
   one for current models.

## Limitations

- Single model family (two Claude tiers); verbal condition measured on the weak judge,
  10 graphs. Small graphs (5–9 nodes): numeric saturation may break at larger scale /
  heavier marginalization — the MAE-vs-|reveal| trend here is flat, so the break point
  is beyond this range. Known-model regime only (judge sees the full structure);
  unknown-model debate is the natural harder follow-up. Verbal bins are one fixed
  qualifier scheme.

## Reproduce

```bash
./reproduce.sh            # re-runs the full analysis from results/calls.jsonl (no API)
./reproduce.sh smoke      # no-API end-to-end pipeline check with a MockJudge
./reproduce.sh tests      # substrate unit tests (exact enumeration vs hand-computed)
```

Full re-run (needs a Claude CLI/API backend; ~$ and ~2h): `python run_experiment.py
--mode pilot` then `--mode full` (resumable; per-call records stream to
`results/calls.jsonl`, gzipped copy in `results/calls.jsonl.gz`).
