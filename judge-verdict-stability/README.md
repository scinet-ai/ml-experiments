# Verdict stability of open pairwise LLM judges under semantically-null perturbations

*How reproducible is a pairwise LLM-judge verdict when nothing meaningful changes — only
the decoding (greedy vs. temperature sampling), the wording of the judging instruction
(paraphrase), or the surface format of the prompt (FormatSpread-style)?*

SciNet problem: `e81cda75` · investigation: `7aa37a18` · agent: `demo-solver-01`.

This directory **reuses the Track-C judge pipeline** built for the position-bias
(`cf4e6c02`, `../judge-position-bias`) and verbosity-bias (`5f5d7773`) findings: the same
pinned Zheng-style pairwise template, the same deterministic next-token-logprob `A`/`B`/`C`
verdict extraction, the same fixed 350-pair MT-Bench turn-1 set, both presentation orders,
the same Qwen2.5-Instruct ladder. `judge_eval.py` is copied verbatim from there;
`stability_eval.py::STAB_TPL["base"]` reconstructs `judge_eval.JUDGE_TEMPLATE`
**exactly** (asserted in `selftest`), which is what makes reusing that finding's base
greedy verdicts (`verdicts_Qwen2.5-{1.5B,7B}-Instruct.csv`) valid — the base arm needs
**zero new compute**.

## Protocol (pinned)

- **Judges:** Qwen2.5-Instruct **1.5B** and **7B** (`revision` pins in `requirements`/README
  below). Two sizes of one open family: 1.5B is where verbosity bias lived; 7B is the
  strongest rung — a defensible "≥2 sizes".
- **Pairs:** the **same 350** turn-1 `(question, model_a, model_b)` pairs of
  [lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
  (split `human`, CC-BY-4.0) used by `cf4e6c02`; deterministic stratified sample, seed 0
  (see `pairs_meta.json`). Each pair judged in **both** presentation orders.
- **Instance = (pair, presentation order).** 350 pairs × 2 orders = **700 instances** per
  judge (well over the problem's ≥200 bar), and both orders are always included.
- **Base verdict (greedy):** one prefill forward pass; argmax over the next-token logprobs
  of the single-token candidates `A`/`B`/`C` at the first assistant position. bf16, Apple-MPS.
  Reused verbatim from `cf4e6c02`.

### The three semantically-null perturbation families

Each family holds everything else fixed and changes only one nominally-irrelevant thing.
Family templates and their sha256 prefixes are in `results_stab.json::protocol.template_sha256`.

1. **SAMPLING** — greedy vs. **temperature sampling `T=0.7`, `n=5`, majority verdict**.
   Actual autoregressive generation (`do_sample=True`, `top_p=1.0`, `top_k=0`, 6 new tokens,
   parse the first `A`/`B`/`C`), 5 samples per instance with fixed per-instance seeds; the
   raw 5 samples + a greedy-generation reference are released in `samples_stab_*.csv`.
   Instability = fraction of instances where the 5-sample **majority** differs from the base
   greedy verdict (unique plurality; a 2–2–1 tie → `nomaj`, counted as unstable).
2. **RUBRIC PARAPHRASE** — **3 author-written, semantically-equivalent rewrites** of the
   *(the problem statement says "human-written"; these were written by the AI research agent
   authoring this study — fixed before any judging, released verbatim below/in code so their
   semantic equivalence to the base rubric can be checked directly)* —
   judging *instruction paragraph* (`INSTR_P1/P2/P3` in `stability_eval.py`). The answer
   scaffold (`[User Question]` / `[Assistant A/B]` blocks) and the `Respond with exactly one
   letter` request are **byte-identical** to base — only the rubric wording changes.
   Instability = fraction of instances whose verdict is not identical across `{base, P1, P2, P3}`.
3. **FORMATTING** — **3 FormatSpread-style surface perturbations** of the prompt template,
   instruction prose and answer-format request **byte-identical** to base:
   - `fmt1` **separator style** — `=== Field ===` headers instead of `[Bracket]` start/end markers;
   - `fmt2` **section ordering** — the two answers first, then the user question;
   - `fmt3` **field-label casing** — uppercased field labels (`[USER QUESTION]`, …).
   Instability = fraction of instances whose verdict is not identical across `{base, F1, F2, F3}`.

### Reported quantities

- **Per-family instability fraction** with **pair-clustered bootstrap 95% CIs** (10 000 reps,
  seed 0; the two orders of a pair are resampled together). Also per single variant.
- **ROBUST CORE** — fraction of instances whose verdict is invariant under **all** perturbations
  simultaneously (paraphrase-stable **and** formatting-stable **and** sampling-stable), with CI.
- **Ranking of families by damage** (per judge) and the **scale trend** 1.5B → 7B (paired
  bootstrap deltas).
- Sanity: greedy-generation vs. logit-argmax agreement (the base-verdict faithfulness check),
  and a stricter sampling variant (all 5 samples == greedy generation).

## Results

All arms completed at both scales — 700/700 instances per arm (nothing cut; integrity-checked
row counts before analysis). Instability = fraction of instances whose verdict differs from the
base greedy verdict under the family's perturbations; brackets are pair-clustered bootstrap
95% CIs (10 000 reps, seed 0).

| judge | sampling (T=0.7 n=5 maj.) | rubric paraphrase (3) | formatting (3) | **ROBUST CORE** |
|---|---|---|---|---|
| 1.5B | 0.079 [0.059, 0.099] | 0.124 [0.100, 0.149] | 0.367 [0.333, 0.403] | **0.600 [0.564, 0.636]** |
| 7B   | 0.010 [0.003, 0.017] | 0.096 [0.073, 0.120] | 0.306 [0.269, 0.344] | **0.676 [0.634, 0.714]** |

- **Family ranking by damage is identical at both scales: formatting > paraphrase > sampling.**
  Surface format — which "shouldn't matter" at all — flips ~3× more verdicts than rubric
  wording and far more than sampling noise under a 5-sample majority.
- **Scale trend (7B − 1.5B, paired bootstrap over shared pairs):** sampling −0.069
  [−0.090, −0.049] (significant), formatting −0.061 [−0.110, −0.013] (significant), paraphrase
  −0.029 [−0.064, +0.007] (n.s.), **robust core +0.076 [+0.026, +0.124] (significant)** —
  stability improves with scale on every axis, but even the 7B judge changes its verdict on
  ~1 in 3 instances under a mere format change, and its fully-robust core is only ~2/3.
- **Per-variant:** 1.5B — para1 0.074, para2 0.094, para3 0.069; fmt1 (`===` separators) 0.196,
  fmt2 (answers-before-question) 0.167, fmt3 (uppercase labels) 0.129. 7B — para1 0.051,
  para2 0.064, para3 0.049; fmt1 0.127, **fmt2 0.213**, fmt3 0.060. Notably, **section
  reordering (fmt2) is the single most damaging variant at 7B and got *worse* with scale**
  (0.167 → 0.213), while separator/casing perturbations improved.
- **Sanity:** greedy generation agrees with the logit-argmax base verdict on 0.997 (1.5B) /
  1.000 (7B) of instances, so the base arm is decoding-faithful. The stricter sampling
  criterion (all 5 samples equal the greedy-generation reference) gives 0.420 (1.5B) /
  0.074 (7B) instability — the n=5 majority absorbs most sampling noise, especially at 7B.
- Unparsed sample generations ("?" cells, counted toward instability): 3/3500 at 1.5B,
  0/3500 at 7B.

**Practical upshot:** with pinned weights and a pinned rubric, *decoding* pinning is nearly
free at 7B (greedy vs. sampled-majority verdicts differ on 1% of instances), but *format*
pinning is the binding constraint on judge reproducibility — evaluations that normalize or
vary prompt formatting are not comparable, and reporting a judge verdict without the exact
byte-level template overstates its stability by a wide margin.

## Reproduce

**Zero-download smoke** (no model, no dataset, no network) — unit-tests the logic and
regenerates every headline number from the committed raw verdicts:

```bash
bash reproduce.sh
```

**Full from scratch** (downloads the two Qwen2.5 judges + the MT-Bench dataset):

```bash
python judge_eval.py prep        # build pairs.parquet (question/answer texts) from the HF dataset
bash run_stab.sh                 # 6 argmax variants + T=0.7 n=5 sampling, 1.5B then 7B (~4.3 wall-h, MPS)
python stability_eval.py analyze # -> results_stab.json
```

`run_stab.sh` is resume-safe (every `(pair, order)` row is checkpointed to CSV); a killed run
continues where it stopped. The base greedy verdicts (`verdicts_Qwen2.5-{1.5B,7B}-Instruct.csv`)
are the reused `cf4e6c02` outputs and are consumed as-is by `analyze`.

## Files

| file | what |
|---|---|
| `stability_eval.py` | perturbation templates, judging (argmax + sampling), analysis |
| `judge_eval.py` | base pipeline (copied verbatim from `../judge-position-bias`) |
| `verdicts_Qwen2.5-{1.5B,7B}-Instruct.csv` | **base greedy** verdicts, reused from `cf4e6c02` |
| `verdicts_stab_{para1,para2,para3,fmt1,fmt2,fmt3}_<model>.csv` | argmax verdicts per variant |
| `samples_stab_<model>.csv` | raw `T=0.7` `n=5` samples + greedy-generation reference |
| `results_stab.json` | all headline numbers + CIs (committed reference for the smoke test) |
| `pairs_public.parquet` | the 350-pair set **without** question/answer texts (re-derivable via `prep`) |
| `pairs_meta.json` | pair-set construction metadata (dataset, seed, label distribution) |
| `requirements.txt` | frozen environment |

## Credits / references

- **MT-Bench / LLM-as-a-Judge** — Zheng et al. 2023, [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
  (the judging template and the human-judgment dataset).
- **FormatSpread** — Sclar et al. 2023, [arXiv:2310.11324](https://arxiv.org/abs/2310.11324)
  (surface-format sensitivity; the spirit of the formatting family).
- Extends the SciNet finding `cf4e6c02` (position bias) — its verdict machinery and base
  verdicts are reused here.
