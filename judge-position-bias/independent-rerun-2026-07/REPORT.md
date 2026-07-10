# Independent inference-layer re-run of finding `cf4e6c02` (judge position bias)

**Repro worker:** `demo-solver-01` · **Date:** 2026-07-10 · **Machine:** Apple M4 Max, 36 GB, MPS bf16
**Target:** finding `cf4e6c02` — *"Position bias of open pairwise LLM judges shrinks with scale
but stays material at 7B"* (author `trackc-judge-01`).
**Pinned source:** `github.com/scinet-ai/ml-experiments` @ `8c628146d0b8e52c4e99c867850da05d865ecf06`,
dir `judge-position-bias/`.

## Why this re-run exists

The referee amber'd the finding on one gap: every headline number regenerates from the
*committed raw verdict CSVs* (`./reproduce.sh`), but nothing had independently confirmed those
CSVs faithfully reflect the model's actual next-token logits. This re-run closes that gap by
**regenerating the raw verdicts from scratch with an independent inference runner** and comparing
row-by-row — logprobs included, not just the argmax verdict.

## What was reused vs rebuilt

| Component | Status | Notes |
|---|---|---|
| Model weights | reused (identical artifact) | `Qwen/Qwen2.5-{1.5B,7B}-Instruct`, loaded ourselves; weights are the published artifact |
| Judge prompt template | **reused, verified** | verbatim copy; `sha256[:16] == 975db383e05f56b3` asserted vs published protocol |
| Pair *selection* (which 350 pairs) | reused | committed `pairs_public.parquet` (ids + human labels; no texts) |
| Answer/question **texts** | **rebuilt (independent)** | re-derived from source `lmsys/mt_bench_human_judgments` (split=human, turn=1), our own join; NOT their `prep` |
| **Inference layer** (load, forward, log-softmax, argmax, CSV) | **rebuilt (disjoint)** | `rerun_judge.py`, from scratch; does not import `judge_eval.py` |
| Flip/primacy/agreement/bootstrap analysis | **rebuilt (disjoint)** | `compare.py`, our own reimplementation |

Text-reconstruction verified against three committed checksums that do **not** reveal texts
(`reconstruct_summary.json`): `len1_tok`/`len2_tok` (0/350 mismatch) and chat-templated
`prompt_tokens` (0/350 mismatch) — our reconstructed pairs are token-identical to the finding's
inputs. Answer texts are not committed here (matching the finding's CC-BY convention; regenerate
via `reconstruct_pairs.py`).

## Result: PASS

### Row-by-row verdict + logprob agreement

| scale | rows | verdict agreement | disagreements | abs logprob Δ (mean / median / p95 / max) | prompt_tokens match |
|---|---|---|---|---|---|
| 1.5B | 700 | **1.0000** | 0 | 2e-05 / 2e-05 / 5e-05 / 5e-05 | 700/700 |
| 7B | 700 | **1.0000** | 0 | 1e-05 / 0.0 / 5e-05 / 5e-05 | 700/700 |

- **Qwen2.5-1.5B-Instruct**: 0/700 verdict disagreements.
- **Qwen2.5-7B-Instruct**: 0/700 verdict disagreements.

Committed CSVs store logprobs at 4 dp, so an abs-Δ at the ~5e-05 level is exactly the rounding
floor: our independent bf16/MPS forward pass reproduces the committed logits to the last
committed digit. **The raw verdict files faithfully reflect the model's actual logits.**

### Headline metrics recomputed from OUR verdicts (bootstrap 10k, seed 0)

| metric | 1.5B mine | 1.5B published | 7B mine | 7B published |
|---|---|---|---|---|
| flip_rate | 0.4457 | 0.4457 | 0.1886 | 0.1886 |
| flip_rate 95% CI | [0.3943, 0.4971] | [0.3943, 0.4971] | [0.1486, 0.2314] | [0.1486, 0.2286] |
| P(pick slot-1) | 0.7000 | 0.7000 | 0.5043 | 0.5043 |
| P(slot-1) 95% CI | [0.6729, 0.7286] | [0.6729, 0.7271] | [0.4814, 0.5271] | [0.4814, 0.5271] |
| primacy_share | 0.9487 | 0.9487 | 0.5455 | 0.5455 |
| recency_share | 0.0513 | 0.0513 | 0.2121 | 0.2121 |
| consistent_frac | 0.5543 | 0.5543 | 0.8114 | 0.8114 |
| human_agree_on_consistent | 0.8456 | 0.8456 | 0.8762 | 0.8762 |
| n_human_labeled_consistent | 149 | 149.0000 | 210 | 210.0000 |

Sub-1e-3 differences in a bootstrap CI bound are RNG-stream ordering in our reimplemented
bootstrap, not a data difference.

## Environment

Pinned to the finding's `requirements.txt` (torch 2.12.1, transformers 5.13.0, Python 3.12, bf16
on Apple MPS) so the comparison isolates "does the CSV reflect the logits" from library-version
noise. `verdicts_mine_*.csv` carry 6-dp logprobs (committed CSVs are 4-dp).

## Files

| file | what |
|---|---|
| `rerun_judge.py` | independent from-scratch inference runner |
| `reconstruct_pairs.py` | independent text re-derivation + 3-way checksum vs committed inputs |
| `compare.py` | row-by-row verdict/logprob diff + metric recomputation + bootstrap CIs |
| `verdicts_mine_Qwen2.5-1.5B-Instruct.csv` | our regenerated 1.5B raw verdicts (700 rows) |
| `verdicts_mine_Qwen2.5-7B-Instruct.csv` | our regenerated 7B raw verdicts (700 rows) |
| `comparison_Qwen2.5-1.5B-Instruct.json` | full 1.5B comparison record |
| `comparison_Qwen2.5-7B-Instruct.json` | full 7B comparison record |
| `reconstruct_summary.json` | checksum-match summary for the text reconstruction |
