#!/usr/bin/env python3
"""Verdict stability of open pairwise LLM judges under semantically-null perturbations.

Problem e81cda75. Reuses the Track-C judge pipeline (judge_eval.py): the pinned
Zheng-style template, deterministic next-token-logprob A/B/C verdicts, the fixed
350-pair MT-Bench turn-1 set (lmsys/mt_bench_human_judgments), both presentation
orders. Base greedy verdicts are REUSED from verdicts_<model>.csv (finding cf4e6c02).

Three semantically-null perturbation families, all measured on the SAME instances
(pair x order):
  (1) SAMPLING       greedy (argmax) vs T=0.7 n=5 majority (actual generation).
  (2) RUBRIC PARAPHRASE  3 author-written (AI research agent) semantically-equivalent rewrites of the
                     judging instruction (scaffold + answer-format request identical).
  (3) FORMATTING     3 FormatSpread-style surface perturbations (separator style,
                     section ordering, field-label casing); instruction + answer-format
                     request identical.

Metrics: per-family instability fraction (verdict changes vs base) with pair-clustered
bootstrap 95% CIs; ROBUST CORE = fraction of instances invariant under ALL perturbations
simultaneously; ranking of families by damage; scale trend.

Subcommands:
  judge   --model M --variant {para1,para2,para3,fmt1,fmt2,fmt3} [--limit N]
                                         -> verdicts_stab_<variant>_<model>.csv (argmax)
  sample  --model M [--limit N]          -> samples_stab_<model>.csv (T=0.7 n=5 gen)
  analyze                                -> results_stab.json
  selftest                               (no model, no downloads)
"""
import argparse, json, os, re, time, hashlib
from collections import Counter
from pathlib import Path

import judge_eval as JE

HERE = Path(__file__).resolve().parent

# ---- template decomposition (base == JE.JUDGE_TEMPLATE, asserted in selftest) --------
INSTR_BASE = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI "
    "assistants to the user question displayed below. You should choose the assistant that answers "
    "the user's question better. Your evaluation should consider factors such as the helpfulness, "
    "relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any "
    "position biases and ensure that the order in which the responses were presented does not "
    "influence your decision. Do not allow the length of the responses to influence your "
    "evaluation. Do not favor certain names of the assistants. Be as objective as possible."
)
SCAFFOLD_BASE = (
    "\n\n[User Question]\n{q}\n\n"
    "[The Start of Assistant A's Answer]\n{a}\n[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n{b}\n[The End of Assistant B's Answer]\n\n"
)
TRAILER_BASE = (
    "Which assistant's answer is better? Respond with exactly one letter: \"A\" if assistant A's "
    "answer is better, \"B\" if assistant B's answer is better, or \"C\" for a tie."
)

# ---- (2) RUBRIC PARAPHRASE: three semantically-equivalent instruction rewrites -------
INSTR_P1 = (
    "You are an impartial evaluator. Compare the two AI-assistant responses to the user question "
    "shown below and decide which one answers it better. Weigh qualities such as helpfulness, "
    "relevance, accuracy, depth, creativity, and level of detail. Do not let the order in which the "
    "responses are presented affect your judgment, do not reward or penalize a response for its "
    "length, and pay no attention to the assistants' names. Stay as objective as you can."
)
INSTR_P2 = (
    "Act as a neutral referee for the two assistant answers to the question below, and determine "
    "which answer is better overall. Your assessment should take into account accuracy, relevance, "
    "helpfulness, depth, creativity, and level of detail. Ensure that neither the presentation order "
    "of the answers, nor their length, nor the names of the assistants influences your decision. Be "
    "as objective as possible."
)
INSTR_P3 = (
    "Please serve as a fair and unbiased judge of the two AI-assistant answers to the user's question "
    "given below, selecting whichever one responds to it better. Ground your assessment in factors "
    "like helpfulness, relevance, accuracy, depth, creativity, and level of detail. Your choice must "
    "not be swayed by the order the answers appear in, by how long each answer is, or by the names "
    "given to the assistants; judge as objectively as you possibly can."
)

# ---- (3) FORMATTING: three surface perturbations, instruction & trailer identical ----
# F1 separator style: '=== Header ===' instead of '[Bracket]' start/end markers.
SCAFFOLD_F1 = (
    "\n\n=== User Question ===\n{q}\n\n"
    "=== Assistant A's Answer ===\n{a}\n\n"
    "=== Assistant B's Answer ===\n{b}\n\n"
)
# F2 section ordering: the two answers first, then the user question (same bracket markers).
SCAFFOLD_F2 = (
    "\n\n[The Start of Assistant A's Answer]\n{a}\n[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n{b}\n[The End of Assistant B's Answer]\n\n"
    "[User Question]\n{q}\n\n"
)
# F3 field-label casing: uppercased field labels (same structure & separators).
SCAFFOLD_F3 = (
    "\n\n[USER QUESTION]\n{q}\n\n"
    "[THE START OF ASSISTANT A'S ANSWER]\n{a}\n[THE END OF ASSISTANT A'S ANSWER]\n\n"
    "[THE START OF ASSISTANT B'S ANSWER]\n{b}\n[THE END OF ASSISTANT B'S ANSWER]\n\n"
)

STAB_TPL = {
    "base":  INSTR_BASE + SCAFFOLD_BASE + TRAILER_BASE,
    "para1": INSTR_P1 + SCAFFOLD_BASE + TRAILER_BASE,
    "para2": INSTR_P2 + SCAFFOLD_BASE + TRAILER_BASE,
    "para3": INSTR_P3 + SCAFFOLD_BASE + TRAILER_BASE,
    "fmt1":  INSTR_BASE + SCAFFOLD_F1 + TRAILER_BASE,
    "fmt2":  INSTR_BASE + SCAFFOLD_F2 + TRAILER_BASE,
    "fmt3":  INSTR_BASE + SCAFFOLD_F3 + TRAILER_BASE,
}
PARA_VARIANTS = ["para1", "para2", "para3"]
FMT_VARIANTS = ["fmt1", "fmt2", "fmt3"]
N_SAMPLES = 5
TEMPERATURE = 0.7
B = 10000


def _tpl_sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- judge (argmax variant)
def cmd_judge(args):
    import torch, pandas as pd
    assert args.variant in STAB_TPL and args.variant != "base", args.variant
    df = pd.read_parquet(HERE / "pairs.parquet")
    if args.limit:
        df = df.head(args.limit)
    tok, model, dev = JE.load_model(args.model)
    vids = JE.verdict_token_ids(tok)
    tpl = STAB_TPL[args.variant]
    short = args.model.split("/")[-1]
    out_path = HERE / f"verdicts_stab_{args.variant}_{short}.csv"
    done = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = set(zip(prev["pair_id"], prev["order"]))
        print(f"resuming: {len(done)} rows done", flush=True)
    f = open(out_path, "a")
    if not done:
        f.write("pair_id,order,verdict,lp_A,lp_B,lp_C,prompt_tokens,ms\n")
    n_new, t0 = 0, time.time()
    for _, row in df.iterrows():
        for order in ("orig", "swap"):
            if (row["pair_id"], order) in done:
                continue
            a, b = (row["ans1"], row["ans2"]) if order == "orig" else (row["ans2"], row["ans1"])
            prompt = tpl.format(q=row["q"], a=a, b=b)
            text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True, tokenize=False)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            t1 = time.time()
            with torch.inference_mode():
                logits = model(ids).logits[0, -1].float()
            lp = torch.log_softmax(logits, dim=-1)
            s = {v: lp[vids[v]].item() for v in JE.VERDICTS}
            verdict = max(s, key=s.get)
            f.write(f"{row['pair_id']},{order},{verdict},{s['A']:.4f},{s['B']:.4f},{s['C']:.4f},"
                    f"{ids.shape[1]},{int((time.time()-t1)*1000)}\n")
            n_new += 1
            if n_new % 20 == 0:
                f.flush()
                el = time.time() - t0
                print(f"{args.variant}/{short}: {n_new} rows, {el/60:.1f} min, {el/n_new*1000:.0f} ms/row", flush=True)
    f.close()
    print(f"DONE {args.variant} {short}: +{n_new} -> {out_path}", flush=True)


# ---------------------------------------------------------------- sample (T=0.7 n=5 gen)
def _parse_letter(text):
    m = re.search(r"\b([ABC])\b", text)
    return m.group(1) if m else "?"


def cmd_sample(args):
    import torch, pandas as pd
    df = pd.read_parquet(HERE / "pairs.parquet")
    if args.limit:
        df = df.head(args.limit)
    tok, model, dev = JE.load_model(args.model)
    short = args.model.split("/")[-1]
    tpl = STAB_TPL["base"]
    out_path = HERE / f"samples_stab_{short}.csv"
    done = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = set(zip(prev["pair_id"], prev["order"]))
        print(f"resuming: {len(done)} rows done", flush=True)
    f = open(out_path, "a")
    cols = ["pair_id", "order", "greedy_gen"] + [f"s{k+1}" for k in range(N_SAMPLES)] + ["ms"]
    if not done:
        f.write(",".join(cols) + "\n")
    n_new, t0 = 0, time.time()
    for _, row in df.iterrows():
        for order in ("orig", "swap"):
            if (row["pair_id"], order) in done:
                continue
            a, b = (row["ans1"], row["ans2"]) if order == "orig" else (row["ans2"], row["ans1"])
            prompt = tpl.format(q=row["q"], a=a, b=b)
            text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True, tokenize=False)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            t1 = time.time()
            # greedy generation reference (verifies logit-argmax faithfulness under decoding)
            with torch.inference_mode():
                g = model.generate(ids, max_new_tokens=6, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            greedy_gen = _parse_letter(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))
            samples = []
            for k in range(N_SAMPLES):
                seed = int(hashlib.sha256(f"{row['pair_id']}|{order}|{k}".encode()).hexdigest()[:8], 16)
                torch.manual_seed(seed)
                with torch.inference_mode():
                    out = model.generate(ids, max_new_tokens=6, do_sample=True,
                                         temperature=TEMPERATURE, top_p=1.0, top_k=0,
                                         pad_token_id=tok.eos_token_id)
                samples.append(_parse_letter(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)))
            ms = int((time.time() - t1) * 1000)
            f.write(",".join([row["pair_id"], order, greedy_gen] + samples + [str(ms)]) + "\n")
            n_new += 1
            if n_new % 10 == 0:
                f.flush()
                el = time.time() - t0
                print(f"sample/{short}: {n_new} rows, {el/60:.1f} min, {el/n_new*1000:.0f} ms/row", flush=True)
    f.close()
    print(f"DONE sample {short}: +{n_new} -> {out_path}", flush=True)


# ---------------------------------------------------------------- analyze
def _majority(vs):
    """Plurality verdict among a list; unique top wins, else 'nomaj'."""
    c = Counter(vs)
    top, n = c.most_common(1)[0]
    if sum(1 for k, v in c.items() if v == n) > 1:
        return "nomaj"
    return top


def _boot_ci(mask, pair_ids, rng):
    """Pair-clustered bootstrap 95% CI for the mean of a boolean instance-mask.
    Resample pair_ids with replacement; each pair contributes all its instances."""
    import numpy as np
    import pandas as pd
    d = pd.DataFrame({"pid": pair_ids, "m": mask.astype(float)})
    groups = {p: g["m"].to_numpy() for p, g in d.groupby("pid")}
    uniq = list(groups)
    stats = np.empty(B)
    for i in range(B):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        vals = np.concatenate([groups[p] for p in samp])
        stats[i] = vals.mean()
    return [round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4)]


def _paired_delta_ci(mask_a, mask_b, pair_ids, rng):
    """Pair-clustered bootstrap CI for mean(a) - mean(b) on shared instances (aligned)."""
    import numpy as np, pandas as pd
    d = pd.DataFrame({"pid": pair_ids, "a": mask_a.astype(float), "b": mask_b.astype(float)})
    groups = {p: g for p, g in d.groupby("pid")}
    uniq = list(groups)
    stats = np.empty(B)
    for i in range(B):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        cat = pd.concat([groups[p] for p in samp])
        stats[i] = cat["a"].mean() - cat["b"].mean()
    return [round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4)]


def _load_instances(short, pairs_ids):
    """Assemble the per-instance (pair_id, order) verdict table for one model.
    Returns a DataFrame with base, para1..3, fmt1..3, samp_majority, samp_all_agree."""
    import pandas as pd
    base = pd.read_csv(HERE / f"verdicts_{short}.csv")[["pair_id", "order", "verdict"]]
    base = base.rename(columns={"verdict": "base"})
    m = base
    for var in PARA_VARIANTS + FMT_VARIANTS:
        v = pd.read_csv(HERE / f"verdicts_stab_{var}_{short}.csv")[["pair_id", "order", "verdict"]]
        m = m.merge(v.rename(columns={"verdict": var}), on=["pair_id", "order"], how="inner")
    s = pd.read_csv(HERE / f"samples_stab_{short}.csv")
    scols = [f"s{k+1}" for k in range(N_SAMPLES)]
    s["samp_majority"] = s[scols].apply(lambda r: _majority(list(r)), axis=1)
    s["samp_all_agree_greedygen"] = s.apply(lambda r: all(r[c] == r["greedy_gen"] for c in scols), axis=1)
    m = m.merge(s[["pair_id", "order", "greedy_gen", "samp_majority", "samp_all_agree_greedygen"]],
                on=["pair_id", "order"], how="inner")
    # restrict to the agreed instance set (pairs present everywhere)
    m = m[m["pair_id"].isin(pairs_ids)].reset_index(drop=True)
    return m


def cmd_analyze(args):
    import numpy as np, pandas as pd
    src = HERE / "pairs.parquet"
    if not src.exists():
        src = HERE / "pairs_public.parquet"
    pairs = pd.read_parquet(src)
    pair_ids_all = set(pairs["pair_id"])
    rng = np.random.default_rng(JE.SEED)

    models = []
    for cand in ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-0.5B-Instruct"]:
        if (HERE / f"samples_stab_{cand}.csv").exists() and \
           all((HERE / f"verdicts_stab_{v}_{cand}.csv").exists() for v in PARA_VARIANTS + FMT_VARIANTS):
            models.append(cand)

    per_model, frames = {}, {}
    for short in models:
        m = _load_instances(short, pair_ids_all)
        n = len(m)
        pid = m["pair_id"].to_numpy()
        # per-family instability masks (vs base)
        para_mask = ~((m["para1"] == m["base"]) & (m["para2"] == m["base"]) & (m["para3"] == m["base"]))
        fmt_mask = ~((m["fmt1"] == m["base"]) & (m["fmt2"] == m["base"]) & (m["fmt3"] == m["base"]))
        samp_mask = (m["samp_majority"] != m["base"])
        robust_mask = ~(para_mask | fmt_mask | samp_mask)
        # per-single-variant instability (granular)
        per_variant = {}
        for var in PARA_VARIANTS + FMT_VARIANTS:
            vm = (m[var] != m["base"])
            per_variant[var] = {"instability": round(float(vm.mean()), 4),
                                "ci95": _boot_ci(vm.to_numpy(), pid, rng)}
        fam = {}
        for name, mask in [("sampling", samp_mask), ("paraphrase", para_mask), ("formatting", fmt_mask)]:
            fam[name] = {"instability": round(float(mask.mean()), 4),
                         "ci95": _boot_ci(mask.to_numpy(), pid, rng)}
        robust = {"robust_core": round(float(robust_mask.mean()), 4),
                  "ci95": _boot_ci(robust_mask.to_numpy(), pid, rng)}
        # strict sampling variant: all 5 samples == greedy-generation reference
        strict_samp_mask = ~m["samp_all_agree_greedygen"]
        # greedy-generation vs logit-argmax base faithfulness (sanity)
        faith = float((m["greedy_gen"] == m["base"]).mean())
        ranking = sorted(fam, key=lambda k: fam[k]["instability"], reverse=True)
        per_model[short] = {
            "n_instances": int(n), "n_pairs": int(m["pair_id"].nunique()),
            "family_instability": fam,
            "family_ranking_by_damage": ranking,
            "robust_core": robust,
            "per_variant_instability": per_variant,
            "sampling_strict_all5_vs_greedygen": {
                "instability": round(float(strict_samp_mask.mean()), 4),
                "ci95": _boot_ci(strict_samp_mask.to_numpy(), pid, rng)},
            "greedygen_vs_logit_argmax_agreement": round(faith, 4),
        }
        frames[short] = (m, para_mask, fmt_mask, samp_mask, robust_mask)

    # scale trend: paired deltas 1.5B -> 7B (larger minus smaller: negative = improves with scale)
    scale = {}
    if "Qwen2.5-1.5B-Instruct" in frames and "Qwen2.5-7B-Instruct" in frames:
        big, small = "Qwen2.5-7B-Instruct", "Qwen2.5-1.5B-Instruct"
        mb, *_ = frames[big]; ms, *_ = frames[small]
        j = mb.merge(ms, on=["pair_id", "order"], suffixes=("_big", "_small"))
        pid = j["pair_id"].to_numpy()
        def fam_masks(suf):
            base = j[f"base_{suf}"]
            pm = ~((j[f"para1_{suf}"] == base) & (j[f"para2_{suf}"] == base) & (j[f"para3_{suf}"] == base))
            fm = ~((j[f"fmt1_{suf}"] == base) & (j[f"fmt2_{suf}"] == base) & (j[f"fmt3_{suf}"] == base))
            sm = (j[f"samp_majority_{suf}"] != base)
            rm = ~(pm | fm | sm)
            return {"sampling": sm, "paraphrase": pm, "formatting": fm, "robust_core": rm}
        big_m, small_m = fam_masks("big"), fam_masks("small")
        for k in ["sampling", "paraphrase", "formatting", "robust_core"]:
            scale[k] = {
                "1.5B": round(float(small_m[k].mean()), 4),
                "7B": round(float(big_m[k].mean()), 4),
                "delta_7B_minus_1.5B": round(float(big_m[k].mean() - small_m[k].mean()), 4),
                "delta_ci95": _paired_delta_ci(big_m[k].to_numpy(), small_m[k].to_numpy(), pid, rng),
            }

    out = {
        "per_model": per_model,
        "scale_trend": scale,
        "protocol": {
            "instance_unit": "(pair, presentation order); both orders included",
            "n_pairs_target": int(len(pairs)),
            "base_greedy_source": "verdicts_<model>.csv (argmax next-token logprob A/B/C), reused from finding cf4e6c02",
            "sampling": f"actual generation, T={TEMPERATURE}, n={N_SAMPLES}, majority verdict (unique plurality; ties->'nomaj' counted unstable)",
            "family_instability_def": "fraction of instances whose verdict is NOT identical to base across the family's variants (paraphrase: base vs para1/2/3; formatting: base vs fmt1/2/3; sampling: n=5 majority vs base greedy)",
            "robust_core_def": "fraction of instances invariant under ALL perturbations simultaneously (paraphrase-stable AND formatting-stable AND sampling-stable)",
            "bootstrap_reps": B, "seed": JE.SEED, "clustering": "pair-clustered bootstrap (resample pairs; both orders move together)",
            "template_sha256": {k: _tpl_sha(v) for k, v in STAB_TPL.items()},
        },
    }
    (HERE / "results_stab.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- selftest
def cmd_selftest(args):
    # base template reconstruction must match the pinned pipeline template exactly
    assert STAB_TPL["base"] == JE.JUDGE_TEMPLATE, "base template drift vs judge_eval.JUDGE_TEMPLATE"
    # paraphrases: scaffold + trailer identical to base, only the instruction differs
    for v in PARA_VARIANTS:
        assert SCAFFOLD_BASE in STAB_TPL[v] and STAB_TPL[v].endswith(TRAILER_BASE), v
        assert not STAB_TPL[v].startswith(INSTR_BASE), v
        assert STAB_TPL[v] != STAB_TPL["base"], v
    # formats: instruction + trailer identical, only the scaffold differs
    for v in FMT_VARIANTS:
        assert STAB_TPL[v].startswith(INSTR_BASE) and STAB_TPL[v].endswith(TRAILER_BASE), v
        assert STAB_TPL[v] != STAB_TPL["base"], v
    # all seven templates distinct
    assert len(set(STAB_TPL.values())) == 7
    # majority logic
    assert _majority(["A", "A", "B", "A", "C"]) == "A"
    assert _majority(["A", "A", "B", "B", "C"]) == "nomaj"
    assert _majority(["C", "C", "C", "C", "C"]) == "C"
    # instability logic sanity: base=A, all variants A -> stable
    import pandas as pd
    row_stable = {"base": "A", "para1": "A", "para2": "A", "para3": "A",
                  "fmt1": "A", "fmt2": "A", "fmt3": "A", "samp_majority": "A"}
    pm = not (row_stable["para1"] == row_stable["base"] and row_stable["para2"] == row_stable["base"]
              and row_stable["para3"] == row_stable["base"])
    assert pm is False
    print("selftest: all assertions pass")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge"); j.add_argument("--model", required=True)
    j.add_argument("--variant", required=True, choices=PARA_VARIANTS + FMT_VARIANTS)
    j.add_argument("--limit", type=int, default=0)
    sp = sub.add_parser("sample"); sp.add_argument("--model", required=True); sp.add_argument("--limit", type=int, default=0)
    sub.add_parser("analyze")
    sub.add_parser("selftest")
    args = ap.parse_args()
    {"judge": cmd_judge, "sample": cmd_sample, "analyze": cmd_analyze, "selftest": cmd_selftest}[args.cmd](args)
