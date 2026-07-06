#!/usr/bin/env python3
"""Order-flip position bias vs model scale — Qwen2.5-Instruct ladder on MT-Bench pairs.

Subcommands:
  prep      build the fixed pair set (turn-1 unique pairs, human majority labels,
            length-filtered, stratified sample) -> pairs.parquet
  judge     run one judge model over all pairs x both orders, deterministic
            logit-based verdict extraction -> verdicts_<model>.csv (incremental)
  genchk    free-generation robustness check on a subsample -> genchk_<model>.csv
  analyze   compute flip/primacy/agreement + bootstrap CIs -> results.json/md
  selftest  unit-test mirror/flip/majority logic (no downloads, no model)

Protocol is pinned: prompt template below, verdict tokens A/B/C, greedy/argmax over
next-token logprobs at the first assistant position. Every pair judged in both orders.
"""
import argparse, json, os, sys, time, hashlib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("HF_HOME", str(Path.home() / "research/scinet_seeding/track-c/cache/hf")))
os.environ.setdefault("HF_HOME", str(CACHE))

N_PAIRS = 350
MAX_PROMPT_TOKENS = 3072
SEED = 0
GEN_CHECK_N = 60
TOKENIZER_REF = "Qwen/Qwen2.5-0.5B-Instruct"  # tokenizer identical across the ladder

JUDGE_TEMPLATE = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI "
    "assistants to the user question displayed below. You should choose the assistant that answers "
    "the user's question better. Your evaluation should consider factors such as the helpfulness, "
    "relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any "
    "position biases and ensure that the order in which the responses were presented does not "
    "influence your decision. Do not allow the length of the responses to influence your "
    "evaluation. Do not favor certain names of the assistants. Be as objective as possible.\n\n"
    "[User Question]\n{q}\n\n"
    "[The Start of Assistant A's Answer]\n{a}\n[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n{b}\n[The End of Assistant B's Answer]\n\n"
    "Which assistant's answer is better? Respond with exactly one letter: \"A\" if assistant A's "
    "answer is better, \"B\" if assistant B's answer is better, or \"C\" for a tie."
)

VERDICTS = ["A", "B", "C"]


def mirror(v: str) -> str:
    """Verdict in swapped presentation, mapped back to original slots."""
    return {"A": "B", "B": "A", "C": "C"}[v]


def is_flip(v_orig: str, v_swap: str) -> bool:
    """Order-sensitive: swapped verdict is not the mirror of the original."""
    return mirror(v_swap) != v_orig


def primacy_class(v_orig: str, v_swap: str) -> str:
    """Among flipped pairs: 'primacy' = picks slot-1 both times; 'recency' = slot-2 both times."""
    if v_orig == "A" and v_swap == "A":
        return "primacy"
    if v_orig == "B" and v_swap == "B":
        return "recency"
    return "other"


def canon_label(winner: str, m_a: str, m_b: str) -> str:
    if winner == "model_a":
        return m_a
    if winner == "model_b":
        return m_b
    return "tie"


def majority(labels):
    c = Counter(labels)
    top, n = c.most_common(1)[0]
    ties = [k for k, v in c.items() if v == n]
    if len(ties) > 1:
        return "no-majority", n / len(labels)
    return top, n / len(labels)


# ---------------------------------------------------------------- prep
def cmd_prep(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import pandas as pd

    ds = load_dataset("lmsys/mt_bench_human_judgments", split="human")
    rows = {}
    for r in ds:
        if r["turn"] != 1:
            continue
        ca, cb = r["conversation_a"], r["conversation_b"]
        q = ca[0]["content"]
        if cb[0]["content"] != q:
            continue
        a_ans, b_ans = ca[1]["content"], cb[1]["content"]
        m_a, m_b = r["model_a"], r["model_b"]
        m1, m2 = sorted([m_a, m_b])
        key = (r["question_id"], m1, m2)
        lab = canon_label(r["winner"], m_a, m_b)
        rec = rows.setdefault(key, {"question_id": r["question_id"], "m1": m1, "m2": m2,
                                    "labels": [], "q": q, "ans": {}})
        rec["labels"].append(lab)
        rec["ans"][m_a] = a_ans
        rec["ans"][m_b] = b_ans

    tok = AutoTokenizer.from_pretrained(TOKENIZER_REF)
    out = []
    for key in sorted(rows):
        rec = rows[key]
        maj, frac = majority(rec["labels"])
        a1, a2 = rec["ans"][rec["m1"]], rec["ans"][rec["m2"]]
        prompt = JUDGE_TEMPLATE.format(q=rec["q"], a=a1, b=a2)
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
        if len(ids) > MAX_PROMPT_TOKENS:
            continue
        hlab = "tie" if maj == "tie" else ("m1" if maj == rec["m1"] else ("m2" if maj == rec["m2"] else "no-majority"))
        out.append({"question_id": rec["question_id"], "m1": rec["m1"], "m2": rec["m2"],
                    "q": rec["q"], "ans1": a1, "ans2": a2,
                    "human_majority": hlab, "n_judgments": len(rec["labels"]),
                    "majority_frac": round(frac, 3), "prompt_tokens": len(ids),
                    "len1_tok": len(tok.encode(a1)), "len2_tok": len(tok.encode(a2))})
    df = pd.DataFrame(out)
    n_total = len(df)

    # stratified deterministic sample by human_majority
    df = df.sort_values(["question_id", "m1", "m2"]).reset_index(drop=True)
    parts = []
    rng_order = df.sample(frac=1.0, random_state=SEED)  # fixed shuffle
    for lab, grp in rng_order.groupby("human_majority"):
        share = int(round(N_PAIRS * len(grp) / n_total))
        parts.append(grp.head(share))
    import pandas as pd  # noqa
    samp = pd.concat(parts).head(N_PAIRS).sort_values(["question_id", "m1", "m2"]).reset_index(drop=True)
    samp.insert(0, "pair_id", [f"p{i:04d}" for i in range(len(samp))])
    samp.to_parquet(HERE / "pairs.parquet")
    meta = {"n_unique_turn1_pairs": n_total, "n_sampled": len(samp),
            "max_prompt_tokens": MAX_PROMPT_TOKENS, "seed": SEED,
            "label_dist_sample": samp["human_majority"].value_counts().to_dict(),
            "label_dist_all": df["human_majority"].value_counts().to_dict(),
            "dataset": "lmsys/mt_bench_human_judgments split=human turn=1",
            "template_sha256": hashlib.sha256(JUDGE_TEMPLATE.encode()).hexdigest()[:16]}
    (HERE / "pairs_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


# ---------------------------------------------------------------- judge
def load_model(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(dev)
    model.eval()
    return tok, model, dev


def verdict_token_ids(tok):
    ids = {}
    for v in VERDICTS:
        enc = tok.encode(v, add_special_tokens=False)
        assert len(enc) == 1, f"verdict {v!r} is not a single token: {enc}"
        ids[v] = enc[0]
    return ids


def cmd_judge(args):
    import torch, pandas as pd
    df = pd.read_parquet(HERE / "pairs.parquet")
    tok, model, dev = load_model(args.model)
    vids = verdict_token_ids(tok)
    short = args.model.split("/")[-1]
    out_path = HERE / f"verdicts_{short}.csv"
    done = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = set(zip(prev["pair_id"], prev["order"]))
        print(f"resuming: {len(done)} rows already done", flush=True)
    f = open(out_path, "a")
    if not done:
        f.write("pair_id,order,verdict,lp_A,lp_B,lp_C,prompt_tokens,ms\n")
    t_start = time.time()
    n_new = 0
    for _, row in df.iterrows():
        for order in ("orig", "swap"):
            if (row["pair_id"], order) in done:
                continue
            a, b = (row["ans1"], row["ans2"]) if order == "orig" else (row["ans2"], row["ans1"])
            prompt = JUDGE_TEMPLATE.format(q=row["q"], a=a, b=b)
            text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            add_generation_prompt=True, tokenize=False)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            t0 = time.time()
            with torch.inference_mode():
                logits = model(ids).logits[0, -1].float()
            lp = torch.log_softmax(logits, dim=-1)
            scores = {v: lp[vids[v]].item() for v in VERDICTS}
            verdict = max(scores, key=scores.get)
            ms = int((time.time() - t0) * 1000)
            f.write(f"{row['pair_id']},{order},{verdict},{scores['A']:.4f},{scores['B']:.4f},"
                    f"{scores['C']:.4f},{ids.shape[1]},{ms}\n")
            n_new += 1
            if n_new % 20 == 0:
                f.flush()
                el = time.time() - t_start
                print(f"{short}: {n_new} new rows, {el/60:.1f} min elapsed, {el/n_new*1000:.0f} ms/row", flush=True)
    f.close()
    print(f"DONE {short}: +{n_new} rows -> {out_path}", flush=True)


# ---------------------------------------------------------------- genchk
def cmd_genchk(args):
    import torch, pandas as pd, re
    df = pd.read_parquet(HERE / "pairs.parquet").head(GEN_CHECK_N)
    tok, model, dev = load_model(args.model)
    short = args.model.split("/")[-1]
    out_path = HERE / f"genchk_{short}.csv"
    f = open(out_path, "w")
    f.write("pair_id,order,gen_text,gen_verdict\n")
    for _, row in df.iterrows():
        for order in ("orig", "swap"):
            a, b = (row["ans1"], row["ans2"]) if order == "orig" else (row["ans2"], row["ans1"])
            prompt = JUDGE_TEMPLATE.format(q=row["q"], a=a, b=b)
            text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            add_generation_prompt=True, tokenize=False)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            with torch.inference_mode():
                out = model.generate(ids, max_new_tokens=8, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
            m = re.search(r"\b([ABC])\b", text)
            gv = m.group(1) if m else "unparsed"
            safe = text.replace('"', "'").replace("\n", " ")[:40]
            f.write(f'{row["pair_id"]},{order},"{safe}",{gv}\n')
    f.close()
    print(f"DONE genchk {short} -> {out_path}", flush=True)


# ---------------------------------------------------------------- analyze
def metrics_for(df):
    """df: one row per pair with v_orig, v_swap columns."""
    import numpy as np
    flips = df.apply(lambda r: is_flip(r["v_orig"], r["v_swap"]), axis=1)
    flip_rate = flips.mean()
    fl = df[flips]
    if len(fl):
        pc = fl.apply(lambda r: primacy_class(r["v_orig"], r["v_swap"]), axis=1)
        primacy = (pc == "primacy").mean()
        recency = (pc == "recency").mean()
    else:
        primacy = recency = float("nan")
    # order-debiased verdict + agreement with human majority (non-tie majorities only)
    def debiased(r):
        if not is_flip(r["v_orig"], r["v_swap"]):
            return r["v_orig"]
        return "inconsistent"
    df = df.assign(deb=df.apply(debiased, axis=1))
    lab = df[df["human_majority"].isin(["m1", "m2"])]
    lab_c = lab[lab["deb"].isin(["A", "B"])]
    agree = ((lab_c["deb"] == "A") == (lab_c["human_majority"] == "m1")).mean() if len(lab_c) else float("nan")
    pick1 = ((df["v_orig"] == "A").sum() + (df["v_swap"] == "A").sum()) / (2 * len(df))
    return {"n": len(df), "flip_rate": flip_rate, "primacy_share": primacy,
            "recency_share": recency, "p_pick_slot1": pick1,
            "consistent_frac": (df["deb"] != "inconsistent").mean(),
            "tie_frac_deb": (df["deb"] == "C").mean(),
            "human_agree_on_consistent": agree, "n_human_labeled_consistent": len(lab_c)}


def cmd_mkpublic(args):
    """pairs_public.parquet = pair set minus the question/answer texts (re-derivable via
    `prep` from the CC-BY-4.0 lmsys/mt_bench_human_judgments dataset). Enables the
    zero-download smoke repro: analyze runs from this + committed verdict CSVs."""
    import pandas as pd
    df = pd.read_parquet(HERE / "pairs.parquet")
    df.drop(columns=["q", "ans1", "ans2"]).to_parquet(HERE / "pairs_public.parquet")
    print(f"pairs_public.parquet: {len(df)} rows, no texts")


def cmd_analyze(args):
    import numpy as np, pandas as pd
    src = HERE / "pairs.parquet"
    if not src.exists():
        src = HERE / "pairs_public.parquet"  # zero-download smoke path
    pairs = pd.read_parquet(src)
    models = sorted(HERE.glob("verdicts_*.csv"))
    results, per_model_df = {}, {}
    rng = np.random.default_rng(SEED)
    B = 10000
    for path in models:
        name = path.stem.replace("verdicts_", "")
        v = pd.read_csv(path)
        piv = v.pivot(index="pair_id", columns="order", values="verdict").dropna()
        piv.columns = [f"v_{c}" for c in piv.columns]
        m = piv.merge(pairs[["pair_id", "human_majority", "len1_tok", "len2_tok"]],
                      on="pair_id", how="inner")
        if len(m) < len(pairs):
            print(f"warning: {name} has {len(m)}/{len(pairs)} complete pairs")
        per_model_df[name] = m
        base = metrics_for(m)
        # bootstrap CIs on flip_rate and p_pick_slot1 (both resampled over pairs)
        fl = m.apply(lambda r: is_flip(r["v_orig"], r["v_swap"]), axis=1).to_numpy()
        bs = rng.choice(fl, size=(B, len(fl)), replace=True).mean(axis=1)
        base["flip_rate_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        p1 = ((m["v_orig"] == "A").astype(float) + (m["v_swap"] == "A").astype(float)).to_numpy() / 2.0
        bs1 = rng.choice(p1, size=(B, len(p1)), replace=True).mean(axis=1)
        base["p_pick_slot1_ci95"] = [float(np.percentile(bs1, 2.5)), float(np.percentile(bs1, 97.5))]
        results[name] = {k: (round(float(x), 4) if isinstance(x, (int, float)) else x)
                         for k, x in base.items()}
    # pairwise scale deltas (bootstrap, paired over shared pair_ids)
    names = sorted(per_model_df)
    deltas = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = per_model_df[names[i]], per_model_df[names[j]]
            mm = a[["pair_id", "v_orig", "v_swap"]].merge(
                b[["pair_id", "v_orig", "v_swap"]], on="pair_id", suffixes=("_a", "_b"))
            fa = mm.apply(lambda r: is_flip(r["v_orig_a"], r["v_swap_a"]), axis=1).to_numpy()
            fb = mm.apply(lambda r: is_flip(r["v_orig_b"], r["v_swap_b"]), axis=1).to_numpy()
            d = fa.astype(float) - fb.astype(float)
            bs = rng.choice(d, size=(B, len(d)), replace=True).mean(axis=1)
            deltas[f"{names[i]} - {names[j]}"] = {
                "delta_flip": round(float(d.mean()), 4),
                "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
    out = {"per_model": results, "pairwise_flip_deltas": deltas,
           "protocol": {"n_pairs": int(len(pairs)), "template_sha256":
                        hashlib.sha256(JUDGE_TEMPLATE.encode()).hexdigest()[:16],
                        "verdict_extraction": "argmax over next-token logprobs of tokens A/B/C",
                        "bootstrap_reps": B, "seed": SEED}}
    (HERE / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    # genchk agreement
    for path in sorted(HERE.glob("genchk_*.csv")):
        name = path.stem.replace("genchk_", "")
        g = pd.read_csv(path)
        v = pd.read_csv(HERE / f"verdicts_{name}.csv")
        mm = g.merge(v, on=["pair_id", "order"])
        ok = mm[mm["gen_verdict"].isin(["A", "B", "C"])]
        print(f"genchk {name}: parsed {len(ok)}/{len(mm)}, agreement with logit verdict: "
              f"{(ok['gen_verdict'] == ok['verdict']).mean():.3f}")


# ---------------------------------------------------------------- selftest
def cmd_selftest(args):
    assert mirror("A") == "B" and mirror("B") == "A" and mirror("C") == "C"
    # consistent: verdict A then (swapped) B -> mirror(B)=A == orig -> no flip
    assert not is_flip("A", "B") and not is_flip("B", "A") and not is_flip("C", "C")
    # flips
    assert is_flip("A", "A") and is_flip("B", "B") and is_flip("A", "C") and is_flip("C", "B")
    assert primacy_class("A", "A") == "primacy" and primacy_class("B", "B") == "recency"
    assert primacy_class("A", "C") == "other"
    assert majority(["x", "x", "tie"]) == ("x", 2 / 3)
    assert majority(["x", "tie"])[0] == "no-majority"
    assert canon_label("model_a", "vicuna", "gpt4") == "vicuna"
    assert canon_label("tie", "a", "b") == "tie"
    print("selftest: all assertions pass")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    j = sub.add_parser("judge"); j.add_argument("--model", required=True)
    g = sub.add_parser("genchk"); g.add_argument("--model", required=True)
    sub.add_parser("analyze")
    sub.add_parser("selftest")
    sub.add_parser("mkpublic")
    args = ap.parse_args()
    {"prep": cmd_prep, "judge": cmd_judge, "genchk": cmd_genchk,
     "analyze": cmd_analyze, "selftest": cmd_selftest, "mkpublic": cmd_mkpublic}[args.cmd](args)
