#!/usr/bin/env python3
"""Verbosity bias on human-preferred-shorter MT-Bench pairs — instruction ablation.

Arms:
  default  — JUDGE_TEMPLATE from judge_eval.py (Zheng-style; includes its standard
             'do not allow length to influence' sentence). Verdicts REUSED from
             verdicts_<model>.csv (finding cf4e6c02) — no new compute.
  noinstr  — same template with the anti-position/length/name sentences REMOVED.
  antiverb — default + explicit strengthened anti-verbosity instruction.

Subcommands:
  judge --model M --arm {noinstr|antiverb}   -> verdicts_vb_<arm>_<model>.csv
  analyze                                    -> results_vb.json
  selftest                                   (no model, no downloads)
"""
import argparse, json, os, time, hashlib
from pathlib import Path

import judge_eval as JE

HERE = Path(__file__).resolve().parent

REMOVED = (
    "Avoid any position biases and ensure that the order in which the responses were presented does not "
    "influence your decision. Do not allow the length of the responses to influence your "
    "evaluation. Do not favor certain names of the assistants. "
)
ANTIVERB_ADD = (
    "IMPORTANT: The longer response is not necessarily the better one; length itself must never "
    "count in favor of a response. Judge only quality, correctness, and relevance to the question. "
)

TPL = {
    "default": JE.JUDGE_TEMPLATE,
    "noinstr": JE.JUDGE_TEMPLATE.replace(REMOVED, ""),
    "antiverb": JE.JUDGE_TEMPLATE.replace(
        "Be as objective as possible.", "Be as objective as possible. " + ANTIVERB_ADD.rstrip()),
}
assert TPL["noinstr"] != TPL["default"], "noinstr removal failed"
assert TPL["antiverb"] != TPL["default"], "antiverb insertion failed"

SUBSET = "subset_vb.parquet"
B = 10000


def prefers_longer(verdict, order, len1, len2):
    """Map a slot verdict (A/B/C) for a given presentation order to longer/shorter/tie."""
    if verdict == "C":
        return "tie"
    slot1_is_ans1 = (order == "orig")
    picked_ans1 = (verdict == "A") == slot1_is_ans1
    picked_len = len1 if picked_ans1 else len2
    other_len = len2 if picked_ans1 else len1
    return "longer" if picked_len > other_len else "shorter"


def cmd_judge(args):
    import torch, pandas as pd
    df = pd.read_parquet(HERE / SUBSET)
    tok, model, dev = JE.load_model(args.model)
    vids = JE.verdict_token_ids(tok)
    tpl = TPL[args.arm]
    short = args.model.split("/")[-1]
    out_path = HERE / f"verdicts_vb_{args.arm}_{short}.csv"
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
                print(f"{args.arm}/{short}: {n_new} rows, {el/60:.1f} min, {el/n_new*1000:.0f} ms/row", flush=True)
    f.close()
    print(f"DONE {args.arm} {short}: +{n_new} -> {out_path}", flush=True)


def logit_fit(x, y):
    """2-param logistic MLE via IRLS (plain numpy). Returns (intercept, slope)."""
    import numpy as np
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(60):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (y - p)
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def arm_frame(arm, model_short, pairs):
    """Rows: one per (pair, order) with outcome in {longer, shorter, tie} + log length ratio."""
    import pandas as pd, numpy as np
    src = (HERE / f"verdicts_{model_short}.csv") if arm == "default" \
        else (HERE / f"verdicts_vb_{arm}_{model_short}.csv")
    v = pd.read_csv(src)
    m = v.merge(pairs[["pair_id", "len1_tok", "len2_tok"]], on="pair_id", how="inner")
    m["choice"] = [prefers_longer(r.verdict, r.order, r.len1_tok, r.len2_tok)
                   for r in m.itertuples()]
    m["log_ratio"] = np.log(np.maximum(m.len1_tok, m.len2_tok) / np.minimum(m.len1_tok, m.len2_tok))
    return m


def arm_metrics(m, rng):
    import numpy as np
    p_longer = float((m["choice"] == "longer").mean())
    tie_frac = float((m["choice"] == "tie").mean())
    # pair-clustered bootstrap for P(longer)
    by_pair = m.groupby("pair_id")["choice"].apply(lambda c: (c == "longer").mean())
    arr = by_pair.to_numpy()
    bs = rng.choice(arr, size=(B, len(arr)), replace=True).mean(axis=1)
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    # logistic slope on decided judgments, cluster bootstrap over pairs
    d = m[m["choice"] != "tie"]
    slope, slope_ci = float("nan"), [float("nan")] * 2
    if len(d) >= 20 and d["choice"].nunique() == 2:
        x, y = d["log_ratio"].to_numpy(), (d["choice"] == "longer").to_numpy(float)
        slope = float(logit_fit(x, y)[1])
        pids = d["pair_id"].unique()
        groups = {p: g for p, g in d.groupby("pair_id")}
        sl = []
        for _ in range(2000):
            samp = rng.choice(pids, size=len(pids), replace=True)
            import pandas as pd
            bb = pd.concat([groups[p] for p in samp])
            if bb["choice"].nunique() < 2:
                continue
            sl.append(logit_fit(bb["log_ratio"].to_numpy(), (bb["choice"] == "longer").to_numpy(float))[1])
        if sl:
            slope_ci = [float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))]
    return {"n_judgments": len(m), "p_longer": round(p_longer, 4), "p_longer_ci95": ci,
            "tie_frac": round(tie_frac, 4), "slope_logratio": round(slope, 4) if slope == slope else None,
            "slope_ci95": slope_ci, "n_decided": int(len(m[m['choice'] != 'tie']))}


def cmd_analyze(args):
    import numpy as np, pandas as pd
    src = HERE / SUBSET
    if not src.exists():
        src = HERE / "subset_vb_public.parquet"
    pairs = pd.read_parquet(src)
    rng = np.random.default_rng(JE.SEED)
    out = {"per_arm": {}, "ablation_deltas": {}, "protocol": {
        "subset": "human-majority-preferred-shorter (63) + human-tie (62) pairs, unequal token lengths, n=125",
        "arms": {k: hashlib.sha256(v.encode()).hexdigest()[:16] for k, v in TPL.items()},
        "bootstrap_reps": B, "seed": JE.SEED}}
    frames = {}
    for m in JE.__dict__.get("LADDER", None) or ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct",
                                                 "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"]:
        for arm in ("default", "noinstr", "antiverb"):
            src_csv = (HERE / f"verdicts_{m}.csv") if arm == "default" else (HERE / f"verdicts_vb_{arm}_{m}.csv")
            if not src_csv.exists():
                continue
            fr = arm_frame(arm, m, pairs)
            frames[(m, arm)] = fr
            out["per_arm"][f"{arm}/{m}"] = arm_metrics(fr, rng)
    # paired deltas per scale: antiverb-noinstr and default-noinstr
    for m in {k[0] for k in frames}:
        for a, b in (("antiverb", "noinstr"), ("default", "noinstr")):
            if (m, a) in frames and (m, b) in frames:
                fa = frames[(m, a)].groupby("pair_id")["choice"].apply(lambda c: (c == "longer").mean())
                fb = frames[(m, b)].groupby("pair_id")["choice"].apply(lambda c: (c == "longer").mean())
                j = fa.to_frame("pa").join(fb.to_frame("pb"), how="inner")
                d = (j["pa"] - j["pb"]).to_numpy()
                bs = rng.choice(d, size=(B, len(d)), replace=True).mean(axis=1)
                out["ablation_deltas"][f"{a}-{b}/{m}"] = {
                    "delta_p_longer": round(float(d.mean()), 4),
                    "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                    "n_pairs": int(len(j))}
    (HERE / "results_vb.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def cmd_selftest(args):
    assert prefers_longer("A", "orig", 10, 100) == "shorter"   # slot1=ans1 shorter, picked A
    assert prefers_longer("A", "swap", 10, 100) == "longer"    # slot1=ans2 longer
    assert prefers_longer("B", "orig", 10, 100) == "longer"
    assert prefers_longer("B", "swap", 10, 100) == "shorter"
    assert prefers_longer("C", "orig", 10, 100) == "tie"
    import numpy as np
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 4000)
    p = 1 / (1 + np.exp(-(0.5 + 2.0 * x)))
    y = (rng.random(4000) < p).astype(float)
    w = logit_fit(x, y)
    assert abs(w[1] - 2.0) < 0.25 and abs(w[0] - 0.5) < 0.15, w
    assert "Avoid any position biases" not in TPL["noinstr"]
    assert "length itself must never count" in TPL["antiverb"]
    print("selftest: all assertions pass")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge"); j.add_argument("--model", required=True); j.add_argument("--arm", required=True, choices=["noinstr", "antiverb"])
    sub.add_parser("analyze")
    sub.add_parser("selftest")
    args = ap.parse_args()
    {"judge": cmd_judge, "analyze": cmd_analyze, "selftest": cmd_selftest}[args.cmd](args)
