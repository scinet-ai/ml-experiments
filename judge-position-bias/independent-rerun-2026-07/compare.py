#!/usr/bin/env python3
"""Row-by-row comparison of my independently-regenerated verdicts vs the committed
verdicts for finding cf4e6c02, plus independent recomputation of the headline metrics
with bootstrap CIs from MY verdicts.

Reimplements the mirror/flip/primacy/agreement logic from the published protocol
description (README) rather than importing the finding's judge_eval.py, so the
analysis layer is independent too.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE / "repo-read" / "judge-position-bias"
SEED, B = 0, 10000

# --- our own copy of the order/flip/primacy semantics (verified below) ---
def mirror(v):        return {"A": "B", "B": "A", "C": "C"}[v]
def is_flip(vo, vs):  return mirror(vs) != vo
def is_primacy(vo, vs): return vo == "A" and vs == "A"
def is_recency(vo, vs): return vo == "B" and vs == "B"
# sanity: consistent = verdict then its mirror; flip = same letter twice (e.g. AA)
assert not is_flip("A", "B") and not is_flip("C", "C")
assert is_flip("A", "A") and is_flip("B", "B")
assert is_primacy("A", "A") and is_recency("B", "B")


def per_pair(vdf, pairs):
    piv = vdf.pivot(index="pair_id", columns="order", values="verdict").dropna()
    piv.columns = [f"v_{c}" for c in piv.columns]
    return piv.merge(pairs[["pair_id", "human_majority"]], on="pair_id", how="inner")


def metrics(m):
    vo, vs = m["v_orig"].to_numpy(), m["v_swap"].to_numpy()
    flip = np.array([is_flip(a, b) for a, b in zip(vo, vs)])
    prim = np.array([is_primacy(a, b) for a, b in zip(vo, vs)])
    rec = np.array([is_recency(a, b) for a, b in zip(vo, vs)])
    pick1 = ((vo == "A").astype(float) + (vs == "A").astype(float)) / 2.0
    consistent = ~flip
    deb = np.where(consistent, vo, "inconsistent")
    hm = m["human_majority"].to_numpy()
    lab_mask = np.isin(hm, ["m1", "m2"]) & np.isin(deb, ["A", "B"])
    if lab_mask.sum():
        agree = np.mean((deb[lab_mask] == "A") == (hm[lab_mask] == "m1"))
    else:
        agree = float("nan")
    nflip = flip.sum()
    return {
        "n": int(len(m)),
        "flip_rate": float(flip.mean()),
        "primacy_share": float(prim.sum() / nflip) if nflip else float("nan"),
        "recency_share": float(rec.sum() / nflip) if nflip else float("nan"),
        "p_pick_slot1": float(pick1.mean()),
        "consistent_frac": float(consistent.mean()),
        "human_agree_on_consistent": float(agree),
        "n_human_labeled_consistent": int(lab_mask.sum()),
        "_flip_vec": flip, "_pick1_vec": pick1,
    }


def boot_ci(vec, rng):
    bs = rng.choice(vec, size=(B, len(vec)), replace=True).mean(axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="short name e.g. Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()
    short = args.model
    mine_p = HERE / f"verdicts_mine_{short}.csv"
    comm_p = REPO / f"verdicts_{short}.csv"
    pairs = pd.read_parquet(HERE / "pairs_reconstructed.parquet")

    mine = pd.read_csv(mine_p)
    comm = pd.read_csv(comm_p)
    print(f"=== {short}: mine={len(mine)} rows, committed={len(comm)} rows ===")

    # ---- row-by-row verdict + logprob comparison ----
    j = mine.merge(comm, on=["pair_id", "order"], suffixes=("_mine", "_comm"))
    assert len(j) == len(comm), f"join size {len(j)} != {len(comm)}"
    verdict_agree = float((j["verdict_mine"] == j["verdict_comm"]).mean())
    n_disagree = int((j["verdict_mine"] != j["verdict_comm"]).mean() * len(j))
    dlp = {v: (j[f"lp_{v}_mine"] - j[f"lp_{v}_comm"]).abs() for v in "ABC"}
    all_dlp = pd.concat(list(dlp.values()))
    # margin at disagreements: how close were the top-2 committed logprobs?
    dis = j[j["verdict_mine"] != j["verdict_comm"]].copy()
    def top2margin(r):
        s = sorted([r["lp_A_comm"], r["lp_B_comm"], r["lp_C_comm"]], reverse=True)
        return s[0] - s[1]
    dis_margins = dis.apply(top2margin, axis=1).tolist() if len(dis) else []
    prompt_tok_match = float((j["prompt_tokens_mine"] == j["prompt_tokens_comm"]).mean())

    cmp = {
        "n_rows": int(len(j)),
        "verdict_agreement": round(verdict_agree, 4),
        "n_verdict_disagreements": n_disagree,
        "prompt_tokens_match_frac": round(prompt_tok_match, 4),
        "abs_logprob_delta": {
            "mean": round(float(all_dlp.mean()), 5),
            "median": round(float(all_dlp.median()), 5),
            "p95": round(float(all_dlp.quantile(0.95)), 5),
            "max": round(float(all_dlp.max()), 5),
        },
        "disagreement_committed_top2_margins": [round(x, 4) for x in sorted(dis_margins)],
    }

    # ---- recompute headline metrics from MINE and from COMMITTED ----
    rng = np.random.default_rng(SEED)
    mm_mine = per_pair(mine, pairs)
    mm_comm = per_pair(comm, pairs)
    met_mine = metrics(mm_mine)
    met_comm = metrics(mm_comm)
    rng2 = np.random.default_rng(SEED)
    met_mine["flip_rate_ci95"] = boot_ci(met_mine.pop("_flip_vec"), rng2)
    met_mine["p_pick_slot1_ci95"] = boot_ci(met_mine.pop("_pick1_vec"), rng2)
    met_comm.pop("_flip_vec"); met_comm.pop("_pick1_vec")

    # committed published numbers (from results.json) for a 3-way check
    pub = json.load(open(REPO / "results.json"))["per_model"][short]

    out = {
        "model": short,
        "row_comparison": cmp,
        "metrics_mine": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in met_mine.items()},
        "metrics_recomputed_from_committed_csv": {k: (round(v, 4) if isinstance(v, float) else v)
                                                  for k, v in met_comm.items()},
        "metrics_published_results_json": {k: pub[k] for k in
            ["flip_rate", "primacy_share", "recency_share", "p_pick_slot1",
             "consistent_frac", "human_agree_on_consistent", "n_human_labeled_consistent",
             "flip_rate_ci95", "p_pick_slot1_ci95"] if k in pub},
    }
    (HERE / f"comparison_{short}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
