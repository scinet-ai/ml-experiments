#!/usr/bin/env python3
"""INDEPENDENT inference-layer re-run of finding cf4e6c02's pairwise-judge verdicts.

This is a from-scratch runner: it loads the Qwen2.5-Instruct judge ourselves and
extracts a verdict from the model's own next-token logits. It does NOT import or
call the finding's judge_eval.py. What it reuses (verified against the published
protocol): the pinned Zheng-style prompt template string, the {A,B,C} verdict set,
and the 350-pair selection (via pairs_reconstructed.parquet, whose texts we
re-derived from the source dataset and checksum-matched). Everything about the
inference -- model loading, tokenization call, forward pass, log-softmax, verdict
argmax, CSV emission, resume -- is our own code.

Emits verdicts_mine_<short>.csv incrementally (append + flush) so a silent MPS
death (KNOWN-ISSUES 13) leaves a checkpoint. Columns match the committed CSV for
row-by-row diffing, plus we keep 6-dp logprobs for a logprob-delta analysis.
"""
import argparse, hashlib, sys, time
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
VERDICTS = ["A", "B", "C"]
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
assert hashlib.sha256(JUDGE_TEMPLATE.encode()).hexdigest()[:16] == "975db383e05f56b3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--pairs", default=str(HERE / "pairs_reconstructed.parquet"))
    args = ap.parse_args()

    short = args.model.split("/")[-1]
    out_path = HERE / f"verdicts_mine_{short}.csv"
    df = pd.read_parquet(args.pairs)
    print(f"[load] {len(df)} pairs from {args.pairs}", flush=True)

    dev = args.device
    if dev == "mps" and not torch.backends.mps.is_available():
        dev = "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev)
    model.eval()
    print(f"[model] {short} loaded on {dev} dtype={next(model.parameters()).dtype}", flush=True)

    # verdict token ids (single-token requirement, our own assertion)
    vid = {}
    for v in VERDICTS:
        enc = tok.encode(v, add_special_tokens=False)
        assert len(enc) == 1, f"{v!r} not single-token: {enc}"
        vid[v] = enc[0]

    # resume: skip (pair_id, order) already written
    done = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = set(zip(prev["pair_id"], prev["order"]))
        print(f"[resume] {len(done)} rows already present", flush=True)
    fh = open(out_path, "a")
    if not done:
        fh.write("pair_id,order,verdict,lp_A,lp_B,lp_C,prompt_tokens,ms\n")

    t_all = time.time()
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
            logprobs = torch.log_softmax(logits, dim=-1)
            lp = {v: logprobs[vid[v]].item() for v in VERDICTS}
            verdict = max(lp, key=lp.get)
            ms = int((time.time() - t0) * 1000)
            fh.write(f"{row['pair_id']},{order},{verdict},{lp['A']:.6f},{lp['B']:.6f},"
                     f"{lp['C']:.6f},{ids.shape[1]},{ms}\n")
            n_new += 1
            if n_new % 20 == 0:
                fh.flush()
                el = time.time() - t_all
                print(f"[prog] {short}: {n_new} new rows, {el/60:.1f} min, "
                      f"{el/n_new*1000:.0f} ms/row", flush=True)
    fh.flush(); fh.close()
    # MPS-hazard guard: assert we produced the expected number of rows
    final = pd.read_csv(out_path)
    exp = 2 * len(df)
    print(f"[done] {short}: {len(final)}/{exp} rows -> {out_path} "
          f"(+{n_new} new, {(time.time()-t_all)/60:.1f} min)", flush=True)
    if len(final) != exp:
        print(f"[FAIL] expected {exp} rows, got {len(final)} -- possible silent MPS death",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
