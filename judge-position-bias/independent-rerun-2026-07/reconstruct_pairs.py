#!/usr/bin/env python3
"""INDEPENDENT reconstruction of the 350-pair set with texts.

We reuse ONLY the committed pair-selection identity (pairs_public.parquet: which
(question_id, m1, m2) pairs, and their pair_id labels). We re-derive the actual
question/answer TEXTS ourselves straight from the CC-BY-4.0 source dataset
lmsys/mt_bench_human_judgments (split=human, turn=1), with our own join logic --
NOT their prep code path.

We then cross-check our reconstruction against three committed checksums that do
NOT reveal the texts:
  - len1_tok / len2_tok  (Qwen-tokenized answer lengths, from pairs_public.parquet)
  - prompt_tokens        (chat-templated prompt length, from the committed verdicts CSV)
If all match, our reconstructed texts are byte-identical to the ones the finding used.
"""
import hashlib, json, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE / "repo-read" / "judge-position-bias"

# --- our own verbatim copy of the pinned Zheng-style template (verified by sha256) ---
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
TEMPLATE_SHA = hashlib.sha256(JUDGE_TEMPLATE.encode()).hexdigest()[:16]
assert TEMPLATE_SHA == "975db383e05f56b3", f"template hash mismatch: {TEMPLATE_SHA}"
print(f"[ok] template sha256[:16] = {TEMPLATE_SHA}  (matches published protocol)")

# --- independently load the source dataset and build (qid,model)->answer, qid->question ---
from datasets import load_dataset
ds = load_dataset("lmsys/mt_bench_human_judgments", split="human")
qtext = {}
ans = {}       # (question_id, model) -> answer text
collisions = 0
for r in ds:
    if r["turn"] != 1:
        continue
    ca, cb = r["conversation_a"], r["conversation_b"]
    q = ca[0]["content"]
    if cb[0]["content"] != q:
        continue
    qtext.setdefault(r["question_id"], q)
    for conv, m in ((ca, r["model_a"]), (cb, r["model_b"])):
        a_ans = conv[1]["content"]
        key = (r["question_id"], m)
        if key in ans and ans[key] != a_ans:
            collisions += 1
        ans[key] = a_ans
print(f"[info] dataset: {len(qtext)} turn-1 questions, {len(ans)} (qid,model) answers, "
      f"{collisions} answer-text collisions")

# --- reuse committed pair identity, attach our re-derived texts ---
pairs = pd.read_parquet(REPO / "pairs_public.parquet")
rows = []
missing = 0
for _, r in pairs.iterrows():
    qid, m1, m2 = int(r["question_id"]), r["m1"], r["m2"]
    k1, k2 = (qid, m1), (qid, m2)
    if k1 not in ans or k2 not in ans or qid not in qtext:
        missing += 1
        continue
    rows.append({"pair_id": r["pair_id"], "question_id": qid, "m1": m1, "m2": m2,
                 "q": qtext[qid], "ans1": ans[k1], "ans2": ans[k2],
                 "human_majority": r["human_majority"],
                 "len1_tok_committed": int(r["len1_tok"]), "len2_tok_committed": int(r["len2_tok"])})
print(f"[info] reconstructed {len(rows)}/{len(pairs)} pairs ({missing} missing)")
df = pd.DataFrame(rows)

# --- cross-check 1: Qwen-tokenized answer lengths vs committed len1_tok/len2_tok ---
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
df["len1_tok_mine"] = df["ans1"].map(lambda a: len(tok.encode(a)))
df["len2_tok_mine"] = df["ans2"].map(lambda a: len(tok.encode(a)))
mism1 = int((df["len1_tok_mine"] != df["len1_tok_committed"]).sum())
mism2 = int((df["len2_tok_mine"] != df["len2_tok_committed"]).sum())
print(f"[check] len1_tok mismatches: {mism1}/{len(df)}   len2_tok mismatches: {mism2}/{len(df)}")

# --- cross-check 2: chat-templated prompt length (orig order) vs committed verdicts prompt_tokens ---
vc = pd.read_csv(REPO / "verdicts_Qwen2.5-1.5B-Instruct.csv")
ptok_orig = vc[vc["order"] == "orig"].set_index("pair_id")["prompt_tokens"].to_dict()
def prompt_len(row):
    prompt = JUDGE_TEMPLATE.format(q=row["q"], a=row["ans1"], b=row["ans2"])
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    ids = tok(text, add_special_tokens=False).input_ids
    return len(ids)
df["ptok_mine_orig"] = df.apply(prompt_len, axis=1)
df["ptok_committed_orig"] = df["pair_id"].map(ptok_orig)
mismp = int((df["ptok_mine_orig"] != df["ptok_committed_orig"]).sum())
print(f"[check] prompt_tokens (orig) mismatches: {mismp}/{len(df)}")

df.to_parquet(HERE / "pairs_reconstructed.parquet")
summary = {"n_pairs": len(df), "template_sha": TEMPLATE_SHA,
           "len1_mismatch": mism1, "len2_mismatch": mism2, "prompt_tok_mismatch": mismp,
           "answer_collisions": collisions}
(HERE / "reconstruct_summary.json").write_text(json.dumps(summary, indent=1))
print("[done]", json.dumps(summary))
if mism1 or mism2 or mismp:
    print("[WARN] some checksums mismatched -- reconstructed texts may differ from the finding's",
          file=sys.stderr)
