"""Prepare a fixed MMLU MCQ subset for the selection-bias experiment.

Downloads the `cais/mmlu` "all" test split (small: plain text) and writes a
deterministic subset to data/mmlu_subset.jsonl. Each record has exactly 4
options (A/B/C/D), which is what makes the cyclic-permutation bias analysis clean.

Also writes data/mmlu_sample.jsonl (a tiny vendored sample) used by verify.py.

Usage: python prepare_data.py --n 400 --seed 0
"""
import argparse, json, os, random, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/mmlu_subset.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    # keep only well-formed 4-option questions
    idxs = [i for i in range(len(ds)) if len(ds[i]["choices"]) == 4]
    rng = random.Random(args.seed)
    rng.shuffle(idxs)
    idxs = idxs[: args.n]

    here = pathlib.Path(__file__).parent
    out = here / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for qid, i in enumerate(idxs):
            row = ds[i]
            rec = {
                "qid": qid,
                "subject": row["subject"],
                "question": row["question"].strip(),
                "options": [c.strip() for c in row["choices"]],
                "answer": int(row["answer"]),  # 0..3 index into options
            }
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(idxs)} questions -> {out}")

    # tiny vendored sample for zero-download smoke tests (10 questions)
    sample = here / "data/mmlu_sample.jsonl"
    with open(out) as f, open(sample, "w") as g:
        for k, line in enumerate(f):
            if k >= 10:
                break
            g.write(line)
    print(f"wrote 10-question vendored sample -> {sample}")

if __name__ == "__main__":
    main()
