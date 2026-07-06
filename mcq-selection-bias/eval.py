"""Score MCQ option-ID logits for a Pythia model over cyclic option permutations.

For each question we present the 4 options labelled A/B/C/D and read the model's
next-token logits at the "Answer:" position, restricted to the 4 option-ID tokens
(" A"," B"," C"," D"). We do this for all 4 CYCLIC shifts of the option->label
mapping, so the correct answer visits every label exactly once. This exposes
selection bias: a content-independent preference for particular option IDs.

Output: a raw-scores CSV with one row per (question, shift), columns
  model, revision, qid, shift, correct_label, logit_A, logit_B, logit_C, logit_D
All downstream metrics (recall/RStd, accuracy, PriDe residual) are recomputed
from THIS CSV by analyze.py using only numpy/pandas (no model, no GPU).

Usage:
  python eval.py --model EleutherAI/pythia-160m-deduped --revision step143000 \
      --data data/mmlu_subset.jsonl --out results/scores_pythia-160m.csv \
      --dtype float32 --batch-size 16
"""
import argparse, json, os, sys, time, pathlib
import torch

PROMPT_TMPL = (
    "The following is a multiple choice question. Answer with the letter of the "
    "correct option.\n\nQuestion: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"
)
LETTERS = ["A", "B", "C", "D"]


def load_data(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_prompt(q, opts_in_display_order):
    a, b, c, d = opts_in_display_order
    return PROMPT_TMPL.format(q=q, a=a, b=b, c=c, d=d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--data", default="data/mmlu_subset.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap #questions (0=all)")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    print(f"[eval] model={args.model} rev={args.revision} device={device} dtype={args.dtype}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # option-ID token ids: the token for " A" .. " D" (leading space after "Answer:")
    label_ids = []
    for L in LETTERS:
        ids = tok(" " + L, add_special_tokens=False).input_ids
        label_ids.append(ids[-1])
    assert len(set(label_ids)) == 4, f"label token ids not distinct: {label_ids}"
    print(f"[eval] label token ids (' A'..' D') = {label_ids}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=dtype)
    model.to(device)
    model.eval()

    data = load_data(args.data)
    if args.limit:
        data = data[: args.limit]
    print(f"[eval] {len(data)} questions x 4 cyclic shifts = {len(data)*4} forward passes", flush=True)

    # Build all (qid, shift) prompts.
    jobs = []  # (qid, shift, correct_label, prompt)
    for rec in data:
        q = rec["question"]
        opts = rec["options"]
        c = rec["answer"]  # original correct index
        for s in range(4):
            # display position p shows original option (p+s) mod 4
            display = [opts[(p + s) % 4] for p in range(4)]
            # correct option originally at index c is shown at display position p where (p+s)%4==c
            corr_pos = (c - s) % 4
            prompt = build_prompt(q, display)
            jobs.append((rec["qid"], s, LETTERS[corr_pos], prompt))

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = len(jobs)
    rows_out = []
    bs = args.batch_size
    with torch.no_grad():
        for bstart in range(0, n, bs):
            batch = jobs[bstart : bstart + bs]
            prompts = [j[3] for j in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to(device)
            out = model(**enc)
            last_logits = out.logits[:, -1, :]  # (B, vocab); left-padded so -1 is real last token
            sel = last_logits[:, label_ids].float().cpu()  # (B, 4)
            for k, (qid, s, corr, _) in enumerate(batch):
                la, lb, lc, ld = [float(x) for x in sel[k]]
                rows_out.append((args.model, args.revision, qid, s, corr, la, lb, lc, ld))
            done = bstart + len(batch)
            if (bstart // bs) % 10 == 0 or done >= n:
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                eta = (n - done) / rate if rate > 0 else 0
                print(f"[eval] {done}/{n}  {el:6.1f}s  {rate:5.1f} it/s  eta {eta:5.0f}s", flush=True)

    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "revision", "qid", "shift", "correct_label",
                    "logit_A", "logit_B", "logit_C", "logit_D"])
        w.writerows(rows_out)
    print(f"[eval] DONE wrote {len(rows_out)} rows -> {out_path} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
