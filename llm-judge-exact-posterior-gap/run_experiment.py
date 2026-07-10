"""
INV-E1 experiment runner.

Conditions (KNOWN-MODEL regime):
  A  numeric-known-credulous   -> BOTH judges (weak + strong), all 20 graphs
  B  numeric-known-skeptical   -> weak judge, first 10 graphs
  C  verbal-known-credulous    -> weak judge, first 10 graphs

Writes results/calls.jsonl incrementally (one JSON per call). Resumable: skips
(graph_id, set_id, condition, judge_model) keys already recorded. Calls are
ordered by graph_id so early graphs complete first (publishable partial).

Usage:
  python run_experiment.py --mode pilot   # 2 graphs x 8 sets x weak x cond A
  python run_experiment.py --mode full
  python run_experiment.py --mode smoke    # 1 graph, MockJudge, no API
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from substrate import make_graph, sample_sets_for_llm, set_id
from nl_render import DOMAIN_NAMES, build_prompt
from judge import call_judge, parse_final

WEAK = "claude-haiku-4-5-20251001"
STRONG = "claude-sonnet-5"

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_GRAPHS = 20
MAX_WORKERS = 8
MAX_ATTEMPTS = 3   # initial + 2 retries
CALL_TIMEOUT = 180  # s. Raised from the 90s planning figure: pilot showed the
                    # judge reasons step-by-step (max 87s), so 90s would fail the
                    # hardest (largest-graph / strong-judge) calls systematically.

CONDITIONS = {
    "A": dict(presentation="numeric", skeptical=False, judges=[WEAK, STRONG],
              graphs=range(N_GRAPHS)),
    "B": dict(presentation="numeric", skeptical=True, judges=[WEAK],
              graphs=range(10)),
    "C": dict(presentation="verbal", skeptical=False, judges=[WEAK],
              graphs=range(10)),
}


def all_graphs():
    return [make_graph(i, DOMAIN_NAMES[i % len(DOMAIN_NAMES)]) for i in range(N_GRAPHS)]


def build_tasks(graphs, cap=24, conditions=CONDITIONS, sets_by_graph=None):
    """Yield task dicts, ordered by graph_id (early graphs first)."""
    tasks = []
    for g in graphs:
        sel = sets_by_graph[g.graph_id] if sets_by_graph else sample_sets_for_llm(g, cap)
        for cond, meta in conditions.items():
            if g.graph_id not in list(meta["graphs"]):
                continue
            for judge_model in meta["judges"]:
                for s in sel:
                    tasks.append(dict(
                        graph_id=g.graph_id,
                        set_id=set_id(g, s),
                        condition=cond,
                        judge_model=judge_model,
                        presentation=meta["presentation"],
                        skeptical=meta["skeptical"],
                        reveal_ids=sorted(s),
                        exact_posterior=g.posterior(s),
                        prior=g.p0,
                        domain=g.domain,
                    ))
    tasks.sort(key=lambda t: (t["graph_id"], t["condition"], t["judge_model"],
                              len(t["reveal_ids"]), t["set_id"]))
    return tasks


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # only count non-failed OR failed records so we don't retry
                # endlessly across restarts; a fully-exhausted call is 'done'.
                keys.add((r["graph_id"], r["set_id"], r["condition"], r["judge_model"]))
    return keys


class MockJudge:
    """Deterministic offline judge: exact posterior + seeded noise. For smoke."""
    def __init__(self, sigma=0.06):
        self.sigma = sigma

    def __call__(self, prompt, model, timeout=90, exact=0.5, key=0):
        import random
        rng = random.Random(hash((model, key)) & 0xffffffff)
        v = min(0.999, max(0.001, exact + rng.gauss(0, self.sigma)))
        return f"Reasoning briefly.\nFINAL: {v:.3f}", 0.001, True


def run(mode):
    graphs = all_graphs()

    if mode == "pilot":
        out_path = os.path.join(RESULTS_DIR, "pilot_calls.jsonl")
        sub = [graphs[0], graphs[1]]
        sets_by_graph = {g.graph_id: sample_sets_for_llm(g, cap=8) for g in sub}
        conds = {"A": dict(presentation="numeric", skeptical=False,
                           judges=[WEAK], graphs=range(N_GRAPHS))}
        tasks = build_tasks(sub, cap=8, conditions=conds, sets_by_graph=sets_by_graph)
        mock = None
    elif mode == "smoke":
        out_path = os.path.join(RESULTS_DIR, "smoke_calls.jsonl")
        sub = [graphs[0]]
        sets_by_graph = {graphs[0].graph_id: sample_sets_for_llm(graphs[0], cap=8)}
        tasks = build_tasks(sub, cap=8, sets_by_graph=sets_by_graph)
        mock = MockJudge()
    else:  # full
        out_path = os.path.join(RESULTS_DIR, "calls.jsonl")
        tasks = build_tasks(graphs, cap=24)
        mock = None

    done = done_keys(out_path)
    todo = [t for t in tasks
            if (t["graph_id"], t["set_id"], t["condition"], t["judge_model"]) not in done]
    print(f"[{mode}] total tasks={len(tasks)}  already done={len(done & set((t['graph_id'],t['set_id'],t['condition'],t['judge_model']) for t in tasks))}  to run={len(todo)}",
          flush=True)

    lock = threading.Lock()
    out_f = open(out_path, "a", buffering=1)
    counters = dict(done=0, failed=0, parsed=0)
    t_start = time.time()

    graph_by_id = {g.graph_id: g for g in graphs}

    def work(t):
        g = graph_by_id[t["graph_id"]]
        prompt = build_prompt(g, set(t["reveal_ids"]), t["presentation"], t["skeptical"])
        raw, latency, ok, verdict = None, 0.0, False, None
        attempts = 0
        for attempt in range(MAX_ATTEMPTS):
            attempts = attempt + 1
            if mock is not None:
                raw, latency, ok = mock(prompt, t["judge_model"],
                                        exact=t["exact_posterior"],
                                        key=(t["graph_id"], t["set_id"], t["condition"]))
            else:
                raw, latency, ok = call_judge(prompt, t["judge_model"], timeout=CALL_TIMEOUT)
            verdict = parse_final(raw) if ok else None
            if ok and verdict is not None:
                break
        rec = dict(
            graph_id=t["graph_id"], set_id=t["set_id"], condition=t["condition"],
            judge_model=t["judge_model"], presentation=t["presentation"],
            skeptical=t["skeptical"], domain=t["domain"], prior=t["prior"],
            reveal_ids=t["reveal_ids"], n_revealed=len(t["reveal_ids"]),
            exact_posterior=t["exact_posterior"], raw_response_text=raw,
            parsed_verdict=verdict, latency=round(latency, 3), attempts=attempts,
            ok=bool(ok and verdict is not None),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
        return rec

    with ThreadPoolExecutor(max_workers=(1 if mock else MAX_WORKERS)) as ex:
        futs = [ex.submit(work, t) for t in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                out_f.write(json.dumps(rec) + "\n")
                counters["done"] += 1
                if rec["parsed_verdict"] is None:
                    counters["failed"] += 1
                else:
                    counters["parsed"] += 1
                d = counters["done"]
                if d % 20 == 0 or d == len(todo):
                    el = time.time() - t_start
                    rate = d / el if el > 0 else 0
                    eta = (len(todo) - d) / rate if rate > 0 else 0
                    print(f"  progress {d}/{len(todo)}  parsed={counters['parsed']} "
                          f"failed={counters['failed']}  elapsed={el:.0f}s "
                          f"rate={rate:.2f}/s eta={eta:.0f}s", flush=True)

    out_f.close()
    el = time.time() - t_start
    print(f"[{mode}] DONE ran={counters['done']} parsed={counters['parsed']} "
          f"failed={counters['failed']} parse_rate="
          f"{counters['parsed']/max(1,counters['done']):.3f} elapsed={el:.0f}s -> {out_path}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full", "smoke"], default="full")
    args = ap.parse_args()
    run(args.mode)
