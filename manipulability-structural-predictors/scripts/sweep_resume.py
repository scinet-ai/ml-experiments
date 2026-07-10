"""Resume-safe wrapper around the main sweep: appends to an existing corpus,
skips seeds already present, and guards each graph with a SIGALRM timeout so a
single pathological high-fan-in graph (exponential corner enumeration in the
outer bound) cannot hang the run. Timeouts are counted and reported in the
manifest — they are a declared coverage cap, not silent truncation.

Usage: python scripts/sweep_resume.py [N] [out_csv] [timeout_s]
"""
from __future__ import annotations

import csv
import json
import os
import random
import signal
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features  # noqa: E402
from sweep import REGIMES, build_schedule, PARAM_SEED_BASE  # noqa: E402

from probability_flow.aspic.generate import generate  # noqa: E402

warnings.simplefilter("ignore")


class GraphTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise GraphTimeout()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 7200
    out = sys.argv[2] if len(sys.argv) > 2 else "results/corpus.csv"
    timeout_s = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    done = set()
    fieldnames = None
    if os.path.exists(out) and os.path.getsize(out) > 0:
        with open(out, newline="") as f:
            r = csv.DictReader(f)
            fieldnames = r.fieldnames
            for row in r:
                done.add(int(row["seed"]))
    print(f"resuming: {len(done)} seeds already in corpus", flush=True)

    samplers = {name: fn for name, _w, fn in REGIMES}
    schedule = build_schedule(n)
    signal.signal(signal.SIGALRM, _alarm)

    kept = 0
    failed = 0
    discarded = 0
    timeouts = []
    fh = open(out, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames) if fieldnames else None

    t0 = time.time()
    attempted = 0
    for i in range(n):
        if i in done:
            continue
        attempted += 1
        regime = schedule[i]
        prng = random.Random(PARAM_SEED_BASE + i)
        params, targets, meta = samplers[regime](prng)
        signal.alarm(timeout_s)
        try:
            arg = generate(seed=i, structural=params, targets=targets,
                           max_attempts=200)
            row = extract_features(arg)
        except GraphTimeout:
            timeouts.append(i)
            continue
        except Exception:
            failed += 1
            continue
        finally:
            signal.alarm(0)
        if row is None:
            discarded += 1
            continue
        row["seed"] = i
        row["regime"] = regime
        row["target_posterior"] = meta.get("target_posterior", "")
        if writer is None:
            fieldnames = (["seed", "regime", "target_posterior"]
                          + [k for k in row.keys()
                             if k not in ("seed", "regime", "target_posterior")])
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
        writer.writerow(row)
        kept += 1
        if attempted % 250 == 0:
            fh.flush()
            el = time.time() - t0
            print(f"[resume {attempted}] kept={kept} fail={failed} "
                  f"timeout={len(timeouts)} {el:.1f}s", flush=True)

    fh.close()
    total_rows = len(done) + kept
    manifest = {
        "attempted_total": n, "resumed_from": len(done), "kept_new": kept,
        "total_rows": total_rows, "failed_generation": failed,
        "discarded_nonpolytree": discarded,
        "timeout_skipped": len(timeouts), "timeout_seconds": timeout_s,
        "timeout_seeds": timeouts[:50],
        "wall_seconds": round(time.time() - t0, 1),
        "param_seed_base": PARAM_SEED_BASE,
        "regime_weights": {name: w for name, w, _f in REGIMES},
        "note": "timeout-skipped graphs are a declared coverage cap (pathological "
                "high-fan-in outer-bound enumeration); their seeds are recorded so "
                "the exclusion is reproducible.",
    }
    with open(os.path.join(os.path.dirname(out) or ".", "sweep_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\n=== RESUME SWEEP DONE ===")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
