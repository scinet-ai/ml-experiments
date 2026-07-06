#!/usr/bin/env bash
# Zero-download smoke reproduction: trains ONE small seed of modular addition to
# grok (synthetic data, no download) and prints the grok step + each
# mechanism-agnostic measure's threshold-cross step and lead time.
# Exits 0 (PASS) iff the model groks and >=1 measure has positive lead time.
#
# Usage:   bash reproduce.sh
# Runtime: ~1-3 min on CPU.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
# Prefer a local uv venv if present; otherwise assume torch+numpy are importable.
if [ -f ".venv/bin/activate" ]; then source .venv/bin/activate; fi

export GROK_DEVICE="${GROK_DEVICE:-cpu}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "== dependency check =="
"$PY" - <<'PYEOF'
import importlib, sys
for m in ("torch", "numpy"):
    try:
        importlib.import_module(m)
    except Exception as e:
        sys.exit(f"missing dependency {m!r}: {e}\n  -> pip install -r requirements.txt")
print("torch + numpy present")
PYEOF

echo "== smoke reproduction (modular addition grokking + lead time) =="
"$PY" verify.py
