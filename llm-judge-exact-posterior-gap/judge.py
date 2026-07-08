"""Judge caller (subprocess -> claude CLI one-shot) and FINAL: parser."""

import re
import subprocess

CLAUDE_BIN = ("/Users/alexroman/Library/Application Support/Claude/"
              "claude-code/2.1.197/claude.app/Contents/MacOS/claude")

_DECIMAL = re.compile(r"(?<![\w.])(\d?\.\d+|\d+(?:\.\d+)?)(?!\w)")


def parse_final(text):
    """Extract the verdict.  Prefer the number on a 'FINAL:' line; else fall back
    to the last in-range [0,1] decimal in the text.  Returns float or None."""
    if text is None:
        return None
    # 1) explicit FINAL: line (last one wins)
    val = None
    for m in re.finditer(r"FINAL\s*:?\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE):
        try:
            v = float(m.group(1))
            if 0.0 <= v <= 1.0:
                val = v
        except ValueError:
            pass
    if val is not None:
        return val
    # 2) fallback: last in-range decimal anywhere
    cands = []
    for m in _DECIMAL.finditer(text):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            cands.append(v)
    return cands[-1] if cands else None


def call_judge(prompt, model, timeout=90):
    """Single one-shot call. Returns (raw_text, latency_seconds, ok)."""
    import time
    t0 = time.time()
    try:
        proc = subprocess.run(
            [CLAUDE_BIN, "-p", "--model", model, prompt],
            capture_output=True, text=True, timeout=timeout)
        dt = time.time() - t0
        out = (proc.stdout or "").strip()
        if proc.returncode != 0 and not out:
            return (proc.stderr or "").strip(), dt, False
        return out, dt, True
    except subprocess.TimeoutExpired:
        return "__TIMEOUT__", time.time() - t0, False
    except Exception as e:  # noqa
        return f"__ERROR__ {e}", time.time() - t0, False
