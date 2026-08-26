#!/usr/bin/env python3
"""Says whether a result file holds every record its own design promised.

Written after a table in the README was filled from `phase_nmax.json` while the run was still
appending to it. The file had 1025 of 1050 records, the missing 25 were one baseline arm's third
pass, and the DFlash2 coefficient it produced was 0.2479 against the finished file's 0.2481. Small
enough to survive review and wrong.

The design block a run writes carries the passes, and the arms and prompt tags are recoverable from
the records, so the expected count can be reconstructed from the file itself without knowing which
matrix produced it.
"""

import json
import sys


def completeness(result):
    """(records, expected, note). expected is None when the file does not say enough."""
    recs = result.get("records") or []
    if not recs:
        return 0, None, "no records"
    design = result.get("design") or {}
    passes = design.get("passes")
    arms = len(result.get("arms") or {}) or len({r["arm"] for r in recs})
    seen_prompts = len({r["prompt"] for r in recs})
    seen_passes = len({r["pass"] for r in recs})
    notes = []
    # Declared, not observed. Deriving the expected count from what is in the file makes the
    # check circular: a run that died in pass 1 after ten prompts has ten distinct prompts, and
    # 1 arm x 10 prompts x 1 pass is exactly what it holds, so it reports complete. bench.py
    # writes design.n_prompts from the matrix, which is what the run set out to do.
    prompts = (result.get("design") or {}).get("n_prompts") or seen_prompts
    if not (result.get("design") or {}).get("n_prompts"):
        notes.append("prompt count not recorded; taken from the records, so a run that stopped "
                     "inside its first pass cannot be detected")
    if not passes:
        passes = seen_passes
        notes.append("passes not recorded; taken from the records, so this can only detect a "
                     "short pass")
    note = "; ".join(notes)
    expected = arms * prompts * passes
    return len(recs), expected, note


def warn_if_incomplete(result, path=""):
    """Print a warning when a file is short. Returns True when it is complete or unknowable."""
    n, expected, note = completeness(result)
    if not expected or n >= expected:
        return True
    pct = 100.0 * n / expected
    print(f"\n[incomplete] {path or 'this file'} holds {n} of {expected} records ({pct:.0f} %). "
          f"A run still appending")
    print(f"             to it produces numbers that move: phase_nmax at 1025 of 1050 gave a "
          f"DFlash2 coefficient")
    print(f"             of 0.2479 against 0.2481 finished. Everything below is provisional.")
    if note:
        print(f"             {note}")
    return False


if __name__ == "__main__":
    for p in sys.argv[1:]:
        with open(p) as fh:
            d = json.load(fh)
        n, e, note = completeness(d)
        state = "complete" if (e and n >= e) else ("short" if e else "unknown")
        print(f"  {p:<46} {n:>5} / {e if e else '?':<5}  {state}" + (f"   {note}" if note else ""))
