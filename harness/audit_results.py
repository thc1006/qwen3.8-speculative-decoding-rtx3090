"""Audit every result file this study has produced, on the checks that have actually failed.

`completeness.py` answers one question -- does the file hold as many records as its own design
promised -- and it answers it from the file itself, which is the right scope for the callers that
use it inline. It cannot see three failure modes this repository has hit:

  SHAPE      A count is not a design. 375 records is also one arm repeated fifteen times, and
             the rung gates delete 20 GB of weights on a passing count. scripts/run_phase_q.sh's gate was
             rewritten in August to compare against the explicit product of arms and passes after
             a fabricated result satisfied the old one.

  INCIDENTS  A run can be complete and still be contaminated. On 2026-08-26 a `sha256sum` on a
             17.5 GB model landed on `pass02_baseline@Q4_K_M` at 57 % CPU; the baseline is the
             divisor for every speculative arm in its pass and came out 0.49 % slow, five to ten
             times the 0.04-0.10 % those arms move between passes. The file was complete.

  PROVENANCE `env.model_sha256` says which weights ran; `env.model_size_bytes` is what makes a
             quantization ladder plottable after its weights are staged out. Results measured
             before those fields existed are not wrong, but a ladder cannot use them, and the
             difference has to be visible rather than discovered when a figure refuses to draw.

Everything here is read-only and reports rather than repairs. FAIL means the result should not be
used as it stands; NOTE means something is absent that a later phase added, which is history and
not a defect.

    python3 harness/audit_results.py                 # every results/*.json
    python3 harness/audit_results.py results/phase_m.json
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import completeness as CP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def audit(path: Path) -> dict:
    out = {"path": path, "fails": [], "notes": []}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        out["fails"].append(f"unreadable: {e.__class__.__name__}")
        return out

    recs = d.get("records") or []
    out["n"] = len(recs)
    if not recs:
        out["fails"].append("no records")
        return out

    got, expected, note = CP.completeness(d)
    out["expected"] = expected
    if expected and got < expected:
        out["fails"].append(f"{got}/{expected} records")
    elif expected and got > expected:
        out["fails"].append(f"{got} records against an expected {expected}; more than the design")
    if note:
        out["notes"].append(note)

    # SHAPE. The declared arms come from the matrix at run time and are the design; the records
    # are what happened. Comparing the two is what a count cannot do.
    design = d.get("design") or {}
    passes = design.get("passes")
    n_prompts = design.get("n_prompts")
    declared = sorted(d.get("arms") or {})
    shape = collections.Counter((r.get("arm"), r.get("pass")) for r in recs)
    if declared and passes and n_prompts:
        want = {(a, p) for a in declared for p in range(1, passes + 1)}
        missing, extra = want - set(shape), set(shape) - want
        if missing:
            out["fails"].append(f"{len(missing)}/{len(want)} arm-passes missing, "
                                f"e.g. {sorted(missing)[:2]}")
        if extra:
            out["fails"].append(f"arm-passes the design does not declare: {sorted(extra)[:2]}")
        ragged = {k: v for k, v in shape.items() if v != n_prompts}
        if ragged:
            out["fails"].append(f"arm-passes not holding {n_prompts} prompts: "
                                f"{sorted(ragged.items())[:2]}")
        out["shape"] = f"{len(shape)}/{len(want)}"
    else:
        # A result that declares no arms cannot have its shape checked against intent, and
        # `completeness()` falls back to counting arms in the records -- which is self-certifying.
        # Silence here would let a truncated or edited file pass every check, which is the exact
        # hole the rung drivers' gates were rewritten to close. Missing arms is a FAIL when there
        # are records to check; only a design block predating the field is a note.
        if recs and not declared:
            out["fails"].append(
                "records present but the result declares no arms, so shape cannot be checked "
                "against intent and any permutation would pass")
        else:
            out["notes"].append("design does not declare arms/passes/n_prompts; shape unchecked")
        out["shape"] = f"{len(shape)}/?"

    inc = d.get("incidents") or []
    out["incidents"] = len(inc)
    for i in inc:
        kind = i.get("kind", "?") if isinstance(i, dict) else "?"
        where = f"{i.get('arm')} pass {i.get('pass')}" if isinstance(i, dict) else ""
        out["fails"].append(f"incident {kind} at {where}: {str(i.get('detail', ''))[:70]}"
                            if isinstance(i, dict) else f"incident {str(i)[:70]}")

    env = d.get("env") or {}
    out["sha"] = bool(env.get("model_sha256")) and env.get("model_sha256") != "unknown"
    out["size"] = env.get("model_size_bytes")
    if not out["sha"]:
        out["fails"].append("env.model_sha256 missing or 'unknown'; the weights are unidentified")
    if out["size"] is None:
        out["notes"].append("env.model_size_bytes absent (predates the field); a ladder cannot "
                            "place this rung -- backfill_model_size.py if it is one")
    elif out["size"] < 10**8:
        out["fails"].append(f"env.model_size_bytes {out['size']} is implausible for a model file")

    # Every arm the records mention must be declared, or `analyze.py` scores it against nothing.
    seen_arms = {r.get("arm") for r in recs}
    undeclared = sorted(a for a in seen_arms if declared and a not in declared)
    if undeclared:
        out["fails"].append(f"records name arms the design does not declare: {undeclared[:3]}")

    # A speculative arm that never drafted is a baseline wearing its name.
    arms_meta = d.get("arms") or {}
    silent = []
    for a in sorted(seen_arms):
        m = arms_meta.get(a) or {}
        if not m.get("expects_drafter"):
            continue
        drafted = sum((r.get("timings") or {}).get("t_draft_n") or 0
                      for r in recs if r.get("arm") == a)
        if not drafted:
            silent.append(a)
    if silent:
        # An n-gram method that never fires is not the same defect as a spec-type that was
        # accepted and ignored. `ngram-mod` defaults to n_min 48: it must predict at least 48
        # consecutive tokens from its table or it discards the whole draft
        # (common/speculative.cpp, draft_one). On a 25-prompt general writing/code/reasoning set
        # at 400 max_tokens, a 48-token verbatim repeat is not expected to occur, so zero drafts
        # is the designed behaviour rather than a fault. Reported as a note so it stays visible.
        ngram = [a for a in silent if "ngram" in a.lower()]
        other = [a for a in silent if a not in ngram]
        if other:
            out["fails"].append(f"arms declared speculative that drafted nothing: {other}")
        if ngram:
            out["notes"].append(
                f"n-gram arms that never drafted: {ngram}. Check the method's own threshold "
                f"before reading this as a defect -- ngram-mod needs n_min consecutive matched "
                f"tokens (default 48) and emits nothing below it. Such an arm measures the "
                f"baseline, so its speedup and divergence figures are not about the method.")

    return out


def main() -> int:
    args = [Path(p) for p in sys.argv[1:]]
    if not args:
        args = sorted(p for p in (ROOT / "results").glob("*.json")
                      if ".partial." not in p.name and ".pre_repair" not in p.name)
    if not args:
        print("no result files", file=sys.stderr)
        return 2

    rows = [audit(p) for p in args]
    W = 118
    print("=" * W)
    print(f"RESULT AUDIT -- {len(rows)} files")
    print("=" * W)
    print(f"{'file':38s} {'records':>9s} {'shape':>9s} {'inc':>4s} {'sha':>4s} {'size':>14s}  verdict")
    print("-" * W)
    n_fail = 0
    for r in rows:
        name = r["path"].stem[:38]
        if r["fails"]:
            n_fail += 1
        size = f"{r['size']:,}" if r.get("size") else "-"
        print(f"{name:38s} {r.get('n', 0):>9d} {r.get('shape', '-'):>9s} "
              f"{r.get('incidents', 0):>4d} {'yes' if r.get('sha') else 'NO':>4s} {size:>14s}  "
              f"{'FAIL' if r['fails'] else 'ok'}")
    print("-" * W)
    for r in rows:
        if not (r["fails"] or r["notes"]):
            continue
        print(f"\n{r['path'].name}")
        for f in r["fails"]:
            print(f"  FAIL  {f}")
        for n in r["notes"]:
            print(f"  note  {n}")
    print("\n" + "=" * W)
    print(f"{len(rows) - n_fail} of {len(rows)} clean; {n_fail} with at least one FAIL")
    print("A FAIL means the result should not be used as it stands. A note is history: something")
    print("a later phase added that this file predates.")
    print("=" * W)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
