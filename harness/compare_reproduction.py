#!/usr/bin/env python3
"""Compare a reproduction against the run it is meant to reproduce.

`scripts/reproduce_phase_a.sh` checked that a rerun produced 875 records. A record count is a
shape, not a result: a rerun that lands on entirely different throughput would pass it. This
compares what the two runs actually measured, on the same estimator and against each arm's own
tree's baseline, and reports the provenance that decides whether the comparison means anything.

Two things it deliberately does not do.

It does not put an interval on the difference between the two runs. They are separate sessions on
one card; prompt pairing removes prompt difficulty and removes nothing that differs between two
moments hours or months apart. Each run's own interval is reported, and the gap between the point
estimates is reported, and no third interval is invented to cover the pair.

It does not read overlapping intervals as agreement. Correction 26 in PREREGISTRATION.md is about
exactly that mistake: two intervals that overlap have failed to exclude each other, which is much
weaker than landing on the same number. Non-overlap is evidence of disagreement and is what makes
this exit non-zero; overlap is reported as what it is.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import analyze as AN          # noqa: E402
import stats as ST            # noqa: E402

# Fields that describe what was measured rather than what came out. A difference in any of them
# means the two runs are not the same experiment, whatever their numbers say.
# Paths that actually resolve. The first version of this list carried three that did not --
# `design.kernel_facts.master.commit`, `env.llama_commit` and `env.driver` -- and none of them
# exists in a result file. `dig` returned None for both sides, None equals None, and the table
# printed three reassuring rows for fields nobody was comparing. A provenance check that resolves
# nothing is worse than no provenance check, because it looks like one.
PROVENANCE = [
    ("engine revisions", ("env", "llama_cpp_revisions")),
    ("model sha256", ("env", "model_sha256")),
    ("model bytes", ("env", "model_size_bytes")),
    ("model path", ("env", "model")),
    ("gpu and driver", ("env", "gpu")),
    ("clock state", ("env", "overclock_state")),
    ("kernel", ("env", "kernel")),
    ("python", ("env", "python")),
    ("host", ("env", "host")),
    ("device", ("design", "device", "name")),
    ("matrix sha256", ("matrix", "file_sha256")),
    ("max_tokens", ("design", "max_tokens")),
    ("passes", ("design", "passes")),
    ("n_prompts", ("design", "n_prompts")),
]


def dig(d, path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def baseline_of(result):
    """arm -> the baseline built from its own tree, the mapping analyze.py uses."""
    meta = result.get("arms") or {}
    base_by_tree = {(m or {}).get("tree"): a for a, m in meta.items()
                    if not (m or {}).get("extra_args")}
    return {a: base_by_tree.get((m or {}).get("tree")) for a, m in meta.items()}, base_by_tree


def effects(result):
    """(arm -> (baseline used, Interval), per-arm exclusions, per-arm quality flags).

    Everything comes from `analyze.build_series`, so the estimator, the exclusion rule and the
    quality flags are the same code that produced the committed report rather than a second
    implementation of them that could drift from it.
    """
    series, prompt_class, excluded, flagged = AN.build_series(result)
    bmap, base_by_tree = baseline_of(result)
    baselines = set(base_by_tree.values())
    out = {}
    for arm in sorted(series):
        ref = bmap.get(arm)
        if arm in baselines or not ref or ref not in series:
            continue
        arm_s, base_s = AN._balanced(series[arm], series[ref])
        if not arm_s:
            continue
        # (baseline, arm), not (arm, baseline): the other order silently inverts every sign.
        out[arm] = (ref, ST.paired_cluster_bootstrap(base_s, arm_s, prompt_class, relative=True))
    return out, dict(excluded), dict(flagged)


def incidents(result):
    return list(result.get("incidents") or [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("reference", help="the committed result the rerun should reproduce")
    ap.add_argument("candidate", help="the rerun")
    ap.add_argument("--allow-incidents", action="store_true",
                    help="do not fail on recorded incidents in either run; they still print")
    args = ap.parse_args()

    ref = AN.load(Path(args.reference))
    cand = AN.load(Path(args.candidate))
    problems = []

    print("=" * 100)
    print("REPRODUCTION COMPARISON")
    print("=" * 100)
    print(f"  reference {args.reference}")
    print(f"  candidate {args.candidate}")

    print("\n--- provenance: what was measured ---")
    for label, path in PROVENANCE:
        a, b = dig(ref, path), dig(cand, path)
        if a is None and b is None:
            # Not "they agree". Neither file has the field, so this row compared nothing, and
            # saying so is the whole point: three rows used to pass silently this way.
            print(f"  {label:16s} {'-- absent from both files, nothing compared --':<70}")
            problems.append(f"provenance field '{label}' resolves in neither file; "
                            f"{'.'.join(path)} is not a path a result carries")
            continue
        same = a == b
        if not same and (a is None or b is None):
            # Absence is not disagreement. `matrix sha256` reported "provenance differs" when
            # comparing an 08-27 result against an 08-28 one, and the two runs did not differ on
            # it at all -- the older schema had no such field to record. Calling that a
            # difference invites the reader to look for a change that is not there, and hides the
            # thing that IS true: this row could not be compared, so the matrix has to be
            # verified some other way or not claimed.
            which = "reference" if a is None else "candidate"
            print(f"  {label:16s} {str(a)[:34]:34s} {str(b)[:34]:34s}"
                  f"   <-- NOT RECORDED in {which}")
            problems.append(
                f"provenance not verifiable: {label} is absent from the {which}, so the two "
                f"runs cannot be compared on it (usually an older result written before the "
                f"field existed); verify it another way or do not claim it")
            continue
        mark = "" if same else "   <-- DIFFERS"
        print(f"  {label:16s} {str(a)[:34]:34s} {str(b)[:34]:34s}{mark}")
        if not same:
            problems.append(f"provenance differs: {label}")

    print("\n--- shape ---")
    for label, fn in (("records", lambda d: len(d.get("records") or [])),
                      ("arms", lambda d: len(d.get("arms") or {})),
                      ("incidents", lambda d: len(incidents(d)))):
        a, b = fn(ref), fn(cand)
        print(f"  {label:16s} {a:<34d} {b:<34d}{'' if a == b else '   <-- DIFFERS'}")
    n_incidents = {"reference": len(incidents(ref)), "candidate": len(incidents(cand))}
    for name, d in (("reference", ref), ("candidate", cand)):
        for inc in incidents(d):
            print(f"    {name} incident: {inc.get('kind')} at {inc.get('arm')} "
                  f"pass {inc.get('pass')}: {str(inc.get('detail'))[:70]}")
            if not args.allow_incidents:
                problems.append(f"{name} carries a recorded incident")

    ea, exa, fla = effects(ref)
    eb, exb, flb = effects(cand)

    print("\n--- records the quality rule set aside, and records it flagged ---")
    for label, ex, fl in (("reference", exa, fla), ("candidate", exb, flb)):
        n_ex = sum(len(v) for v in ex.values())
        n_fl = sum(len(v) for v in fl.values())
        print(f"  {label:10s} excluded {n_ex:<4d} flagged {n_fl:<4d}"
              + ("" if not n_ex else "  arms: " + ", ".join(sorted(ex))[:60]))
    if sum(len(v) for v in exa.values()) != sum(len(v) for v in exb.values()):
        problems.append("the two runs exclude different numbers of records")

    print("\n--- effect per arm, each against its own tree's baseline ---")
    print(f"  {'arm':22s} {'reference':>26s} {'candidate':>26s} {'delta pt':>9s}  overlap")
    for arm in sorted(set(ea) | set(eb)):
        if arm not in ea or arm not in eb:
            print(f"  {arm:22s} {'present' if arm in ea else 'absent':>26s} "
                  f"{'present' if arm in eb else 'absent':>26s}      <-- ONLY IN ONE")
            problems.append(f"{arm} is measured in only one of the two runs")
            continue
        (_, ia), (_, ib) = ea[arm], eb[arm]
        overlap = not (ia.hi < ib.lo or ib.hi < ia.lo)
        if not overlap:
            problems.append(f"{arm}: intervals do not overlap")
        print(f"  {arm:22s} {f'{ia.point:+.2f} [{ia.lo:+.2f}, {ia.hi:+.2f}]':>26s} "
              f"{f'{ib.point:+.2f} [{ib.lo:+.2f}, {ib.hi:+.2f}]':>26s} "
              f"{ib.point - ia.point:+9.2f}  {'yes' if overlap else 'NO'}")
    print("\n  Overlap is a failure to exclude, not agreement (Correction 26). No interval is put")
    print("  on the delta column: the two runs are separate sessions and nothing in this design")
    print("  pairs them.")

    print("\n" + "=" * 100)
    if problems:
        print(f"  NOT ESTABLISHED as a reproduction: {len(problems)} problem(s)")
        for p in problems:
            print(f"    - {p}")
        # Only when an incident is actually among the problems. This hint used to print on
        # every failure, so a run that failed solely on an unrecorded provenance field told the
        # reader to pass a flag that had already been passed, and pointed at contention that was
        # not the reason for anything.
        if any("recorded incident" in x for x in problems):
            print("  A recorded incident fails this by default; pass --allow-incidents to say the")
            print("  contention was checked and did not move the measurement, and say so in writing.")
        return 1
    print("  Every arm's intervals overlap and the provenance that both files record matches.")
    if n_incidents["reference"] or n_incidents["candidate"]:
        # The success line asserted "neither run carries an incident" while --allow-incidents was
        # suppressing exactly that check, and it printed that sentence over a reference file
        # carrying two. A summary that states the opposite of the shape table above it is worse
        # than no summary.
        print(f"  The reference carries {n_incidents['reference']} recorded incident(s) and the "
              f"candidate {n_incidents['candidate']};")
        print("  --allow-incidents was passed, so whether that contention moved the measurement")
        print("  is a judgement made outside this tool, not a check it performed.")
    else:
        print("  Neither run carries a recorded incident.")
    print("  That is consistent with a reproduction and is not proof of one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
