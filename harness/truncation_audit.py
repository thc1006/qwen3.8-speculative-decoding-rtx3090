#!/usr/bin/env python3
"""Finds divergence verdicts that the generation cap could have censored.

Every arm generates a fixed number of tokens and stops, so a record whose speculative output
never differs from the greedy baseline is reported identical. That is only the same thing as
"identical" when the output ran long enough for a divergence to have shown. Four hundred tokens
is roughly 2500 characters of English and roughly 950 of dense Chinese or of code, and forks in
this study have been resolved as late as character 1537, so a 960-character record marked
identical is not evidence of identity: it is evidence of no divergence in the first 960
characters.

The threshold is taken per file as the largest fork that file actually resolved. A fork that late
demonstrably occurs under these conditions, so an output shorter than it could have hidden one.
Using a fixed number instead would import an assumption from a different run.

This matters only where a censored verdict carries a conclusion. A prompt whose widths do not
partition cleanly is consistent with everything already; a prompt whose two-group split rests on
a censored identical is asserting something the data does not establish.

Usage:
    truncation_audit.py results/*.json
"""

import collections
import glob
import json
import sys

LOW, HIGH = (3, 4), (5, 6, 8)


def width_of(arm_name, meta):
    """Verification width from the arm's own arguments: n_max + 1, or 1 for a greedy baseline.

    Reading it from extra_args rather than a table of arm names keeps this correct for matrices
    that use different names for the same width, which is why phase_a and phase_nmax cannot share
    a hard-coded map.
    """
    args = (meta or {}).get("extra_args") or []
    for i, a in enumerate(args):
        if a == "--spec-draft-n-max" and i + 1 < len(args):
            try:
                return int(args[i + 1]) + 1
            except ValueError:
                return None
    return 1 if not (meta or {}).get("expects_drafter") else None


def audit(path):
    with open(path) as fh:
        data = json.load(fh)
    arms = data.get("arms", {})
    widths = {name: width_of(name, meta) for name, meta in arms.items()}

    resolved = [r["divergence"]["first_diff_char"] for r in data["records"]
                if r.get("divergence") and not r["divergence"].get("identical")
                and r["divergence"].get("first_diff_char") is not None]
    if not resolved:
        return path, None, [], [], 0
    threshold = max(resolved)

    pos = collections.defaultdict(dict)
    censored = collections.defaultdict(set)
    for rec in data["records"]:
        w = widths.get(rec["arm"])
        if w is None or w == 1:
            continue
        div = rec.get("divergence")
        if not div:
            continue
        if div.get("identical"):
            pos[rec["prompt"]][w] = "same"
            if len(rec.get("text") or "") < threshold:
                censored[rec["prompt"]].add(w)
        else:
            pos[rec["prompt"]][w] = div["first_diff_char"]

    need = set(LOW) | set(HIGH)
    full = [p for p in pos if need.issubset(pos[p])]
    tainted, harmless = [], []
    for p in full:
        q = pos[p]
        split = (len({q[w] for w in LOW}) == 1
                 and len({q[w] for w in HIGH}) == 1
                 and q[LOW[0]] != q[HIGH[0]])
        if not censored.get(p):
            continue
        (tainted if split else harmless).append(p)
    return path, threshold, sorted(tainted), sorted(harmless), len(full)


def main():
    paths = sys.argv[1:] or sorted(glob.glob("results/*.json"))
    paths = [p for p in paths if ".partial." not in p]
    print("=" * 100)
    print("TRUNCATION AUDIT: divergence verdicts the generation cap could have censored")
    print("=" * 100)
    any_tainted = False
    for path in paths:
        try:
            path, threshold, tainted, harmless, full = audit(path)
        except Exception as exc:
            print("  %-44s unreadable: %s" % (path, exc))
            continue
        if threshold is None:
            print("  %-44s no resolved forks; nothing to censor" % path)
            continue
        if full == 0:
            print("  %-44s no prompt carries every width in %s + %s; not covered by this test"
                  % (path, list(LOW), list(HIGH)))
            continue
        print("  %-44s %3d prompts, cap threshold %4d chars" % (path, full, threshold))
        if tainted:
            any_tainted = True
            print("       SPLIT VERDICT RESTS ON A CENSORED IDENTICAL: %s" % ", ".join(tainted))
        if harmless:
            print("       censored but carrying no split verdict: %s" % ", ".join(harmless))
        if not tainted and not harmless:
            print("       clean")
    if any_tainted:
        print("\n  A tainted prompt is not a wrong measurement; the record correctly says the two")
        print("  outputs did not differ over the tokens that were generated. It is a verdict that")
        print("  should not be counted as a clean split until the prompt is re-run long enough for")
        print("  a late fork to show. Exclude, or extend the cap for those prompts and re-measure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
