#!/usr/bin/env python3
"""Reports divergence as a time-to-event with an observation window, because it is one.

Every arm generates a fixed number of tokens and stops. A record whose speculative output never
differs from the greedy baseline inside that window is recorded identical, which is not the same
claim as identical: it is no divergence observed within N tokens. The distinction is already in
the data, in `finish_reason`, and does not need inferring.

    diverged        the outputs differ, at token t
    identical       the generation reached EOS and never differed          - exact identity
    right-censored  the generation stopped at the cap and never differed   - not identity

The unit is tokens. An earlier version of this file measured the window in characters and reported
that 15 of 25 prompts in Phase A were censored while the rest were clean. That was an artefact of
the unit: characters per token run from 1.36 to 6.17 across this suite, a 4.5x spread, so a
character window varies 4.5x while the token window the design actually fixes does not vary at
all. Measured in tokens every record has the same 400-token window and no prompt is censored more
than another. What is true is both simpler and worse: every record in this study stopped at the
cap, so every identical verdict in it is right-censored, uniformly.

Token positions are recovered per record from its own characters-per-token ratio, since the server
response carries `predicted_n` and the text but not the token ids. That is exact at the record
level, and it is what makes a position on a Chinese prompt comparable to one on an English prompt.

Usage:
    truncation_audit.py results/*.json
"""

import collections
import glob
import json
import statistics
import sys

DIVERGED, IDENTICAL_EOS, CENSORED = "diverged", "identical_eos", "right_censored"


def chars_per_token(rec):
    n = rec.get("predicted_n") or 0
    t = len(rec.get("text") or "")
    return (t / n) if (n and t) else None


def classify(rec):
    """(state, token position or None, window in tokens).

    `hit_cap` and `finish_reason` both say whether the generation ran out of budget; either alone
    is enough, and both are checked so a file from an older harness still classifies.
    """
    div = rec.get("divergence")
    window = rec.get("predicted_n") or 0
    if not div:
        return None, None, window
    capped = bool(rec.get("hit_cap")) or rec.get("finish_reason") == "length"
    if div.get("identical"):
        return (CENSORED if capped else IDENTICAL_EOS), None, window
    i = div.get("first_diff_char")
    if i is None:
        return None, None, window
    # A difference reported at the end of the shorter text is a length difference, not a fork.
    limit = min(div.get("len_ref", 0), div.get("len_arm", 0))
    if limit and i >= limit:
        return None, None, window
    cpt = chars_per_token(rec)
    return DIVERGED, (i / cpt if cpt else None), window


def resolved_forks(data):
    """Token positions of the divergences this file actually established."""
    out = []
    for rec in data["records"]:
        state, pos, _ = classify(rec)
        if state == DIVERGED and pos is not None:
            out.append(pos)
    return out


def windows(data):
    return sorted({w for _, _, w in (classify(r) for r in data["records"]) if w})


def censored_prompts(data, threshold=None):
    """Prompts holding an identical verdict the generation cap could have produced.

    With a uniform token window this is every prompt with an identical record, because the
    censoring is uniform: there is no subset of prompts that is safe to check the others against.
    `threshold` is accepted and ignored - it belonged to the character version, and callers still
    pass it.
    """
    out = set()
    for rec in data["records"]:
        if classify(rec)[0] == CENSORED:
            out.add(rec["prompt"])
    ws = windows(data)
    return out, (ws[0] if len(ws) == 1 else None)


def study_threshold(paths=None):
    """The latest token at which a divergence has been resolved anywhere in the study."""
    if paths is None:
        paths = [p for p in glob.glob("results/*.json") if ".partial." not in p]
    best = 0.0
    for path in paths:
        try:
            with open(path) as fh:
                best = max([best] + resolved_forks(json.load(fh)))
        except Exception:
            continue
    return best


def pass_agreement(data):
    """Does a prompt-arm cell give the same verdict in every pass it was measured in?

    The analysers collapse passes by overwriting, which is only sound if they agree. They do here,
    but nothing asserted it.
    """
    cell = collections.defaultdict(dict)
    for rec in data["records"]:
        state, pos, _ = classify(rec)
        if state is None:
            continue
        key = (rec["arm"], rec["prompt"])
        cell[key][rec["pass"]] = (state, None if pos is None else round(pos))
    multi = {k: v for k, v in cell.items() if len(v) > 1}
    disagree = {k: v for k, v in multi.items() if len(set(v.values())) > 1}
    return len(multi), disagree


def audit(path):
    with open(path) as fh:
        data = json.load(fh)
    counts = collections.Counter()
    forks = []
    ws = collections.Counter()
    for rec in data["records"]:
        state, pos, window = classify(rec)
        if state is None:
            continue
        counts[state] += 1
        if window:
            ws[window] += 1
        if state == DIVERGED and pos is not None:
            forks.append(pos)
    return path, counts, forks, ws, pass_agreement(data)


def main():
    paths = sys.argv[1:] or sorted(glob.glob("results/*.json"))
    paths = [p for p in paths if ".partial." not in p]
    print("=" * 100)
    print("DIVERGENCE AS A TIME-TO-EVENT, IN TOKENS")
    print("=" * 100)
    total = collections.Counter()
    for path in paths:
        try:
            path, counts, forks, ws, (n_multi, disagree) = audit(path)
        except Exception as exc:
            print("  %-40s unreadable: %s" % (path, exc))
            continue
        if not counts:
            continue
        total.update(counts)
        n = sum(counts.values())
        win = ", ".join("%d tokens x%d" % (w, c) for w, c in sorted(ws.items()))
        print("\n  %s" % path)
        print("    window            : %s" % (win or "unknown"))
        print("    diverged          : %4d / %d" % (counts[DIVERGED], n))
        print("    identical at EOS  : %4d / %d   <- the only exact identities" % (counts[IDENTICAL_EOS], n))
        print("    right-censored    : %4d / %d   <- no divergence within the window" % (counts[CENSORED], n))
        if forks:
            forks.sort()
            print("    fork position     : min %.0f  median %.0f  max %.0f tokens"
                  % (forks[0], statistics.median(forks), forks[-1]))
            if len(ws) == 1:
                w = next(iter(ws))
                print("                        the latest sits at %.0f %% of the window, so a fork"
                      % (100.0 * forks[-1] / w))
                print("                        past it would not have been seen")
        if n_multi:
            if disagree:
                print("    PASSES DISAGREE   : %d of %d cells give different verdicts across passes;"
                      % (len(disagree), n_multi))
                print("                        collapsing them by overwriting is unsound here")
                for k, v in list(disagree.items())[:3]:
                    print("                          %s %s" % (k, v))
            else:
                print("    pass agreement    : %d cells measured more than once, all agree" % n_multi)

    print("\n%s" % ("=" * 100))
    if sum(total.values()):
        print("  Across every file: %d diverged, %d identical at EOS, %d right-censored."
              % (total[DIVERGED], total[IDENTICAL_EOS], total[CENSORED]))
        if total[IDENTICAL_EOS] == 0:
            print("  No record anywhere reached EOS, so no identity in this study is exact. Every")
            print("  one means 'did not diverge within the token budget'. The censoring is uniform,")
            print("  so there is no subset of prompts to check the others against and the only")
            print("  thing that resolves it is re-running with a larger budget. TODO.md item D2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
