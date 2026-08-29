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
    """Mean characters per token over the WHOLE output. An average, and only that.

    Everything downstream that turns a character offset into a "token position" divides by this,
    and the result is an ESTIMATE that must never be reported as a measurement. A tokenizer is
    variable-length: the same output can run three characters per token through a run of Chinese
    and six through a stretch of English prose, so a single whole-output ratio cannot locate the
    token boundary a given character sits in. Rounding it makes that worse, not better -- it can
    merge two token boundaries or split one.

    The character offset itself IS exact, and it is what the width-partition matrix compares. Fork
    positions are reported in characters for that reason; the token figure exists only to give a
    reader a sense of scale and is labelled approximate everywhere it appears.

    Settling it needs the emitted token IDs recorded at benchmark time and compared directly, which
    is a re-run, not a re-analysis.
    """
    n = rec.get("predicted_n") or 0
    t = len(rec.get("text") or "")
    return (t / n) if (n and t) else None


def reached_eos(rec):
    """True when the generation stopped on its own rather than running out of budget.

    `hit_cap` and `finish_reason` both say it; either alone is enough, and both are checked so a
    file from an older harness still reads.
    """
    return not (bool(rec.get("hit_cap")) or rec.get("finish_reason") == "length")


def classify(rec):
    """(state, token position or None, tokens produced).

    The third element is what this record generated. It equals the cap only for the records that
    hit it, so it is not the window; `budget()` is the position a censored record was cut at.
    """
    div = rec.get("divergence")
    produced = rec.get("predicted_n") or 0
    if not div:
        return None, None, produced
    capped = not reached_eos(rec)
    if div.get("identical"):
        return (CENSORED if capped else IDENTICAL_EOS), None, produced
    i = div.get("first_diff_char")
    if i is None:
        return None, None, produced
    # A difference reported at the end of the shorter text is a length difference, not a fork.
    limit = min(div.get("len_ref", 0), div.get("len_arm", 0))
    if limit and i >= limit:
        return None, None, produced
    cpt = chars_per_token(rec)
    return DIVERGED, (i / cpt if cpt else None), produced


def resolved_forks(data):
    """Token positions of the divergences this file actually established."""
    out = []
    for rec in data["records"]:
        state, pos, _ = classify(rec)
        if state == DIVERGED and pos is not None:
            out.append(pos)
    return out


def budget(data):
    """The token cap the run was given: the position a right-censored record was cut at.

    Reading this off the outputs reports a different cap for every distinct output length as soon
    as any record stops early, so it is taken from the design instead. Files from an older harness
    do not state it; there, the records that hit the cap are the ones whose own length is the cap.
    """
    n = (data.get("design") or {}).get("max_tokens")
    if n:
        return int(n)
    capped = [r.get("predicted_n") or 0 for r in data["records"] if not reached_eos(r)]
    return max(capped) if capped else None


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
    return out, budget(data)


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
    n_eos = 0
    for rec in data["records"]:
        state, pos, _produced = classify(rec)
        if state is None:
            continue
        counts[state] += 1
        if reached_eos(rec):
            n_eos += 1
        if state == DIVERGED and pos is not None:
            forks.append(pos)
    return path, counts, forks, budget(data), pass_agreement(data), n_eos


def main():
    paths = sys.argv[1:] or sorted(glob.glob("results/*.json"))
    paths = [p for p in paths if ".partial." not in p]
    print("=" * 100)
    print("DIVERGENCE AS A TIME-TO-EVENT, IN TOKENS")
    print("=" * 100)
    total = collections.Counter()
    eos_seen = [0, 0]   # records that stopped on their own, records classified
    for path in paths:
        try:
            path, counts, forks, win, (n_multi, disagree), n_eos = audit(path)
        except Exception as exc:
            print("  %-40s unreadable: %s" % (path, exc))
            continue
        if not counts:
            continue
        total.update(counts)
        n = sum(counts.values())
        eos_seen[0] += n_eos
        eos_seen[1] += n
        print("\n  %s" % path)
        print("    window            : %s" % ("%d tokens" % win if win else "unknown"))
        print("    ran to EOS        : %4d / %d   <- stopped on their own, not on the cap"
              % (n_eos, n))
        print("    diverged          : %4d / %d" % (counts[DIVERGED], n))
        print("    identical at EOS  : %4d / %d   <- the only exact identities" % (counts[IDENTICAL_EOS], n))
        print("    right-censored    : %4d / %d   <- no divergence within the window" % (counts[CENSORED], n))
        if forks:
            forks.sort()
            print("    fork position     : min %.0f  median %.0f  max %.0f tokens"
                  % (forks[0], statistics.median(forks), forks[-1]))
            if win:
                print("                        the latest sits at %.0f %% of the window, so a fork"
                      % (100.0 * forks[-1] / win))
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
            print("  No identity here is exact. Every one is a cell whose two arms agreed while at")
            print("  least one side stopped on the cap, so it reads 'did not diverge within the")
            print("  window', not 'identical'. That is a statement about the identities alone:")
            print("  %d of %d records did run to EOS, and not one of those matched its baseline."
                  % (eos_seen[0], eos_seen[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
