#!/usr/bin/env python3
"""Scores the forced-warp intervention against the outcomes registered in PREREGISTRATION.md.

The observational finding is that fork positions fall into two groups, {3,4} and {5,6,8}, and
that the split sits exactly where `calc_nwarps` changes the GENERIC table from four warps to two.
Consistency is not causation: anything that changes at the same width fits the same data. The
intervention edits the table and nothing else, so the warp count moves while the width stays.

Three builds from revision c060ca9:

    control      1-4 -> 4,  5-8 -> 2     stock
    forced_up    1-4 -> 4,  5-8 -> 4     high widths get the low-width warp count
    forced_down  1-4 -> 2,  5-8 -> 2     low widths get the high-width warp count

`width_groups.py` hard-codes the stock table, which is correct for an ordinary run and wrong for
these files: it would print the stock prediction for a build that no longer has the stock table
and report agreement that was never tested. The table is a property of the build, so it is passed
in here per file rather than assumed.

Usage:
    warp_intervention.py control.json forced_up.json [forced_down.json]
"""

import collections
import json
import sys

# arm name -> verification width. Width is n_max + 1; the greedy baseline runs at width 1.
WIDTH = {"baseline": 1, "mtp-n2": 3, "mtp-n3": 4, "mtp-n4": 5, "mtp-n5": 6, "mtp-n7": 8}

# The GENERIC table each build was compiled with. Keep in step with run_warp.sh, which writes the
# patched table to warp/<build>/table.txt; validate_tables() checks these against those files when
# they are reachable.
TABLES = {
    "control":     {1: 4, 3: 4, 4: 4, 5: 2, 6: 2, 8: 2},
    "forced_up":   {1: 4, 3: 4, 4: 4, 5: 4, 6: 4, 8: 4},
    "forced_down": {1: 2, 3: 2, 4: 2, 5: 2, 6: 2, 8: 2},
}


def load(path):
    """(arm, prompt, pass) -> record, plus the fork position of every speculative arm.

    The fork position is the record's own divergence against the greedy baseline of the same
    file. It is not comparable across files as a raw number unless the builds agree, which is
    exactly what the baseline gate below establishes.
    """
    with open(path) as fh:
        data = json.load(fh)
    by_key = {}
    forks = collections.defaultdict(dict)
    for rec in data["records"]:
        by_key[(rec["arm"], rec["prompt"], rec["pass"])] = rec
        if rec["arm"] in WIDTH and WIDTH[rec["arm"]] != 1 and rec.get("divergence"):
            div = rec["divergence"]
            forks[rec["prompt"]][WIDTH[rec["arm"]]] = "same" if div["identical"] else div["first_diff_char"]
    return by_key, forks


def text_of(rec):
    for key in ("text", "generated", "output"):
        if rec.get(key) is not None:
            return rec[key]
    return None


def compare_texts(base, other, widths):
    """How many records are byte-identical between two builds, restricted to `widths`."""
    same = diff = 0
    for key, rec in base.items():
        if WIDTH.get(key[0]) not in widths or key not in other:
            continue
        a, b = text_of(rec), text_of(other[key])
        if a is None or b is None:
            continue
        if a == b:
            same += 1
        else:
            diff += 1
    return same, diff


def gate(name, ok, detail):
    print("    [%s] %s" % ("PASS" if ok else "FAIL", name))
    print("           %s" % detail)
    return ok


def score(tag, control_forks, forced_forks, control_by, forced_by, table):
    """Score one forced build against its registered prediction."""
    stock = TABLES["control"]
    touched = sorted(w for w in table if w != 1 and table[w] != stock[w])
    untouched = sorted(w for w in table if w != 1 and w not in touched)

    print("\n%s" % ("=" * 92))
    print("%s : GENERIC table %s" % (tag.upper(), ", ".join("w%d->%d" % (w, table[w]) for w in sorted(table))))
    print("%s" % ("=" * 92))
    print("    widths whose warp count changed : %s" % touched)
    print("    widths left at the stock count  : %s" % untouched)

    print("\n  --- validity gates (registered: a failure here voids the comparison) ---")
    b_same, b_diff = compare_texts(control_by, forced_by, {1})
    ok_base = gate(
        "greedy baseline byte-identical across builds",
        b_diff == 0 and b_same > 0,
        "width 1 maps to %d warps in both builds; identical %d, differs %d" % (table[1], b_same, b_diff),
    )
    u_same, u_diff = compare_texts(control_by, forced_by, set(untouched))
    ok_clean = gate(
        "intervention did not touch widths it should not have",
        u_diff == 0,
        "widths %s: identical %d, differs %d" % (untouched, u_same, u_diff),
    )
    t_same, t_diff = compare_texts(control_by, forced_by, set(touched))
    ok_effect = gate(
        "intervention had a measurable effect where it was applied",
        t_diff > 0,
        "widths %s: identical %d, differs %d" % (touched, t_same, t_diff),
    )
    if not (ok_base and ok_clean):
        print("\n  COMPARISON VOID. Not scoring the prediction.")
        return
    if not ok_effect:
        print("\n  The build differs but produced no output change at the widths it edited.")
        print("  That is not the registered 'neither moves': it means the edited table row never")
        print("  reached the kernel. Resolve that before reading anything else here.")
        return

    # Only prompts where the control build actually shows the two-group split can discriminate.
    # A prompt whose five widths already share one fork position is consistent with every
    # hypothesis and would inflate agreement if counted.
    full = [p for p in control_forks if len(control_forks[p]) == 5 and len(forced_forks.get(p, {})) == 5]
    low, high = [3, 4], [5, 6, 8]
    informative = [p for p in full
                   if len({control_forks[p][w] for w in low}) == 1
                   and len({control_forks[p][w] for w in high}) == 1
                   and control_forks[p][3] != control_forks[p][5]]

    print("\n  --- discriminating set ---")
    print("    prompts with all five widths present      : %d" % len(full))
    print("    of those, control shows the two-group split: %d   <- the only informative ones" % len(informative))

    # The registered prediction: the forced widths adopt the fork positions of the widths whose
    # warp count they were given.
    target = high if tag == "forced_up" else low
    source = low if tag == "forced_up" else high
    followed = moved_other = unmoved = 0
    for p in informative:
        c, f = control_forks[p], forced_forks[p]
        anchor = f[source[0]]
        if all(f[w] == anchor for w in target):
            followed += 1
        elif any(f[w] != c[w] for w in target):
            moved_other += 1
        else:
            unmoved += 1

    print("\n  --- registered prediction: widths %s adopt the %s fork position ---" % (target, source))
    if not informative:
        print("    no informative prompts; nothing to score")
        return
    pct = lambda n: 100.0 * n / len(informative)
    print("    followed the forced warp count      : %3d/%d  (%.0f %%)" % (followed, len(informative), pct(followed)))
    print("    moved, but not onto %-8s        : %3d/%d  (%.0f %%)" % (str(source), moved_other, len(informative), pct(moved_other)))
    print("    did not move                        : %3d/%d  (%.0f %%)" % (unmoved, len(informative), pct(unmoved)))

    print("\n  --- verdict for this direction ---")
    if followed == len(informative):
        print("    The forced widths followed the warp count on every informative prompt.")
    elif followed == 0:
        print("    The forced widths never adopted the %s position. The warp count changed the" % source)
        print("    numerics - %d of the touched records differ - but it did not carry the fork" % t_diff)
        print("    positions with it, so it is not what puts the widths into two groups.")
    else:
        print("    Split result: the prediction held on %d of %d informative prompts and failed on" % (followed, len(informative)))
        print("    the other %d. Registered as unresolved; no account was prepared for a partial" % (len(informative) - followed))
        print("    effect and none is invented here.")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    paths = {"control": sys.argv[1], "forced_up": sys.argv[2]}
    if len(sys.argv) > 3:
        paths["forced_down"] = sys.argv[3]

    loaded = {}
    for tag, path in paths.items():
        try:
            loaded[tag] = load(path)
        except FileNotFoundError:
            print("  %s: %s not present yet; skipping that direction" % (tag, path))

    if "control" not in loaded:
        print("  control build is required")
        return 1
    c_by, c_forks = loaded["control"]

    print("=" * 92)
    print("FORCED-WARP INTERVENTION")
    print("=" * 92)
    print("  control     : %s (%d records)" % (paths["control"], len(c_by)))
    for tag in ("forced_up", "forced_down"):
        if tag in loaded:
            print("  %-11s : %s (%d records)" % (tag, paths[tag], len(loaded[tag][0])))
        else:
            print("  %-11s : not yet run" % tag)

    for tag in ("forced_up", "forced_down"):
        if tag in loaded:
            f_by, f_forks = loaded[tag]
            score(tag, c_forks, f_forks, c_by, f_by, TABLES[tag])

    if "forced_down" not in loaded:
        print("\n%s" % ("=" * 92))
        print("  One direction only. The registered design needs both: a single direction cannot")
        print("  separate the warp count from an accidental one-way effect. No overall verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
