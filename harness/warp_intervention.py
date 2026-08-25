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
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel_facts as KF  # noqa: E402
import quality  # noqa: E402
import truncation_audit as TA  # noqa: E402

# arm name -> verification width. Width is n_max + 1; the greedy baseline runs at width 1.
WIDTH = {"baseline": 1, "mtp-n2": 3, "mtp-n3": 4, "mtp-n4": 5, "mtp-n5": 6, "mtp-n7": 8}

# The GENERIC table each build was compiled with. Keep in step with run_warp.sh, which writes the
# patched table to warp/<build>/table.txt; validate_tables() checks these against those files when
# they are reachable.
TABLES = {
    "control":      {1: 4, 3: 4, 4: 4, 5: 2, 6: 2, 8: 2},
    "control2":     {1: 4, 3: 4, 4: 4, 5: 2, 6: 2, 8: 2},
    "forced_up":    {1: 4, 3: 4, 4: 4, 5: 4, 6: 4, 8: 4},
    "forced_down":  {1: 2, 3: 2, 4: 2, 5: 2, 6: 2, 8: 2},
    # The v2 direction. It differs from forced_down at width 1 and nowhere else that any arm
    # runs, and width 1 is the greedy baseline: leaving it at four warps is the whole reason
    # this build exists, because the first forced_down moved it and so could not have the
    # baseline as its control. Scoring this file against the forced_down row would take the
    # branch that says the baseline is part of the intervention, and skip the one gate the
    # rebuild was for.
    "forced_down2": {1: 4, 3: 2, 4: 2, 5: 2, 6: 2, 8: 2},
}


def build_of(path):
    """Which build a result file holds, from its name rather than its position on the argv.

    Longest name first: "forced_down2" contains "forced_down" and "control2" contains "control",
    so a shortest-match would silently label the v2 file as v1 and score it against a table it
    was not built with.
    """
    name = os.path.basename(path)
    for tag in sorted(TABLES, key=len, reverse=True):
        if tag in name:
            return tag
    return None


def validate_tables(builds, dirs=("upstream/llamacpp", ".")):
    """Check each assumed table against the source the build was compiled from.

    The runs save the patched GENERIC block next to the binary and the collector brings it home
    as warp_builds_v2_<build>_table.txt. Reading it back is the only thing that can catch this
    file's table having drifted from the one that was built, which is not hypothetical: the v2
    run introduced a fourth build and this table was not updated for it.

    -> list of complaints, empty when everything reachable agrees. A table.txt that is not
    present is reported as unchecked rather than assumed correct.
    """
    problems, checked, missing = [], [], []
    for b in builds:
        found = None
        for d in dirs:
            for pat in (f"warp_builds_v2_{b}_table.txt", f"warp_builds_{b}_table.txt"):
                cand = os.path.join(d, pat)
                if os.path.exists(cand):
                    found = cand
                    break
            if found:
                break
        if not found:
            missing.append(b)
            continue
        try:
            src = open(found, encoding="utf-8", errors="replace").read()
        except OSError as e:
            problems.append(f"{b}: {found} unreadable ({e})")
            continue
        parsed = KF.parse_generic_table(src)
        if not parsed:
            problems.append(f"{b}: {found} did not parse as a case/return table")
            continue
        want = TABLES[b]
        bad = {w: (want[w], parsed.get(w)) for w in want if parsed.get(w) != want[w]}
        if bad:
            problems.append(
                f"{b}: this file assumes {want} but {found} was compiled with "
                + ", ".join(f"width {w}: assumed {a}, built {g}" for w, (a, g) in sorted(bad.items())))
        else:
            checked.append(b)
    return problems, checked, missing


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
            forks[rec["prompt"]][WIDTH[rec["arm"]]] = quality.fork_cell(
                div, same="same", prefix="prefix")
    return by_key, forks, data


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


def score(tag, control_forks, forced_forks, control_by, forced_by, table,
          control_raw, forced_raw):
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
    if table[1] != stock[1]:
        # PREREGISTRATION.md says the baseline "runs at width 1, which the table maps to 4 warps
        # in all three builds, so its output must be byte-identical across the three". The
        # forced-down table it specifies in the row above is 1-4 -> 2, which includes width 1, so
        # the control it registers cannot hold for that build by construction. Reporting this as
        # a failure would blame the measurement for a contradiction in the design.
        print("    [N/A ] greedy baseline byte-identical across builds")
        print("           This build maps width 1 to %d warps against the stock %d, so the "
              "baseline is" % (table[1], stock[1]))
        print("           part of the intervention and cannot also be its control. Observed: "
              "identical %d, differs %d." % (b_same, b_diff))
        print("           PREREGISTRATION.md asserts width 1 is at 4 warps in all three builds "
              "while also")
        print("           specifying 1-4 -> 2 here. Both cannot be true; the table is what was "
              "built.")
        ok_base = True
    else:
        ok_base = gate(
            "greedy baseline byte-identical across builds",
            b_diff == 0 and b_same > 0,
            "width 1 is at %d warps in both builds; identical %d, differs %d"
            % (table[1], b_same, b_diff),
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
        if not ok_clean:
            print("\n  This is the fourth registered outcome: \"The forced build changes fork")
            print("  positions for widths it did not touch. The intervention is not clean;")
            print("  something else in the build differs, and nothing is concluded until that is")
            print("  found.\" Widths %s carry the stock warp count in both builds and still" % untouched)
            print("  produce different output, so the warp count is not the only thing the edit")
            print("  changed. Control and forced-up agree byte for byte at every width forced-up")
            print("  left alone, so the build process itself is deterministic and the effect is")
            print("  specific to this edit.")
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
    split = [p for p in full
             if len({control_forks[p][w] for w in low}) == 1
             and len({control_forks[p][w] for w in high}) == 1
             and control_forks[p][3] != control_forks[p][5]]

    # A "same" here means "did not diverge within the token budget", because every generation
    # stopped at the cap. That is uniform across prompts and widths, so there is no cleaner subset
    # to score against; an earlier version scored two denominators on a distinction that came from
    # measuring the window in characters. The window is stated and the count is one number.
    censored, window = TA.censored_prompts(control_raw)
    cens_f, _ = TA.censored_prompts(forced_raw)
    informative = split

    print("\n  --- discriminating set ---")
    print("    prompts with all five widths present      : %d" % len(full))
    print("    of those, control shows the two-group split: %d" % len(split))
    if censored or cens_f:
        print("    every 'same' is right-censored at the %s-token cap, uniformly, so this is a"
              % window)
        print("    statement about the first %s tokens and not about identity." % window)

    # The registered prediction: the forced widths adopt the fork positions of the widths whose
    # warp count they were given.
    target = high if tag == "forced_up" else low
    source = low if tag == "forced_up" else high

    def tally(prompts):
        followed = moved_other = unmoved = 0
        for p in prompts:
            c, f = control_forks[p], forced_forks[p]
            anchor = f[source[0]]
            if all(f[w] == anchor for w in target):
                followed += 1
            elif any(f[w] != c[w] for w in target):
                moved_other += 1
            else:
                unmoved += 1
        return followed, moved_other, unmoved

    print("\n  --- registered prediction: widths %s adopt the %s fork position ---" % (target, source))
    if not informative:
        print("    no prompt shows the two-group split; nothing to score")
        return
    followed, moved_other, unmoved = tally(informative)
    n = len(informative)
    print("      followed the forced warp count    : %3d/%d  (%.0f %%)" % (followed, n, 100.0*followed/n))
    print("      moved, but not onto %-8s      : %3d/%d  (%.0f %%)" % (str(source), moved_other, n, 100.0*moved_other/n))
    print("      did not move                      : %3d/%d  (%.0f %%)" % (unmoved, n, 100.0*unmoved/n))

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


def guard(control_by, control2_by):
    """control against control2: two builds from one configure with no source difference.

    Checked before anything is scored, not after. The collector printed this underneath the
    verdict, with the note that nothing above should be read - which asks a reader to discard a
    conclusion they have already formed. A failure here voids every comparison in the set, so it
    belongs first.
    """
    ks = sorted(set(control_by) & set(control2_by))
    same = sum(1 for k in ks if text_of(control_by[k]) == text_of(control2_by[k]))
    print("\n  --- the guard: two builds, one configure, no source difference ---")
    print("    control against control2: %d/%d byte-identical" % (same, len(ks)))
    if not ks:
        print("    no shared records; the guard could not run")
        return False
    if same != len(ks):
        print("    THE GUARD FAILED. Two builds that differ in nothing produced different output,")
        print("    so nothing in this set is a measurement of the table. Not scoring.")
        return False
    print("    The guard holds, so a difference between control and a forced build is the table.")
    return True


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    # Named by file, never by position. The collector passes forced_down2 third, where an
    # earlier version of this file put forced_down, and the two differ exactly at the width
    # that decides whether the baseline gate applies.
    paths = {}
    for arg in sys.argv[1:]:
        tag = build_of(arg)
        if tag is None:
            print("  cannot tell which build %r holds from its name; refusing to guess" % arg)
            return 2
        if tag in paths:
            print("  two files claim to be %s: %s and %s" % (tag, paths[tag], arg))
            return 2
        paths[tag] = arg

    print("=" * 92)
    print("FORCED-WARP INTERVENTION")
    print("=" * 92)
    for tag in sorted(paths):
        print("  %-13s : %s" % (tag, paths[tag]))

    problems, checked, missing = validate_tables(sorted(paths))
    print("\n  --- tables verified against the source each build was compiled from ---")
    if checked:
        print("    verified: %s" % ", ".join(checked))
    if missing:
        print("    no table.txt found for %s; assumed, not checked" % ", ".join(missing))
    if problems:
        for pr in problems:
            print("    MISMATCH %s" % pr)
        print("    The table this file assumes is not the table that was built. Every prediction")
        print("    below would be scored against the wrong intervention. Refusing to score.")
        return 1

    loaded = {}
    for tag, path in paths.items():
        try:
            loaded[tag] = load(path)
        except FileNotFoundError:
            print("  %s: %s not present; skipping that direction" % (tag, path))

    if "control" not in loaded:
        print("  control build is required")
        return 1
    c_by, c_forks, c_raw = loaded["control"]
    print("\n  control holds %d records" % len(c_by))

    if "control2" in loaded:
        if not guard(c_by, loaded["control2"][0]):
            return 1
    else:
        print("\n  --- the guard ---")
        print("    control2 was not supplied. Two builds from one configure and no source")
        print("    difference is the only check that separates the table from the build, and")
        print("    without it a difference below could be either. Reported, not assumed away.")

    directions = [t for t in ("forced_up", "forced_down", "forced_down2") if t in loaded]
    if not directions:
        print("\n  no forced build supplied; nothing to score")
        return 1
    for tag in directions:
        f_by, f_forks, f_raw = loaded[tag]
        score(tag, c_forks, f_forks, c_by, f_by, TABLES[tag], c_raw, f_raw)

    if len(directions) < 2:
        print("\n%s" % ("=" * 92))
        print("  One direction only. The registered design needs both: a single direction cannot")
        print("  separate the warp count from an accidental one-way effect. No overall verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
