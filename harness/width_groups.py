"""Group speculative arms by the fork position they produce, and check it against the CUDA table.

Written to answer one question posted publicly in llama.cpp #25618 (comment 5396293373), and
written before the data exists so the verdict is not shaped by it.

The claim there is that on sm_86 the fork position of greedy speculative output is a function of
verification width, and that `calc_nwarps` in `ggml/src/ggml-cuda/mmvq.cu` changes at the same
width. That table gives 4 warps up to `ncols_dst` 4, 2 warps from 5 to 8, and 1 above 8, so the
prediction registered as H8 is three groups: {2,3,4}, {5,6,7,8} and {9} alone.

H8a is separate and is about onset rather than grouping: does width 2, which is
`--spec-draft-n-max 1`, diverge at all on CUDA? On Vulkan it does, per frizikk's operator bisect
and Ankk98's original report. If it does not here, the two backends differ on where divergence
starts and that matters more than the grouping.

Two arms at the same width but with different drafters must agree, or the width account fails.
`phase_nmax` runs both at widths 3, 5, 7 and 9, so that control is free.

Usage:  python3 harness/width_groups.py results/phase_nmax.json
"""
import collections
import json
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import completeness as _CO  # noqa: E402
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import truncation_audit as TA  # noqa: E402

# ncols_dst -> warps, MMVQ_PARAMETERS_GENERIC, which is what sm_86 falls through to.
#
# This is the STOCK table. The forced-warp builds ship a different one, so on those files every
# line below that mentions warps would state the stock prediction for a build that does not have
# it, and the H8 verdict would report an agreement that was never tested. is_intervention_file()
# gates that, and harness/warp_intervention.py is what scores those runs.
# ggml/src/ggml-cuda/mmvq.cu. A batch wider than this never reaches MMVQ at all, so the table
# says nothing about it and neither can this file.
MMVQ_MAX_BATCH_SIZE = 8



def recorded_mmvq_max(d, fallback=MMVQ_MAX_BATCH_SIZE):
    """The dispatch limit this result was actually produced under, if the file records it.

    Falls back to the constant above and says so, because a run from before harness/kernel_facts.py
    existed carries no such record and silently using today's value is how an analyser starts
    describing a build it never saw.
    """
    facts = ((d.get("design") or {}).get("kernel_facts") or {})
    seen = {t.get("mmvq", {}).get("mmvq_max_batch_size") for t in facts.values()}
    seen.discard(None)
    if len(seen) == 1:
        return next(iter(seen)), True
    return fallback, False


def warps_for(width: int) -> int | None:
    """Warps for a width, or None when the width does not run through MMVQ.

    The `default: return 1` arm of calc_nwarps is unreachable for a verification batch: MMVQ is
    dispatched only up to MMVQ_MAX_BATCH_SIZE, and a wider batch takes a different kernel family
    entirely. Returning 1 for width 9 put it in a warp group of its own and let H8 be scored
    against a prediction the table never made. phase_nmax runs widths 2 through 9, so this was
    live rather than hypothetical.
    """
    if width > MMVQ_MAX_BATCH_SIZE:
        return None
    if width <= 4:
        return 4
    return 2


def warp_label(group) -> str:
    """How to describe a group's warp count, given some widths have none."""
    counts = {warps_for(w) for w in group}
    if counts == {None}:
        return "n/a (off the MMVQ path)"
    known = sorted(x for x in counts if x is not None)
    return str(known) + (" + off-path widths" if None in counts else "")


def is_intervention_file(d: dict, path: str) -> bool:
    """True for a forced_up / forced_down result, whose GENERIC table is not the stock one.

    The arms carry tree "warp" in all three builds including the control, so the tree alone does
    not separate them; the filename is what names the build.
    """
    warp_tree = any((a or {}).get("tree") == "warp" for a in (d.get("arms") or {}).values())
    return warp_tree and ("forced_up" in path or "forced_down" in path)


def width_of(arm: str, meta: dict) -> int | None:
    ea = (meta or {}).get("extra_args") or []
    if "--spec-draft-n-max" not in ea:
        return None
    return int(ea[ea.index("--spec-draft-n-max") + 1]) + 1


def spec_of(arm: str, meta: dict) -> str:
    ea = (meta or {}).get("extra_args") or []
    return ea[ea.index("--spec-type") + 1] if "--spec-type" in ea else "?"


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "results/phase_nmax.json"
    d = json.load(open(path))
    _CO.warn_if_incomplete(d, path)
    arms = d.get("arms", {})

    if is_intervention_file(d, path):
        print(f"{path} is a forced-warp build. Its GENERIC table is not the stock one, so every")
        print("warp count printed here would be wrong and the H8 verdict would compare the")
        print("observation against a prediction the build never made.")
        print("\nUse:  python3 harness/warp_intervention.py control.json forced_up.json forced_down.json")
        return 2

    # (width, spec_type) -> {prompt -> fork position or "same"}, first pass only; divergence
    # proved perfectly reproducible across passes in Phase A, 125 of 125.
    cell: dict[tuple, dict] = collections.defaultdict(dict)
    for r in d["records"]:
        w = width_of(r["arm"], arms.get(r["arm"]))
        if w is None or r["pass"] != 1 or not r.get("divergence"):
            continue
        v = r["divergence"]
        cell[(w, spec_of(r["arm"], arms.get(r["arm"])))][r["prompt"]] = (
            "same" if v["identical"] else v["first_diff_char"])
    if not cell:
        print(f"no divergence records with a width in {path}")
        return 1

    widths = sorted({w for w, _ in cell})
    prompts = sorted(set().union(*[set(v) for v in cell.values()]))

    # "same" is one of the values a width's signature vector can take, and it means "did not
    # diverge within the token budget" rather than "identical" whenever the generation stopped at
    # the cap. Every arm gets the same budget, so that censoring is uniform across prompts and
    # widths: there is no cleaner subset to check the rest against, and an earlier version of this
    # file that built one was measuring the window in characters, which the design does not hold
    # constant. The window is stated instead, and resolving it needs a larger budget.
    censored, window = TA.censored_prompts(d)

    print("=" * 92)
    print("FORK POSITION BY VERIFICATION WIDTH")
    print("=" * 92)
    print(f"  source: {path}")
    print(f"  widths: {widths}")
    if censored:
        forks = TA.resolved_forks(d)
        latest = max(forks) if forks else 0
        print(f"  every 'same' below is right-censored: the generation stopped at the "
              f"{window}-token cap,")
        print(f"  so it means 'did not diverge within {window} tokens', not 'identical'. Forks here "
              f"have been")
        print(f"  resolved as late as token {latest:.0f} ({100.0*latest/window:.0f} % of the window) "
              f"when {window} is the cap.")
        print(f"  {len(censored)} of {len(prompts)} prompts carry at least one. The censoring is "
              f"uniform, so no")
        print(f"  subset of prompts is cleaner than another; TODO.md D2 is the run that resolves it.")
    _mm, _rec = recorded_mmvq_max(d)
    print("  MMVQ dispatch limit %d, %s" % (
        _mm, "read from this run's own record" if _rec
        else "from the analyser's constant: this file predates harness/kernel_facts.py"))
    print(f"  MMVQ_PARAMETERS_GENERIC warps: "
          + "  ".join(f"w{w}=" + ("n/a" if warps_for(w) is None else str(warps_for(w)))
                      for w in widths))
    off_path = [w for w in widths if warps_for(w) is None]
    if off_path:
        print(f"  widths {off_path} exceed MMVQ_MAX_BATCH_SIZE ({MMVQ_MAX_BATCH_SIZE}) and take a")
        print("  different kernel family, so the table makes no prediction for them.")

    # -------------------------------------------------- control: drafters must agree at a width
    print("\n--- control: two drafters at the same width must agree ---")
    shared = 0
    control_failed = []
    for w in widths:
        specs = sorted({s for ww, s in cell if ww == w})
        if len(specs) < 2:
            continue
        shared += 1
        a, b = cell[(w, specs[0])], cell[(w, specs[1])]
        common = sorted(set(a) & set(b))
        same = sum(1 for p in common if a[p] == b[p])
        if same != len(common):
            control_failed.append(w)
        flag = "" if same == len(common) else "   <-- DISAGREE, the width account fails here"
        print(f"    w={w}  {specs[0]} vs {specs[1]}: {same}/{len(common)} prompts agree{flag}")
    if not shared:
        print("    only one drafter per width in this file; control not available")

    # -------------------------------------------------- the grouping
    # A width's signature is its whole vector of fork positions across prompts. Two widths are in
    # the same group when those vectors are identical, not merely similar.
    sig: dict[int, tuple] = {}
    for w in widths:
        specs = sorted({s for ww, s in cell if ww == w})
        merged = {}
        for s in specs:
            merged.update(cell[(w, s)])
        sig[w] = tuple(merged.get(p, "?") for p in prompts)

    # A width that never diverges has no fork position to compare, so it cannot be grouped by
    # one. Folding it in would report it as its own group and read as the partition failing,
    # when what happened is that the width had nothing to partition.
    lossless = [w for w in widths if all(v in ("same", "?") for v in sig[w])]
    gradable = [w for w in widths if w not in lossless]
    if lossless:
        print(f"\n--- widths that never diverge: {set(lossless)} ---")
        print("    Excluded from the grouping below: with no fork position there is nothing to")
        print("    group by. Reported here instead, and it is the H8a question for width 2.")

    groups: dict[tuple, list] = collections.OrderedDict()
    for w in gradable:
        groups.setdefault(sig[w], []).append(w)
    observed = [tuple(v) for v in groups.values()]

    print("\n--- observed groups (identical fork vector across every prompt) ---")
    for g in observed:
        print(f"    {set(g)}  warps {warp_label(g)}")

    # -------------------------------------------------- verdict on H8
    untestable = [w for w in gradable if warps_for(w) is None]
    testable = [w for w in gradable if warps_for(w) is not None]
    pred: dict[int, list] = collections.OrderedDict()
    for w in testable:
        pred.setdefault(warps_for(w), []).append(w)
    predicted = [tuple(v) for v in pred.values()]
    obs_testable = [tuple(w for w in g if w in testable) for g in observed]
    obs_testable = [g for g in obs_testable if g]
    print("\n--- H8: does the grouping follow calc_nwarps? ---")
    if lossless:
        print(f"    computed over the diverging widths only, {sorted(gradable)}")
    if untestable:
        print(f"    NOT TESTABLE at widths {untestable}: they exceed MMVQ_MAX_BATCH_SIZE and do not")
        print("    share the MMVQ execution path, so calc_nwarps makes no prediction for them.")
        print(f"    Scored on the widths that do share it: {testable}")
        observed = obs_testable
    print(f"    predicted from the table: {[set(g) for g in predicted]}")
    print(f"    observed:                 {[set(g) for g in observed]}")
    # A control failure at a width the table does not cover cannot withhold a verdict about the
    # widths it does. It is reported either way, because two drafters disagreeing is a finding in
    # its own right: at width 9 they share only the verification width and no longer agree, which
    # is the same MMVQ boundary that shows up in the cost fit and in the throughput.
    failed_testable = [w for w in control_failed if warps_for(w) is not None]
    failed_offpath = [w for w in control_failed if warps_for(w) is None]
    if failed_offpath:
        print(f"    Two drafters disagree at width(s) {set(failed_offpath)}, which are off the MMVQ")
        print("    path. That does not bear on the widths below, and is itself the boundary showing")
        print("    up in the control as well as in the cost fit.")
    if failed_testable:
        print(f"    VERDICT WITHHELD. Two drafters give different fork positions at width(s) "
              f"{set(failed_testable)}, so fork position is not a function of width alone and the")
        print("    grouping below was built by letting one drafter overwrite the other. Nothing")
        print("    here supports or refutes H8 until that is resolved, and llama.cpp #25618 needs")
        print("    telling either way.")
    if failed_testable:
        pass
    elif set(map(frozenset, observed)) == set(map(frozenset, predicted)):
        print("    H8 SUPPORTED. The partition is exactly the warp-count table.")
    else:
        print("    H8 NOT SUPPORTED. The grouping tracks something other than the warp count,")
        print("    and the mechanism offered in llama.cpp #25618 needs withdrawing there.")
        for g in observed:
            if len({warps_for(w) for w in g if warps_for(w) is not None}) > 1:
                print(f"      widths {set(g)} share a fork vector but not a warp count")

    # -------------------------------------------------- verdict on H8a
    print("\n--- H8a: does width 2 diverge at all on CUDA? ---")
    w2 = {p: v for (w, _), c in cell.items() if w == 2 for p, v in c.items()}
    if not w2:
        print("    width 2 not present in this file")
    else:
        div = sum(1 for v in w2.values() if v != "same")
        print(f"    width 2 diverges on {div} of {len(w2)} prompts")
        if div:
            print("    H8a SUPPORTED. CUDA and Vulkan agree on where divergence begins.")
        else:
            print("    H8a FALSIFIED. Width 2 is byte-identical here while Vulkan diverges at the")
            print("    same setting. That is a backend difference, it is worth more than the")
            print("    grouping, and llama.cpp #25618 should be told promptly.")

    # -------------------------------------------------- the competing explanation
    print("\n--- competing account: is the grouping just an acceptance threshold? ---")
    accs = {}
    for r in d["records"]:
        w = width_of(r["arm"], arms.get(r["arm"]))
        tm = r.get("timings") or {}
        if w is None or not tm.get("t_draft_n"):
            continue
        accs.setdefault(w, []).append((tm.get("t_draft_n_accepted") or 0) / tm["t_draft_n"])
    if len(accs) < 3:
        print("    not enough widths with acceptance data")
    else:
        import statistics as _st
        mean_acc = {w: _st.fmean(v) for w, v in accs.items()}
        ws = sorted(mean_acc)
        grp_of = {}
        for gi, g in enumerate(observed):
            for w in g:
                grp_of[w] = gi
        rows_ = []
        for a, b in zip(ws, ws[1:]):
            if a in grp_of and b in grp_of:
                rows_.append((abs(mean_acc[b] - mean_acc[a]), a, b, grp_of[a] != grp_of[b]))
        for gap, a, b, split in sorted(rows_, reverse=True):
            print(f"    w={a} -> w={b}: acceptance gap {gap:.4f}   "
                  + ("GROUP BOUNDARY" if split else "no boundary"))
        if rows_:
            biggest = max(rows_)
            if biggest[3]:
                print("    The largest acceptance gap is also a group boundary, so an acceptance")
                print("    threshold explains the split at least as well as the warp table does.")
            else:
                print(f"    The largest gap, w={biggest[1]} to w={biggest[2]}, is not a boundary,")
                print("    so a single acceptance threshold does not pick out the observed split.")

    # -------------------------------------------------- the table itself
    print("\n--- fork position per prompt ---")
    print(f"    {'prompt':26s}" + "".join(f"{'w=' + str(w):>8s}" for w in widths))
    for p in prompts:
        row = ""
        for w in widths:
            merged = {}
            for s in sorted({s for ww, s in cell if ww == w}):
                merged.update(cell[(w, s)])
            row += f"{str(merged.get(p, '-')):>8s}"
        print(f"    {p:26s}{row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
