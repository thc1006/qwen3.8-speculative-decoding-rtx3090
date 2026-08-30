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
import quality  # noqa: E402
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import speclen  # noqa: E402
import statistics as _stats  # noqa: E402
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
    _CO.require_complete(d, path)
    arms = d.get("arms", {})

    if is_intervention_file(d, path):
        print(f"{path} is a forced-warp build. Its GENERIC table is not the stock one, so every")
        print("warp count printed here would be wrong and the H8 verdict would compare the")
        print("observation against a prediction the build never made.")
        print("\nUse:  python3 harness/warp_intervention.py control.json forced_up.json forced_down.json")
        return 2

    # (width, spec_type) -> {prompt -> fork position or "same"}, first pass only; divergence
    # proved perfectly reproducible across passes in Phase A, 150 of 150. (This said 125, which
    # is the extended-cap re-run's count over its three passes, not Phase A's over five.)
    cell: dict[tuple, dict] = collections.defaultdict(dict)
    for r in d["records"]:
        w = width_of(r["arm"], arms.get(r["arm"]))
        if w is None or r["pass"] != 1 or not r.get("divergence"):
            continue
        v = r["divergence"]
        cell[(w, spec_of(r["arm"], arms.get(r["arm"])))][r["prompt"]] = (
            quality.fork_cell(v, same="same", prefix="prefix"))
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
        print(f"  resolved as late as roughly token {latest:.0f} ({100.0*latest/window:.0f} % of the window) "
              f"when {window} is the cap.")
        print(f"  {len(censored)} of {len(prompts)} prompts carry at least one. The censoring is "
              f"uniform across")
        print(f"  widths, so no subset of prompts is cleaner than another. The grouping below is "
              f"built only from")
        print(f"  cells where both widths diverged, so a censored cell narrows the evidence for a "
              f"grouping")
        print(f"  without moving it.")
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
    # Effective width per (asked width, drafter): one plus the drafts actually proposed per
    # verification step. Recorded here rather than assumed from the flag, because the two
    # differ by a full column at the top of this ladder.
    _acc = collections.defaultdict(list)
    for r in d["records"]:
        tm = r.get("timings") or {}
        dn = tm.get("t_draft_n") or 0
        fw = speclen.forwards(r)
        if not dn or not fw:
            continue
        am = arms.get(r["arm"]) or {}
        ww, sp = width_of(r["arm"], am), spec_of(r["arm"], am)
        if ww:
            _acc[(ww, sp)].append(dn / fw + 1.0)
    eff_width = {k: _stats.median(v) for k, v in _acc.items() if v}

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
        # The control asks whether two drafters AT THE SAME WIDTH agree. n_max is what was
        # asked for; the width actually verified is one plus what the drafter proposed, and a
        # drafter that does not fill its budget verifies narrower than its label. On this file
        # the two drafters match to 0.00 columns at widths 3, 5 and 7 and differ by 0.99 at
        # width 9, where DFlash2 fills 87 % of n_max and MTP 99 %: 7.94 columns against 8.93,
        # one inside MMVQ_MAX_BATCH_SIZE and one past it. Their disagreement there is not the
        # control failing, it is the control never having applied.
        ea, eb = eff_width.get((w, specs[0])), eff_width.get((w, specs[1]))
        gap = abs(ea - eb) if (ea is not None and eb is not None) else None
        if gap is not None and gap > 0.25:
            print(f"    w={w}  {specs[0]} vs {specs[1]}: {same}/{len(common)} prompts agree"
                  f"   <-- NOT A CONTROL: effective widths {ea:.2f} and {eb:.2f} differ by "
                  f"{gap:.2f} columns")
            print(f"          Both were asked for width {w}. They did not verify at the same one,")
            print(f"          so nothing here bears on whether width determines the fork.")
            continue
        if same != len(common):
            control_failed.append(w)
        flag = "" if same == len(common) else "   <-- DISAGREE, the width account fails here"
        extra = f"   [effective {ea:.2f} vs {eb:.2f}]" if gap is not None else ""
        print(f"    w={w}  {specs[0]} vs {specs[1]}: {same}/{len(common)} prompts agree{flag}{extra}")
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

    # A censored cell carries no fork position. "same" means the two texts had not disagreed when
    # the budget ran out, so the fork is unobserved, not elsewhere; comparing it against an
    # observed index separates two widths on the strength of a cell that measured nothing, and the
    # partition then moves when the cap moves. It did: at a 400-token cap widths 3 and 4 share a
    # vector, and at 1600 one cell of code_sql_report turns censored for width 4 while width 3
    # forks at char 5423, which is enough to split them and to read as H8 failing. Two widths are
    # separated only by a prompt where BOTH diverged and the indices differ.
    def determined(v):
        return v not in ("same", "prefix", "?")

    def separation(a, b):
        """(prompts that separate a from b, prompts that determined both)."""
        seps, both = [], 0
        for va, vb, p in zip(sig[a], sig[b], prompts):
            if not (determined(va) and determined(vb)):
                continue
            both += 1
            if va != vb:
                seps.append(p)
        return seps, both

    sep_of = {(a, b): separation(a, b) for a in gradable for b in gradable if a != b}
    groups: list[list] = []
    for w in gradable:
        for g in groups:
            if all(not sep_of[(w, x)][0] for x in g):
                g.append(w)
                break
        else:
            groups.append([w])
    observed = [tuple(g) for g in groups]

    print("\n--- observed groups (same fork position on every prompt that determined both) ---")
    for g in observed:
        print(f"    {set(g)}  warps {warp_label(g)}")
    weak = sorted((min(a, b), max(a, b), both) for (a, b), (seps, both) in sep_of.items()
                  if not seps and both < len(prompts))
    for a, b, both in dict.fromkeys((a, b, n) for a, b, n in weak):
        print(f"    w{a} and w{b} are grouped on {both} of {len(prompts)} prompts; the rest are "
              f"censored for one of them and determine nothing")
    # Agreement on the determined cells need not be transitive, and if it is not then "group" is
    # the wrong word for what came out. Say so rather than let the greedy pass above pick one.
    unsep = {k for k, (seps, _) in sep_of.items() if not seps}
    broken = [(a, b, c) for a in gradable for b in gradable for c in gradable
              if len({a, b, c}) == 3 and (a, b) in unsep and (b, c) in unsep and (a, c) not in unsep]
    if broken:
        a, b, c = broken[0]
        print(f"    NOT AN EQUIVALENCE: w{a} groups with w{b} and w{b} with w{c}, but w{a} and w{c} "
              f"are separated.")
        print("    The sets above are one of several readings; the pairwise table is the finding.")

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
        print("    The partition is exactly the warp-count table.")
        # Consistency, and this file only ever had consistency to offer. The intervention that
        # was registered to settle it has since run: four builds from one configure, the GENERIC
        # table edited up at widths 5 to 8 and down at 3 and 4. SASS says the edit reached the
        # machine code and only there, 92 and 46 mul_mat_vec_q kernels at exactly those
        # ncols_dst, and Ampere dispatches every quantized type through MMVQ at ne11 1 to 8, so
        # the edited kernels are the ones that run. The forced builds changed the kernel by up
        # to 26.68 % of its runtime and changed not one output byte in 150 records each.
        print("    H8 IS NOT A CAUSAL CLAIM. Forcing the warp count moves this kernel's runtime")
        print("    by up to 26.7 % and moves no output byte, so it cannot move a fork position,")
        print("    which is a property of the text. The table coincides with the boundary; it is")
        print("    not what puts the widths into two groups. See analysis/warp_intervention_v2.txt")
        print("    and logs/sass_v2_summary.log. Whatever else changes at this width is open.")
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
