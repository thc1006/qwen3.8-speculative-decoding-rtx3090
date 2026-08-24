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
import sys

# ncols_dst -> warps, MMVQ_PARAMETERS_GENERIC, which is what sm_86 falls through to.
def warps_for(width: int) -> int:
    if width <= 4:
        return 4
    if width <= 8:
        return 2
    return 1


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
    arms = d.get("arms", {})

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

    print("=" * 92)
    print("FORK POSITION BY VERIFICATION WIDTH")
    print("=" * 92)
    print(f"  source: {path}")
    print(f"  widths: {widths}")
    print(f"  MMVQ_PARAMETERS_GENERIC warps: "
          + "  ".join(f"w{w}={warps_for(w)}" for w in widths))

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
        print(f"    {set(g)}  warps {sorted({warps_for(w) for w in g})}")

    # -------------------------------------------------- verdict on H8
    pred: dict[int, list] = collections.OrderedDict()
    for w in gradable:
        pred.setdefault(warps_for(w), []).append(w)
    predicted = [tuple(v) for v in pred.values()]
    print("\n--- H8: does the grouping follow calc_nwarps? ---")
    if lossless:
        print(f"    computed over the diverging widths only, {sorted(gradable)}")
    print(f"    predicted from the table: {[set(g) for g in predicted]}")
    print(f"    observed:                 {[set(g) for g in observed]}")
    if control_failed:
        print(f"    VERDICT WITHHELD. Two drafters give different fork positions at width(s) "
              f"{set(control_failed)}, so fork position is not a function of width alone and the")
        print("    grouping below was built by letting one drafter overwrite the other. Nothing")
        print("    here supports or refutes H8 until that is resolved, and llama.cpp #25618 needs")
        print("    telling either way.")
    elif set(map(frozenset, observed)) == set(map(frozenset, predicted)):
        print("    H8 SUPPORTED. The partition is exactly the warp-count table.")
    else:
        print("    H8 NOT SUPPORTED. The grouping tracks something other than the warp count,")
        print("    and the mechanism offered in llama.cpp #25618 needs withdrawing there.")
        for g in observed:
            if len({warps_for(w) for w in g}) > 1:
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
