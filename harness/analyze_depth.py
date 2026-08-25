"""Assemble the Phase L ladder across per-depth result files.

Each depth is its own run because `-c` is a server property, so the ladder only exists once the
rungs are read back together. Three questions, in order of what they can settle:

1. Does the decode collapse reported in llama.cpp #27623 reproduce here? That report is from an
   RTX 4080 SUPER (sm_89); this is sm_86. A reproduction makes it architecture-independent, and
   a non-reproduction is just as informative and narrows the report to Ada.
2. Does speculation survive the collapse? DFlash2 advertises long-context retention, so the
   answer is not guessable from the shallow-context result.
3. Does acceptance move with depth? More context can make the next token more predictable, or
   the drafter's own truncated view can make it less so.

Depth is taken from the server's own counters, not from what the filler was asked for. A rung
whose realised depth missed the request is reported as measured, not as labelled.

Usage:  python3 harness/analyze_depth.py [results/phase_l_*.json]
"""
import glob
import json
import re
import statistics as st
import speclen
import sys
import completeness as _CO  # noqa: E402
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats as S  # noqa: E402


def load(paths):
    rungs = {}
    for p in sorted(paths):
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"  ! unreadable: {p} ({e})")
            continue
        m = re.search(r"phase_l_(\d+)\.json$", str(p))
        if not m:
            continue
        rungs[int(m.group(1))] = (Path(p).name, d)
    return rungs


def method_of(arm):
    """Strip the @<depth> suffix the matrix adds so rungs can be lined up."""
    return arm.split("@")[0]


def main():
    paths = sys.argv[1:] or glob.glob("results/phase_l_*.json")
    paths = [p for p in paths if ".partial." not in p]
    rungs = load(paths)
    if not rungs:
        print("no Phase L rungs found (results/phase_l_<depth>.json)")
        return 1

    print("=" * 78)
    print("PHASE L - speculative decoding across context depth")
    print("=" * 78)

    # ---------------------------------------------------------------- completeness, first
    # A rung is one bench run and the driver calls this after every one of them, so the deepest
    # file is routinely still being appended to when this prints. analyze.py, cost_model.py and
    # width_groups.py all say so; this file did not, and it is the one that decides the cliff.
    # A half-written rung is not simply noisier: records land one arm-pass at a time and the arm
    # order rotates between passes, so a partial file holds an unbalanced set of arms and its
    # median is a different estimator, not a wider one.
    incomplete = set()
    print("\nCOMPLETENESS")
    for depth in sorted(rungs):
        _, d = rungs[depth]
        n, expected, _ = _CO.completeness(d)
        ok = bool(expected) and n >= expected
        if not ok:
            incomplete.add(depth)
        print(f"  {depth:8d}  {n:4d}/{expected or '?':<5} {'complete' if ok else 'STILL BEING WRITTEN'}")
    if incomplete:
        print(f"  Rungs {sorted(incomplete)} are short. Everything below that uses them is")
        print("  provisional, and the cliff verdict is withheld rather than taken from them.")

    # ---------------------------------------------------------------- depth actually reached
    print("\nREALISED DEPTH (from the server's counters, not the request)")
    print(f"  {'asked':>8}  {'filler':>8}  {'prompt':>8}  {'cached':>8}  {'evaluated':>9}  status")
    for depth in sorted(rungs):
        name, d = rungs[depth]
        recs = d["records"]
        if not recs:
            print(f"  {depth:8d}  {'-':>8}  {'-':>8}  {'-':>8}  {'-':>9}  no records")
            continue
        fil = st.median(r.get("filler_tokens") or 0 for r in recs)
        pt = st.median(r.get("prompt_tokens") or 0 for r in recs)
        cn = st.median(r.get("cache_n") or 0 for r in recs)
        ev = st.median((r.get("timings") or {}).get("t_prompt_n") or 0 for r in recs)
        # The KV position decode runs at is the whole prompt, cached or not.
        off = abs(pt - depth)
        status = "ok" if off <= 512 else f"OFF BY {pt - depth:+.0f}"
        if cn == 0 and len(recs) > 4:
            status += "; cache never hit (prefill repeated every request)"
        print(f"  {depth:8d}  {fil:8.0f}  {pt:8.0f}  {cn:8.0f}  {ev:9.0f}  {status}")

    # The ladder's denominator is not perfectly clean. A deeper rung moves more memory traffic,
    # which takes budget from the core inside one power limit, so the SM clock drifts down with
    # depth even when nothing is thermally throttling. Phase R2 measured the baseline's compute
    # elasticity at 0.266 near the top of the clock range, so a clock drift of d per cent costs
    # about 0.27d per cent of throughput and that much of any decline below is not depth.
    print("\nCLOCK, WHICH DEPTH MOVES ON ITS OWN")
    print(f"  {'depth':>8}  {'SM mean':>8}  {'vs first':>9}  {'mem':>7}  {'temp':>6}  {'watt':>6}"
          f"   throughput cost at elasticity 0.266")
    _first_sm = None
    for depth in sorted(rungs):
        _, d = rungs[depth]
        base = [r for r in d["records"] if method_of(r["arm"]).startswith("baseline")]
        vals = lambda k: [(r.get("power") or {}).get(k) for r in base
                          if (r.get("power") or {}).get(k) is not None]
        sm, mem, tp, w = vals("sm_clock_mean_mhz"), vals("mem_clock_mean_mhz"), \
            vals("temp_mean_c"), vals("power_mean_w")
        if not sm:
            continue
        sm_m = st.median(sm)
        if _first_sm is None:
            _first_sm = sm_m
        drift = (sm_m / _first_sm - 1) * 100
        print(f"  {depth:8d}  {sm_m:8.0f}  {drift:>+8.2f}%  {st.median(mem) if mem else 0:7.0f}"
              f"  {st.median(tp) if tp else 0:6.1f}  {st.median(w) if w else 0:6.0f}"
              f"   {drift * 0.266:>+8.2f}%")
    print("  A clock that falls with depth puts part of the decline in the wrong column. Read the")
    print("  last column against the retention above: what is left is depth.")

    # ---------------------------------------------------------------- throughput vs depth
    methods, by = [], {}
    for depth in sorted(rungs):
        _, d = rungs[depth]
        for r in d["records"]:
            m = method_of(r["arm"])
            if m not in methods:
                methods.append(m)
            by.setdefault((depth, m), []).append(r)

    print("\nDECODE THROUGHPUT (tok/s, median over prompts x passes)")
    print("  " + f"{'depth':>8}" + "".join(f"{m:>18}" for m in methods))
    first = {}
    for depth in sorted(rungs):
        row = f"  {depth:8d}"
        for m in methods:
            rs = by.get((depth, m), [])
            if not rs:
                row += f"{'-':>18}"
                continue
            v = st.median(r["decode_tok_s"] for r in rs)
            first.setdefault(m, v)
            row += f"{v:>12.1f}{'':>6}"
        print(row)

    # The anchor is per method, so a method absent from the shallowest rung is measured against
    # a deeper one and shows 100 % there while the others show a fall. Read side by side that
    # looks like better retention. Name the anchors and say so when they differ.
    anchors = {}
    for d in sorted(rungs):
        for m in methods:
            if by.get((d, m)) and m not in anchors:
                anchors[m] = d
    shared = len(set(anchors.values())) <= 1
    print("\n  retention, as a fraction of that method's own shallowest rung")
    if not shared:
        print("  ANCHORS DIFFER: " + ", ".join(f"{m}@{anchors[m]}" for m in methods if m in anchors))
        print("  A method absent from a shallower rung reads 100 % at its own first one. These")
        print("  columns are not on a common scale and must not be compared across methods.")
    else:
        print(f"  all methods anchored at {next(iter(anchors.values())) if anchors else '-'}")
    print("  " + f"{'depth':>8}" + "".join(f"{m:>18}" for m in methods))
    for depth in sorted(rungs):
        row = f"  {depth:8d}"
        for m in methods:
            rs = by.get((depth, m), [])
            if not rs or m not in first or not first[m]:
                row += f"{'-':>18}"
                continue
            row += f"{st.median(r['decode_tok_s'] for r in rs) / first[m]:>12.2%}{'':>6}"
        print(row)

    # ---------------------------------------------------------------- the cliff
    print("\nCLIFF TEST (llama.cpp #27623: ~25x collapse past ~80 K on sm_89)")
    base = [m for m in methods if m.startswith("baseline") and not m.endswith("-pr")]
    if not base:
        print("  no baseline arm; cannot test")
    else:
        b = base[0]
        vals = [(d, st.median(r["decode_tok_s"] for r in by[(d, b)]))
                for d in sorted(rungs) if by.get((d, b))]
        # The report puts the collapse past about 80 K. Rungs below that cannot test it, and an
        # earlier version scored best-against-worst over whatever rungs existed: at 8 K and 32 K
        # it printed "DOES NOT REPRODUCE ... #27623 is not architecture-independent" from two
        # depths that are both under the threshold. That is a conclusion about a depth this run
        # had not reached.
        CLIFF_DEPTH = 80000
        deepest = max(d for d, _ in vals) if vals else 0
        used_incomplete = sorted(incomplete & {d for d, _ in vals})
        if len(vals) < 2:
            print("  need at least two rungs")
        elif used_incomplete:
            print(f"  {b}: rungs {used_incomplete} are still being written.")
            print("  VERDICT WITHHELD. A 25x claim decided from a rung that is half a run is the")
            print("  same mistake that put a DFlash2 coefficient of 0.2479 in the README against")
            print("  0.2481 finished. Re-run this once the ladder is done.")
        elif deepest < CLIFF_DEPTH:
            print(f"  {b}: {vals[0][1]:.1f} tok/s at {vals[0][0]} -> {vals[-1][1]:.1f} tok/s at {deepest}")
            print(f"  VERDICT WITHHELD. The report puts the collapse past about {CLIFF_DEPTH//1000} K and the")
            print(f"  deepest rung here is {deepest}. Nothing below the threshold can reproduce or refute it;")
            print(f"  the ratio over these rungs is {max(v for _, v in vals)/min(v for _, v in vals):.1f}x and describes")
            print(f"  ordinary degradation, not the cliff.")
        else:
            worst = min(vals, key=lambda x: x[1])
            best = max(vals, key=lambda x: x[1])
            ratio = best[1] / worst[1] if worst[1] else float("inf")
            print(f"  {b}: {best[1]:.1f} tok/s at {best[0]} -> {worst[1]:.1f} tok/s at {worst[0]}")
            print(f"  worst-to-best ratio {ratio:.1f}x")
            if ratio >= 10:
                print(f"  REPRODUCES on sm_86. The report is not specific to Ada.")
            elif ratio >= 3:
                print(f"  partial: a real degradation, well short of the reported ~25x.")
            else:
                print(f"  DOES NOT REPRODUCE. Decode degrades gracefully here; on this evidence "
                      f"#27623 is not architecture-independent.")
            # a cliff is a step, not a slope: report the largest single-rung drop
            drops = [(vals[i][0], vals[i - 1][1] / vals[i][1])
                     for i in range(1, len(vals)) if vals[i][1]]
            if drops:
                d0, f0 = max(drops, key=lambda x: x[1])
                print(f"  largest single-rung drop: {f0:.2f}x on entering {d0}")

    # ---------------------------------------------------------------- speedup vs depth
    print("\nSPEEDUP vs the matching baseline, per depth "
          "(cluster bootstrap over prompts)")
    for depth in sorted(rungs):
        name, d = rungs[depth]
        bmap = d.get("baseline_map") or {}
        print(f"\n  depth {depth}")
        for m in methods:
            if m.startswith("baseline"):
                continue
            rs = by.get((depth, m), [])
            if not rs:
                continue
            bname = bmap.get(rs[0]["arm"])
            brs = [r for r in d["records"] if r["arm"] == bname]
            if not brs:
                print(f"    {m:<16} no baseline arm {bname!r} at this depth")
                continue
            def by_tag(xs):
                out = {}
                for r in xs:
                    out.setdefault(r["prompt"], []).append(r["decode_tok_s"])
                return out

            kb, ka = by_tag(brs), by_tag(rs)
            shared = set(kb) & set(ka)
            if not shared:
                print(f"    {m:<16} no paired observations")
                continue
            # The bootstrap refuses unequal coverage rather than quietly dropping prompts, so
            # trim both sides to what they share and say how much was dropped.
            dropped = (set(kb) | set(ka)) - shared
            kb = {k: v for k, v in kb.items() if k in shared}
            ka = {k: v for k, v in ka.items() if k in shared}
            # The record fields are `prompt` and `class`. `prompt_tag`/`prompt_class` were
            # neither, so this raised KeyError before the bootstrap and, had it not, would have
            # put every prompt in class "?" and thrown away the stratification.
            cls = {r["prompt"]: r.get("class", "?") for r in rs + brs}
            # (baseline, arm) - this order sets the sign of every number below.
            iv = S.paired_cluster_bootstrap(
                kb, ka, {k: cls.get(k, "?") for k in shared}, relative=True)
            pt, lo, hi = iv.point, iv.lo, iv.hi
            mark = "" if lo > 0 else ("   no effect" if hi > 0 else "   SLOWER")
            note = f"   ({len(dropped)} prompt(s) dropped)" if dropped else ""
            if iv.width_understated:
                note += ("   INTERVAL IS A LOWER BOUND ON WIDTH: single-prompt class(es) "
                         + ",".join(iv.singleton_classes) + " contributed no variance")
            print(f"    {m:<16} {pt:+7.1f}%  [{lo:+.1f}%, {hi:+.1f}%]{mark}{note}")

    # ---------------------------------------------------------------- acceptance vs depth
    print("\nACCEPTANCE vs depth (mean accepted tokens per verification step)")
    spec = [m for m in methods if not m.startswith("baseline")]
    if spec:
        print("  " + f"{'depth':>8}" + "".join(f"{m:>18}" for m in spec))
        for depth in sorted(rungs):
            row = f"  {depth:8d}"
            for m in spec:
                # `mean_len` is not a record field; it is derived from the counters, and
                # `r.get("mean_len")` was None for every record, so this whole table printed "-"
                # in every cell rather than reporting that it could not compute anything.
                mls = [v for r in by.get((depth, m), [])
                       if (v := speclen.mean_len(r)) is not None]
                row += (f"{st.median(mls):>12.3f}{'':>6}" if mls else f"{'-':>18}")
            print(row)
        print("\n  A drafter that holds acceptance as depth grows is doing what DFlash2 claims.")
        print("  Acceptance falling while throughput falls faster means the loss is in the target,")
        print("  not the drafter, and speculation is inheriting the target's problem.")

    # ---------------------------------------------------------------- incidents
    # Grouped, not enumerated. One incident per request is a real shape here - the cache check
    # raised 180 a rung before it was made conditional on the declared mode - and printing them
    # one by one buries whatever else the ladder found under a page of the same line.
    total_inc = sum(len(d.get("incidents") or []) for _, d in rungs.values())
    repaired = sum(sum(e.get("removed", 0) for e in (d.get("incidents_repaired") or []))
                   for _, d in rungs.values())
    print(f"\nINCIDENTS across the ladder: {total_inc}"
          + (f"   ({repaired} removed as raised against a condition never declared; "
             f"see incidents_repaired)" if repaired else ""))
    for depth in sorted(rungs):
        _, d = rungs[depth]
        incs = d.get("incidents") or []
        if not incs:
            continue
        kinds = {}
        for inc in incs:
            kinds.setdefault(inc.get("kind"), []).append(inc)
        for kind, group in sorted(kinds.items()):
            arms = sorted({i.get("arm") for i in group if i.get("arm")})
            print(f"  {depth}: {kind} x{len(group)}"
                  + (f" on {', '.join(arms[:4])}{' ...' if len(arms) > 4 else ''}" if arms else ""))
            print(f"      e.g. {str(group[0].get('detail'))[:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
