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
import sys
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
    print("PHASE L — speculative decoding across context depth")
    print("=" * 78)

    # ---------------------------------------------------------------- depth actually reached
    print("\nREALISED DEPTH (from the server's counters, not the request)")
    print(f"  {'asked':>8}  {'filler':>8}  {'prompt':>8}  {'cached':>8}  {'evaluated':>9}  status")
    for depth in sorted(rungs):
        name, d = rungs[depth]
        recs = d["records"]
        if not recs:
            print(f"  {depth:8d}  {'—':>8}  {'—':>8}  {'—':>8}  {'—':>9}  no records")
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
                row += f"{'—':>18}"
                continue
            v = st.median(r["decode_tok_s"] for r in rs)
            first.setdefault(m, v)
            row += f"{v:>12.1f}{'':>6}"
        print(row)

    print("\n  retention, as a fraction of that method's own shallowest rung")
    print("  " + f"{'depth':>8}" + "".join(f"{m:>18}" for m in methods))
    for depth in sorted(rungs):
        row = f"  {depth:8d}"
        for m in methods:
            rs = by.get((depth, m), [])
            if not rs or m not in first or not first[m]:
                row += f"{'—':>18}"
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
        if len(vals) < 2:
            print("  need at least two rungs")
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
                    out.setdefault(r["prompt_tag"], []).append(r["decode_tok_s"])
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
            cls = {r["prompt_tag"]: r.get("prompt_class", "?") for r in rs + brs}
            # (baseline, arm) — this order sets the sign of every number below.
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
                rs = [r for r in by.get((depth, m), []) if r.get("mean_len")]
                row += (f"{st.median(r['mean_len'] for r in rs):>12.3f}{'':>6}"
                        if rs else f"{'—':>18}")
            print(row)
        print("\n  A drafter that holds acceptance as depth grows is doing what DFlash2 claims.")
        print("  Acceptance falling while throughput falls faster means the loss is in the target,")
        print("  not the drafter, and speculation is inheriting the target's problem.")

    # ---------------------------------------------------------------- incidents
    total_inc = sum(len(d.get("incidents") or []) for _, d in rungs.values())
    print(f"\nINCIDENTS across the ladder: {total_inc}")
    for depth in sorted(rungs):
        _, d = rungs[depth]
        for inc in (d.get("incidents") or []):
            print(f"  {depth}: {inc.get('kind')}: {str(inc.get('detail'))[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
