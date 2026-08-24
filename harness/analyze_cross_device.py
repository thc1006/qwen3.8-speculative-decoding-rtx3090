"""Cross-device comparison - restricted, on purpose, to quantities that survive the crossing.

The RTX 3090 and RTX A6000 share an architecture (GA102, sm_86) and therefore share CUDA kernels,
but differ by roughly 18 % in memory bandwidth (936 vs 768 GB/s) and 29 % in power budget
(420 vs 300 W). Absolute throughput and absolute energy therefore mean different things on each
card and are NOT compared here. What is compared:

  * **speedup** - a within-device ratio
  * **k, k0, c** - the cost model's coefficients, already expressed in units of a plain decode
    step on the same device, so dimensionless
  * **acceptance** - a ratio, and on identical weights with greedy sampling it should be
    IDENTICAL across two cards of the same architecture
  * **fork positions** - likewise, identical kernels on identical inputs should fork identically

The last two are not results, they are **controls**. If acceptance or fork positions differ
between two sm_86 cards running the same GGUF at greedy, then something other than the device is
varying and every other cross-device number in the report is suspect. The report says so loudly
rather than proceeding.

The scientific point of the second card is the first two: a physical ~18 % bandwidth step, where
software clock offsets on one card can only reach about +-4 %.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import cost_model


def _device_of(result: dict) -> dict:
    """Device facts, from the structured field when present and from the env snapshot when not.

    Result files written before the structured `design.device` field existed still carry the
    full nvidia-smi CSV in `env.gpu`, in this order:

        name, memory.total, driver_version, compute_cap, power.limit,
        power.default_limit, power.max_limit, clocks.max.graphics, clocks.max.memory

    Parsing it keeps older runs - including this study's own primary Phase A result - usable in
    a cross-device comparison instead of showing every field as unknown.
    """
    d = dict((result.get("design") or {}).get("device") or {})
    if d.get("name") and d.get("clocks_max_memory_mhz"):
        return d

    gpu = (result.get("env") or {}).get("gpu", "")
    parts = [x.strip() for x in gpu.split(",")] if gpu else []

    def _num(s: str):
        tok = s.split()[0] if s else ""
        try:
            return float(tok)
        except ValueError:
            return None

    if len(parts) >= 9:
        d.setdefault("name", parts[0])
        mib = _num(parts[1])
        if mib:
            d.setdefault("vram_gb", round(mib / 1024.0, 1))
        d.setdefault("driver", parts[2])
        d.setdefault("compute_cap", parts[3])
        d.setdefault("power_default_w", _num(parts[5]))
        d.setdefault("clocks_max_graphics_mhz", _num(parts[7]))
        d.setdefault("clocks_max_memory_mhz", _num(parts[8]))
    elif parts:
        d.setdefault("name", parts[0])

    # last resort: the overclock snapshot carries the clocks too
    oc = (result.get("env") or {}).get("overclock_state") or {}
    for src, dst in (("clocks_max_memory_mhz", "clocks_max_memory_mhz"),
                     ("clocks_max_graphics_mhz", "clocks_max_graphics_mhz"),
                     ("power_default_limit_w", "power_default_w")):
        if d.get(dst) in (None, "?") and oc.get(src) is not None:
            d[dst] = oc[src]
    return d or {"name": "unknown"}


def _label(result: dict, path: Path) -> str:
    """Column label. The device name alone is ambiguous whenever two runs come from the same
    model of card - including the common case of comparing two runs on one machine - so the
    filename stem is always appended."""
    d = _device_of(result)
    n = (d.get("name", "?") or "?").replace("NVIDIA", "").replace("GeForce", "").strip()
    stem = path.stem.replace("phase_", "")
    return f"{n}/{stem}" if n and n != "?" else stem


def report(paths: list[Path]) -> None:
    runs = []
    problems = []
    for p in paths:
        try:
            r = json.loads(p.read_text())
        except FileNotFoundError:
            problems.append(f"{p}: not found")
            continue
        except json.JSONDecodeError as e:
            problems.append(f"{p}: not valid JSON ({e.msg} at line {e.lineno}) - a run that was "
                            f"killed mid-write leaves this; try the newest file in "
                            f"results/snapshots/ or the .records.jsonl stream")
            continue
        rows = cost_model.collect(r)
        if not rows:
            problems.append(f"{p}: no usable rows (no speculative arms, or no same-tree baseline)")
        runs.append((_label(r, p), r, rows))

    for msg in problems:
        print(f"!! {msg}")
    if len(runs) < 2:
        print(f"\nneed at least two readable result files to compare; got {len(runs)}")
        return

    print("=" * 100)
    print("CROSS-DEVICE COMPARISON - dimensionless quantities only")
    print("=" * 100)
    print("\n--- devices ---")
    for lab, r, _ in runs:
        d = _device_of(r)
        oc = (r.get("env") or {}).get("overclock_state") or {}
        print(f"  {lab:22s} vram={d.get('vram_gb','?')}GB  sm_{str(d.get('compute_cap','?')).replace('.','')}  "
              f"mem_clk_max={d.get('clocks_max_memory_mhz','?')}MHz  "
              f"pl_default={d.get('power_default_w','?')}W  stock={oc.get('is_stock')}")

    caps = {str(_device_of(r).get("compute_cap")) for _, r, _ in runs}
    if len(caps) > 1:
        print(f"\n  !! devices differ in compute capability {caps}. Kernel selection may differ, "
              f"so the controls below are no longer expected to match and a mismatch would not "
              f"be evidence of a problem.")

    # ---------------------------------------------------------------- controls
    print("\n--- CONTROL 1: acceptance should be identical on identical weights + greedy ---")
    acc: dict[str, dict[tuple[str, str], float]] = {}
    for lab, _, rows in runs:
        m: dict[tuple[str, str], list[float]] = defaultdict(list)
        for x in rows:
            m[(x["arm"], x["prompt"])].append(x["acceptance"])
        acc[lab] = {k: statistics.fmean(v) for k, v in m.items()}
    labs = [l for l, _, _ in runs]
    if len(labs) >= 2:
        base = labs[0]
        for other in labs[1:]:
            common = sorted(set(acc[base]) & set(acc[other]))
            if not common:
                print(f"  {base} vs {other}: no shared (arm, prompt) pairs")
                continue
            diffs = [abs(acc[base][k] - acc[other][k]) for k in common]
            worst = max(range(len(common)), key=lambda i: diffs[i])
            print(f"  {base} vs {other}: n={len(common)}  max |deltaacceptance|={max(diffs):.5f}  "
                  f"mean={statistics.fmean(diffs):.5f}  worst={common[worst]}")
            if max(diffs) > 0.005:
                print("     !! acceptance differs by more than rounding. Same architecture and "
                      "greedy sampling should give identical draft decisions; something other "
                      "than the device is varying. Treat every number below as suspect.")

    print("\n--- CONTROL 2: greedy fork positions should be identical ---")
    forks: dict[str, dict[tuple[str, str], object]] = {}
    for lab, r, _ in runs:
        m = {}
        for rec in r["records"]:
            d = rec.get("divergence")
            if d:
                m[(rec["arm"], rec["prompt"])] = "SAME" if d["identical"] else d["first_diff_char"]
        forks[lab] = m
    if len(labs) >= 2:
        base = labs[0]
        for other in labs[1:]:
            common = sorted(set(forks[base]) & set(forks[other]))
            if not common:
                print(f"  {base} vs {other}: no shared pairs")
                continue
            same = sum(1 for k in common if forks[base][k] == forks[other][k])
            print(f"  {base} vs {other}: {same}/{len(common)} fork positions identical")
            if same != len(common):
                for k in common:
                    if forks[base][k] != forks[other][k]:
                        print(f"     differs: {k} -> {base}={forks[base][k]} {other}={forks[other][k]}")
                        break

    # ---------------------------------------------------------------- results
    print("\n--- RESULT 1: speedup by arm (within-device ratio) ---")
    hdr = f"{'arm':18s}" + "".join(f"{l[:16]:>18s}" for l in labs)
    print(hdr)
    arms = sorted({x["arm"] for _, _, rows in runs for x in rows})
    sp: dict[str, dict[str, float]] = {}
    for lab, _, rows in runs:
        m = defaultdict(list)
        for x in rows:
            m[x["arm"]].append(x["speedup"])
        sp[lab] = {k: statistics.fmean(v) for k, v in m.items()}
    for a in arms:
        print(f"{a:18s}" + "".join(
            f"{(f'{sp[l][a]:.3f}x' if a in sp[l] else '-'):>18s}" for l in labs))

    print("\n--- RESULT 2: cost model coefficients (the point of the second device) ---")
    print(f"{'device':22s} {'method':14s} {'widths':>16s} {'k0':>8s} {'c':>8s} {'r2':>8s}")
    coeffs: dict[tuple[str, str], tuple[float, float]] = {}
    for lab, _, rows in runs:
        by_m = defaultdict(list)
        for x in rows:
            by_m[x["spec_type"]].append(x)
        for method, g in sorted(by_m.items()):
            pts = defaultdict(list)
            for x in g:
                pts[x["width"]].append(x["k"])
            if len(pts) < 2:
                print(f"{lab:22s} {method:14s} {str(sorted(pts)):>16s} "
                      f"{'-':>8s} {'-':>8s}   one width only")
                continue
            xs = [w - 1 for w in sorted(pts)]
            ys = [statistics.fmean(pts[w]) for w in sorted(pts)]
            a, b, r2 = cost_model._linfit(xs, ys)
            coeffs[(lab, method)] = (a, b)
            note = "  (2 widths - r2 is meaningless)" if len(pts) == 2 else ""
            print(f"{lab:22s} {method:14s} {str(sorted(pts)):>16s} {a:8.4f} {b:8.4f} {r2:8.4f}{note}")

    if len(labs) >= 2:
        print("\n--- RESULT 3: how the marginal cost c moves with memory bandwidth ---")
        for method in sorted({m for _, m in coeffs}):
            have = [(l, coeffs[(l, method)]) for l in labs if (l, method) in coeffs]
            if len(have) < 2:
                continue
            print(f"\n  {method}")
            for lab, (k0, c) in have:
                d = _device_of(next(r for l, r, _ in runs if l == lab))
                mem = d.get("clocks_max_memory_mhz")
                print(f"    {lab:22s} mem_clk_max={mem}  k0={k0:.4f}  c={c:.4f}")
            (l1, (k01, c1)), (l2, (k02, c2)) = have[0], have[1]
            d1 = _device_of(next(r for l, r, _ in runs if l == l1))
            d2 = _device_of(next(r for l, r, _ in runs if l == l2))
            m1, m2 = d1.get("clocks_max_memory_mhz"), d2.get("clocks_max_memory_mhz")
            if m1 and m2 and m1 != m2:
                # Memory clock is a bandwidth PROXY, and only a valid one when the two cards
                # share a bus width. Both GA102 boards considered here are 384-bit, which is why
                # the clock ratio 9751/8001 reproduces the datasheet bandwidth ratio 936/768.
                # nvidia-smi does not report bus width, so the assumption is stated rather than
                # verified, and any pair whose names are not both known-384-bit is flagged.
                KNOWN_384 = ("3090", "a6000", "3090ti", "a40")
                n1 = (d1.get("name") or "").lower().replace(" ", "")
                n2 = (d2.get("name") or "").lower().replace(" ", "")
                ok = any(k in n1 for k in KNOWN_384) and any(k in n2 for k in KNOWN_384)
                dbw = (m2 - m1) / m1 * 100
                dc = (c2 - c1) / c1 * 100
                print(f"    memory clock {dbw:+.1f} %  ->  c {dc:+.1f} %   (elasticity "
                      f"{dc/dbw if dbw else float('nan'):+.2f})")
                if ok:
                    print("    Both cards are 384-bit GA102, so the memory-clock ratio is a valid")
                    print("    bandwidth proxy (9751/8001 reproduces the 936/768 GB/s ratio).")
                else:
                    print("    !! Bus width is not known to match for this pair, and nvidia-smi")
                    print("       does not report it. The memory-clock ratio is NOT a valid")
                    print("       bandwidth proxy here - treat the elasticity above as unusable")
                    print("       until the bus widths are confirmed equal.")
                print("    A marginal cost that is compute-bound should barely move with memory")
                print("    bandwidth; one that is memory-bound should move with it roughly 1:1.")
                print("    CAVEAT: these two cards also differ in power budget (420 vs 300 W),")
                print("    so this is a two-variable step, not a clean bandwidth-only lever.")
                print("    Phase R varies each independently on ONE card and is the controlled")
                print("    version of this comparison; this is the cross-check, not the test.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    a = ap.parse_args()
    report([Path(x) for x in a.results])


if __name__ == "__main__":
    main()
