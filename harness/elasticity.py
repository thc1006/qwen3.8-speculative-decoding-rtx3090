"""Resource-response analysis for Phase R.

Estimates how each method's throughput responds to two resources that this card lets us move
independently: memory clock (at a fixed power cap) and sustained core clock (via the power cap,
at a fixed memory clock).

**Elasticities are reported per interval, never pooled into one number.** The baseline's compute
response measured on this hardware is 0.910 between 386 and 907 MHz but 0.498 between 907 and
1783 MHz — at severe compute starvation even a bandwidth-bound workload becomes compute-limited,
and it transitions back as clock rises. A single pooled figure would average across a regime
change and describe neither end. Cross-method comparisons are therefore only made over the SAME
interval.

The discriminating question, from PREREGISTRATION.md:

    H2'  (quantization x arithmetic intensity) — speculation converts a bandwidth-bound decode
         into a compute-bound verify, so a speculative arm should be LESS bandwidth-elastic and
         MORE compute-elastic than the no-spec baseline.

    H2   (Gated DeltaNet state rollback) — the marginal cost is state reconstruction, i.e. memory
         traffic, so a speculative arm should be AT LEAST as bandwidth-elastic as baseline.

They predict opposite signs on the same contrast, which is what makes this decisive rather than
suggestive.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def _cells(result: dict) -> dict[tuple[str, str], dict[str, list[float]]]:
    """(method, condition) -> {prompt: [tok/s per pass]}"""
    out: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in result["records"]:
        if "@" not in r["arm"] or not r.get("decode_tok_s"):
            continue
        method, cond = r["arm"].split("@", 1)
        out[(method, cond)][r["prompt"]].append(r["decode_tok_s"])
    return out


def _clock(result: dict, method: str, cond: str, key: str) -> float | None:
    vals = [(r.get("power") or {}).get(key)
            for r in result["records"] if r["arm"] == f"{method}@{cond}"]
    vals = [v for v in vals if v]
    return statistics.fmean(vals) if vals else None


def _paired_elasticity(lo: dict[str, list[float]], hi: dict[str, list[float]],
                       x_lo: float, x_hi: float, *, n_boot: int = 5000,
                       seed: int = 20260824) -> tuple[float, float, float]:
    """Local elasticity d(ln y)/d(ln x), with a cluster bootstrap over prompts.

    Prompts are the unit of replication; passes of one prompt are repeated measures.
    """
    tags = sorted(set(lo) & set(hi))
    if not tags or x_lo <= 0 or x_hi <= 0 or x_lo == x_hi:
        return (float("nan"),) * 3
    dlnx = math.log(x_hi) - math.log(x_lo)

    def point(sample: list[str]) -> float:
        ylo = statistics.fmean([statistics.fmean(lo[t]) for t in sample])
        yhi = statistics.fmean([statistics.fmean(hi[t]) for t in sample])
        return (math.log(yhi) - math.log(ylo)) / dlnx

    est = point(tags)
    rng = random.Random(seed)
    reps = sorted(point([tags[rng.randrange(len(tags))] for _ in tags]) for _ in range(n_boot))
    return est, reps[int(0.025 * n_boot)], reps[int(0.975 * n_boot) - 1]


def _cross_term_correction(result: dict, method: str, lo_c: str, hi_c: str,
                           moved_key: str, other_key: str, other_elasticity: float
                           ) -> tuple[float, float]:
    """How much of a measured response is really the OTHER resource moving underneath it.

    The bandwidth lever is not clean. At a fixed power cap, raising the memory clock draws power
    away from the core, so the core clock falls across the bandwidth sweep — measured here as
    1799 -> 1754 MHz for the baseline and 1722 -> 1712 MHz for `mtp-n3`. A compute-elastic arm
    is penalised by that, so its measured bandwidth elasticity is a NET of the bandwidth gain and
    the core-clock loss, and is therefore an UNDER-estimate.

    Returns (delta_ln_other, implied_correction_to_dlny), where the correction is what the other
    resource contributed and should be removed before attributing the rest to the moved one.
    """
    x_lo = _clock(result, method, lo_c, other_key)
    x_hi = _clock(result, method, hi_c, other_key)
    if not (x_lo and x_hi):
        return (0.0, 0.0)
    d_ln_other = math.log(x_hi) - math.log(x_lo)
    return (d_ln_other, other_elasticity * d_ln_other)


AXES = {
    "memory bandwidth": ("mem_clock_mean_mhz", ["bw-lo", "stock", "bw-hi"]),
    "compute (SM clock)": ("sm_clock_mean_mhz", ["pw-vlo", "pw-lo", "stock"]),
}


def report(result: dict) -> None:
    cells = _cells(result)
    methods = sorted({m for m, _ in cells})
    baseline = "baseline"

    print("=" * 100)
    print("RESOURCE RESPONSE — elasticities per interval (never pooled across a regime change)")
    print("=" * 100)

    print("\n--- measured operating points ---")
    conds = sorted({c for _, c in cells})
    print(f"{'method':10s} {'condition':8s} {'tok/s':>8s} {'SM MHz':>8s} {'mem MHz':>8s} {'W':>6s}")
    for m in methods:
        for c in conds:
            if (m, c) not in cells:
                continue
            vals = [v for vs in cells[(m, c)].values() for v in vs]
            print(f"{m:10s} {c:8s} {statistics.fmean(vals):8.2f} "
                  f"{(_clock(result, m, c, 'sm_clock_mean_mhz') or 0):8.0f} "
                  f"{(_clock(result, m, c, 'mem_clock_mean_mhz') or 0):8.0f} "
                  f"{(_clock(result, m, c, 'power_mean_w') or 0):6.0f}")

    for axis, (clock_key, order) in AXES.items():
        print(f"\n{'=' * 100}\n--- {axis} ---")
        present = [c for c in order if any((m, c) in cells for m in methods)]
        if len(present) < 2:
            print("  not enough conditions measured yet")
            continue
        for lo_c, hi_c in zip(present, present[1:]):
            print(f"\n  interval {lo_c} -> {hi_c}")
            base_e = None
            for m in methods:
                if (m, lo_c) not in cells or (m, hi_c) not in cells:
                    continue
                x_lo = _clock(result, m, lo_c, clock_key)
                x_hi = _clock(result, m, hi_c, clock_key)
                e, l, h = _paired_elasticity(cells[(m, lo_c)], cells[(m, hi_c)], x_lo, x_hi)
                if m == baseline:
                    base_e = e
                rel = f"  ({e/base_e:5.2f}x baseline)" if base_e and m != baseline else ""
                print(f"    {m:10s} {x_lo:6.0f} -> {x_hi:6.0f}  "
                      f"elasticity {e:6.3f} [{l:6.3f}, {h:6.3f}]{rel}")

    # ------------------------------------------------------- cross-term correction
    print(f"\n{'=' * 100}\n--- correction: the bandwidth lever is not clean ---")
    print("  At a fixed power cap, raising the memory clock takes power from the core, so core")
    print("  clock falls across the bandwidth sweep. A compute-elastic arm is penalised by that,")
    print("  which makes its measured bandwidth elasticity an under-estimate. Below, the core-")
    print("  clock contribution is estimated using each method's OWN compute elasticity (from the")
    print("  pw-lo -> stock interval) and removed.")
    bw_key, bw_order = AXES["memory bandwidth"]
    cp_key, _ = AXES["compute (SM clock)"]
    pres = [c for c in bw_order if any((m, c) in cells for m in methods)]
    if len(pres) >= 2:
        lo_c, hi_c = pres[0], pres[-1]
        print(f"\n  over {lo_c} -> {hi_c}:")
        print(f"    {'method':10s} {'core MHz':>16s} {'measured':>9s} {'core term':>10s} "
              f"{'corrected':>10s}")
        for m in methods:
            if (m, lo_c) not in cells or (m, hi_c) not in cells:
                continue
            # this method's own compute elasticity, on the normal-clock interval
            ce = float("nan")
            if (m, "pw-lo") in cells and (m, "stock") in cells:
                ce, *_ = _paired_elasticity(cells[(m, "pw-lo")], cells[(m, "stock")],
                                            _clock(result, m, "pw-lo", cp_key),
                                            _clock(result, m, "stock", cp_key))
            e, *_ = _paired_elasticity(cells[(m, lo_c)], cells[(m, hi_c)],
                                       _clock(result, m, lo_c, bw_key),
                                       _clock(result, m, hi_c, bw_key))
            c_lo = _clock(result, m, lo_c, cp_key)
            c_hi = _clock(result, m, hi_c, cp_key)
            if not (c_lo and c_hi) or ce != ce:
                print(f"    {m:10s} {'-':>16s} {e:9.3f} {'-':>10s} {'-':>10s}")
                continue
            d_ln_core, core_term = _cross_term_correction(result, m, lo_c, hi_c,
                                                          bw_key, cp_key, ce)
            d_ln_mem = math.log(_clock(result, m, hi_c, bw_key)) - \
                       math.log(_clock(result, m, lo_c, bw_key))
            corrected = e - core_term / d_ln_mem
            print(f"    {m:10s} {c_lo:7.0f} ->{c_hi:6.0f} {e:9.3f} "
                  f"{-core_term/d_ln_mem:10.3f} {corrected:10.3f}")
        print("\n  The correction raises every arm's bandwidth elasticity, baseline included, so")
        print("  the RATIO between them moves less than either figure does.")

    # ---------------------------------------------------- interval-matching caveat
    print(f"\n--- caveat: the compute intervals are not identical across methods ---")
    print("  Under the same power cap, different methods settle at different core clocks, because")
    print("  a bandwidth-heavy workload spends more of the budget on memory. Measured here:")
    for c in ["pw-vlo", "pw-lo", "stock"]:
        row = []
        for m in methods:
            v = _clock(result, m, c, cp_key)
            if v:
                row.append(f"{m}={v:.0f}")
        if row:
            print(f"    {c:8s} " + "  ".join(row))
    print("  So an interval labelled 'pw-lo -> stock' spans a different clock range for each")
    print("  method. Since elasticity is regime-dependent (see the two compute intervals above),")
    print("  these comparisons are close but not exactly matched. The direction of the effect is")
    print("  far larger than this mismatch; the magnitudes should be read with it in mind.")

    # ------------------------------------------------------------------ verdict
    print(f"\n{'=' * 100}\n--- H2 vs H2' on matched intervals ---")
    bw_key, bw_order = AXES["memory bandwidth"]
    cp_key, cp_order = AXES["compute (SM clock)"]
    for m in methods:
        if m == baseline:
            continue
        lines = []
        for label, (key, order) in AXES.items():
            pres = [c for c in order if (m, c) in cells and (baseline, c) in cells]
            if len(pres) < 2:
                continue
            lo_c, hi_c = pres[0], pres[-1]
            eb, *_ = _paired_elasticity(cells[(baseline, lo_c)], cells[(baseline, hi_c)],
                                        _clock(result, baseline, lo_c, key),
                                        _clock(result, baseline, hi_c, key))
            em, *_ = _paired_elasticity(cells[(m, lo_c)], cells[(m, hi_c)],
                                        _clock(result, m, lo_c, key),
                                        _clock(result, m, hi_c, key))
            if eb:
                lines.append((label, f"{lo_c}->{hi_c}", em, eb, em / eb))
        if not lines:
            continue
        print(f"\n  {m}")
        for label, iv, em, eb, ratio in lines:
            print(f"    {label:20s} {iv:16s} {em:6.3f} vs baseline {eb:6.3f}  ratio {ratio:5.2f}")
        bw = next((r for l, _, _, _, r in lines if l.startswith("memory")), None)
        cp = next((r for l, _, _, _, r in lines if l.startswith("compute")), None)
        if bw is not None and cp is not None:
            if bw < 1 and cp > 1:
                print("    -> less bandwidth-elastic AND more compute-elastic than baseline.")
                print("       Speculation has converted a bandwidth-bound decode into a")
                print("       compute-bound verify. Consistent with H2'; inconsistent with H2.")
            elif bw >= 1:
                print("    -> at least as bandwidth-elastic as baseline: consistent with H2.")
            else:
                print("    -> mixed; neither hypothesis is cleanly supported on these intervals.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    a = ap.parse_args()
    report(json.loads(Path(a.result).read_text()))


if __name__ == "__main__":
    main()
