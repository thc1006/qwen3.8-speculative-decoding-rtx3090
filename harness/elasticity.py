"""Resource-response analysis for Phase R.

Estimates how each method's throughput responds to two resources that this card lets us move
independently: memory clock (at a fixed power cap) and sustained core clock (via the power cap,
at a fixed memory clock).

**Elasticities are reported per interval, never pooled into one number.** The baseline's compute
response measured on this hardware is 0.910 between 386 and 907 MHz but 0.498 between 907 and
1783 MHz - at severe compute starvation even a bandwidth-bound workload becomes compute-limited,
and it transitions back as clock rises. A single pooled figure would average across a regime
change and describe neither end. Cross-method comparisons are therefore only made over the SAME
interval.

The discriminating question, from PREREGISTRATION.md:

    H2'  (quantization x arithmetic intensity) - speculation converts a bandwidth-bound decode
         into a compute-bound verify, so a speculative arm should be LESS bandwidth-elastic and
         MORE compute-elastic than the no-spec baseline.

    H2   (Gated DeltaNet state rollback) - the marginal cost is state reconstruction, i.e. memory
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
    vals = _clock_samples(result, method, cond, key)
    return statistics.fmean(vals) if vals else None


def _clock_samples(result: dict, method: str, cond: str, key: str) -> list[float]:
    """Per-request clock readings, so the denominator's own variance can be bootstrapped."""
    vals = [(r.get("power") or {}).get(key)
            for r in result["records"] if r["arm"] == f"{method}@{cond}"]
    return [v for v in vals if v]


def _paired_elasticity(lo: dict[str, list[float]], hi: dict[str, list[float]],
                       x_lo: float, x_hi: float, *,
                       x_lo_samples: list[float] | None = None,
                       x_hi_samples: list[float] | None = None,
                       n_boot: int = 6000,
                       seed: int = 20260824) -> tuple[float, float, float]:
    """Local elasticity d(ln y)/d(ln x), cluster bootstrap over prompts AND over the denominator.

    Prompts are the unit of replication for the numerator; passes of one prompt are repeated
    measures. The denominator is a measured clock, not a constant, and the first version of this
    function treated it as exact. That understates the interval: at the 250 W condition the
    achieved SM clock has a standard deviation of 4-5 % of its mean, because the card oscillates
    against the cap instead of settling. Passing the per-request clock readings in
    `x_lo_samples` / `x_hi_samples` bootstraps that too.

    Without the samples the behaviour is the old one, and the returned interval is a lower bound
    on the true width.
    """
    tags = sorted(set(lo) & set(hi))
    if not tags or x_lo <= 0 or x_hi <= 0 or x_lo == x_hi:
        return (float("nan"),) * 3

    def point(sample: list[str], xl: float, xh: float) -> float:
        ylo = statistics.fmean([statistics.fmean(lo[t]) for t in sample])
        yhi = statistics.fmean([statistics.fmean(hi[t]) for t in sample])
        return (math.log(yhi) - math.log(ylo)) / (math.log(xh) - math.log(xl))

    est = point(tags, x_lo, x_hi)
    rng = random.Random(seed)
    reps = []
    for _ in range(n_boot):
        smp = [tags[rng.randrange(len(tags))] for _ in tags]
        if x_lo_samples and x_hi_samples:
            xl = statistics.fmean([x_lo_samples[rng.randrange(len(x_lo_samples))]
                                   for _ in x_lo_samples])
            xh = statistics.fmean([x_hi_samples[rng.randrange(len(x_hi_samples))]
                                   for _ in x_hi_samples])
        else:
            xl, xh = x_lo, x_hi
        if xl <= 0 or xh <= 0 or xl == xh:
            continue
        reps.append(point(smp, xl, xh))
    reps.sort()
    if not reps:
        return est, float("nan"), float("nan")
    return est, reps[int(0.025 * len(reps))], reps[int(0.975 * len(reps)) - 1]


def _cross_term_correction(result: dict, method: str, lo_c: str, hi_c: str,
                           moved_key: str, other_key: str, other_elasticity: float
                           ) -> tuple[float, float]:
    """How much of a measured response is really the OTHER resource moving underneath it.

    The bandwidth lever is not clean. At a fixed power cap, raising the memory clock draws power
    away from the core, so the core clock falls across the bandwidth sweep - measured here as
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


# Fallback for Phase R, whose condition names encode the lever that was pulled.
AXES = {
    "memory bandwidth": ("mem_clock_mean_mhz", ["bw-lo", "stock", "bw-hi"]),
    "compute (SM clock)": ("sm_clock_mean_mhz", ["pw-vlo", "pw-lo", "stock"]),
}

MEM_KEY = "mem_clock_mean_mhz"
SM_KEY = "sm_clock_mean_mhz"


def _condition_clocks(result: dict) -> dict[str, tuple[float, float]]:
    """condition -> (median SM MHz, median memory MHz), pooled over methods.

    Pooling is deliberate. A condition is a property of the card, so if two methods disagree
    about the clock a condition produced, that disagreement is itself the finding, and it is
    reported by the interval-matching check rather than hidden by picking one method.
    """
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in result["records"]:
        if "@" not in r["arm"]:
            continue
        cond = r["arm"].split("@", 1)[1]
        pw = r.get("power") or {}
        for k in (SM_KEY, MEM_KEY):
            if pw.get(k):
                acc[cond][k].append(pw[k])
    return {c: (statistics.median(v[SM_KEY]), statistics.median(v[MEM_KEY]))
            for c, v in acc.items() if v.get(SM_KEY) and v.get(MEM_KEY)}


def _infer_axes(result: dict) -> dict[str, tuple[str, list[str]]]:
    """Work out which conditions form a series, from the clocks they actually produced.

    Phase R named its conditions after the lever (`pw-lo`, `bw-hi`); Phase R2 names them after
    the operating point (`sm1200-bwlo`), and it deliberately runs the bandwidth sweep twice, at
    two pinned core clocks. A hard-coded list of names cannot describe both, and silently
    reporting "not enough conditions" for a complete run is the worst way to fail. So the series
    are read off the measured clocks instead, and printed for checking.

    A series varies one clock while holding the other. The thresholds are loose on the held axis
    because a power cap never holds a clock exactly: Phase R's bandwidth sweep moved the core
    clock by 2.2 % as a side effect, which is a confound to correct for, not a reason to refuse
    to call it a bandwidth series.
    """
    clocks = _condition_clocks(result)
    if len(clocks) < 2:
        return {}
    out: dict[str, tuple[str, list[str]]] = {}

    def spread(vals):
        return (max(vals) - min(vals)) / min(vals) if vals and min(vals) else 0.0

    def series(bucket_of, moving_of, held_of, min_move, base, held_name, key):
        """Group conditions by the held clock, keep groups where the other clock really moves."""
        groups: dict[int, list[str]] = defaultdict(list)
        for c in clocks:
            groups[round(held_of(c) / 200)].append(c)      # 200 MHz buckets on the held axis
        found = []
        for _, conds in sorted(groups.items()):
            if len(conds) >= 2 and spread([moving_of(c) for c in conds]) > min_move:
                found.append(sorted(conds, key=moving_of))
        # Only disambiguate by the held clock when there is genuinely more than one series;
        # a lone series does not need its label qualified.
        for conds in found:
            label = base
            if len(found) > 1:
                label += (f" @ {held_name} ~"
                          f"{statistics.median(held_of(c) for c in conds):.0f} MHz")
            out[label] = (key, conds)

    series(None, lambda c: clocks[c][1], lambda c: clocks[c][0], 0.02,
           "memory bandwidth", "core", MEM_KEY)
    series(None, lambda c: clocks[c][0], lambda c: clocks[c][1], 0.10,
           "compute (SM clock)", "memory", SM_KEY)
    return out


def report(result: dict) -> None:
    cells = _cells(result)
    methods = sorted({m for m, _ in cells})
    baseline = "baseline"

    print("=" * 100)
    print("RESOURCE RESPONSE - elasticities per interval (never pooled across a regime change)")
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

    axes = _infer_axes(result) or AXES
    if axes is not AXES:
        print("\n--- series inferred from the clocks each condition actually produced ---")
        for a, (_, cs) in axes.items():
            print(f"  {a}: {' -> '.join(cs)}")

    for axis, (clock_key, order) in axes.items():
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
                e, l, h = _paired_elasticity(
                    cells[(m, lo_c)], cells[(m, hi_c)], x_lo, x_hi,
                    x_lo_samples=_clock_samples(result, m, lo_c, clock_key),
                    x_hi_samples=_clock_samples(result, m, hi_c, clock_key))
                if m == baseline:
                    base_e = e
                rel = f"  ({e/base_e:5.2f}x baseline)" if base_e and m != baseline else ""
                print(f"    {m:10s} {x_lo:6.0f} -> {x_hi:6.0f}  "
                      f"elasticity {e:6.3f} [{l:6.3f}, {h:6.3f}]{rel}")

    # ------------------------------------------------------- cross-term correction
    bw_axes = [a for a in axes if a.startswith("memory bandwidth")]
    core_moved = False
    for a in bw_axes:
        cs = [c for c in axes[a][1] if any((m, c) in cells for m in methods)]
        if len(cs) >= 2:
            cl = _condition_clocks(result)
            sms = [cl[c][0] for c in cs if c in cl]
            if sms and (max(sms) - min(sms)) / min(sms) > 0.005:
                core_moved = True
    if bw_axes and not core_moved:
        print(f"\n{'=' * 100}\n--- the bandwidth lever is clean here ---")
        cl = _condition_clocks(result)
        for a in bw_axes:
            cs = [c for c in axes[a][1] if c in cl]
            if len(cs) >= 2:
                sms = [cl[c][0] for c in cs]
                print(f"  {a}: core clock {min(sms):.0f} to {max(sms):.0f} MHz across the sweep, "
                      f"a spread of {(max(sms)-min(sms))/min(sms)*100:.2f} %.")
        print("  The core clock is pinned rather than left to fall out of a power cap, so raising")
        print("  the memory clock does not take budget from the core and there is no cross term")
        print("  to remove. Phase R needed that correction; this run does not.")
        bw_axes = []

    if bw_axes:
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
    # Whether the methods share an interval is a property of the data, not of the phase, so it
    # is measured rather than asserted. Phase R let a power cap decide the clock and the methods
    # landed on different ones; Phase R2 pins the clock so they cannot. Printing Phase R's caveat
    # over Phase R2's data would describe a defect that run exists to remove.
    # Reported per condition rather than as one verdict, because a pinned run can hold some
    # conditions exactly and lose others: a pin only binds while the power limit does not, so
    # the top of a clock ladder can fall back to being an outcome while the rest stay settings.
    cp_axes = [a for a in axes if a.startswith("compute")]
    rows = []
    for a in cp_axes:
        for c in axes[a][1]:
            per = {m: _clock(result, m, c, SM_KEY) for m in methods
                   if _clock(result, m, c, SM_KEY)}
            if len(per) < 2:
                continue
            spread = (max(per.values()) - min(per.values())) / min(per.values())
            # Within-arm rigidity: mean equal to min proves every sample sat at that value.
            rigid = {}
            for m in per:
                mean_s = _clock_samples(result, m, c, SM_KEY)
                min_s = _clock_samples(result, m, c, "sm_clock_min_mhz")
                if mean_s and min_s:
                    rigid[m] = (statistics.fmean(mean_s) - min(min_s)) / max(min(min_s), 1)
            rows.append((c, per, spread, rigid))
    if rows:
        matched = [r for r in rows if r[2] <= 0.005]
        print(f"\n--- are the compute intervals matched across methods? "
              f"{len(matched)} of {len(rows)} conditions ---")
        print(f"    {'condition':14s} {'spread':>7s} {'drift':>7s}   per-method core clock")
        for c, per, spread, rigid in rows:
            drift = max(rigid.values()) if rigid else 0.0
            mark = "" if spread <= 0.005 else "  <-- not matched"
            print(f"    {c:14s} {spread:6.2%} {drift:6.2%}   "
                  + "  ".join(f"{m}={v:.0f}" for m, v in sorted(per.items())) + mark)
        print("  spread is between methods at the same condition; drift is within an arm,")
        print("  measured as mean minus min, so 0.00 % means every sample sat on the pin.")
        if matched:
            print(f"  The {len(matched)} matched condition(s) give intervals that span the same")
            print("  clock range for every method, so those elasticities are directly comparable.")
        if len(matched) < len(rows):
            print("  Where a condition is not matched the pin did not bind, because the power")
            print("  limit did first. Check the power column in the operating points above: an")
            print("  arm whose peak draw reaches the cap cannot hold its requested clock, and a")
            print("  speculative arm reaches it sooner because it draws more at the same clock.")
            print("  Elasticities crossing such a condition carry that mismatch; ones that do")
            print("  not, do not. Phase R, for contrast, was 30.0 % and 35.8 % mismatched.")

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
