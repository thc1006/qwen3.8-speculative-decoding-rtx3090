#!/usr/bin/env python3
"""Is the averaged field's offset an energy difference, or an integration artefact?

Phase E2 left one candidate standing. The offset tracks `power_sd_w` at two
scales -- the between-arm ratio `offset / sd` runs 1.57 to 3.30 s, and the
within-arm regression slope reproduces each arm's own ratio to within 10 % in
four arms of six -- which is the test mean power failed. It is still not a
mechanism: the ratio varies 2.1x across arms, the high-spread arms carry
intercepts the spread does not explain, and every between-arm number rests on six
cells however many records sit inside them.

A linear moving average preserves the integral of a stationary signal. So a purely
oscillating trace should contribute NOTHING to a difference between integrating
the smoothed field and the sharp one, and the fact that the difference scales
with the spread at all says either the averaging is not what it is documented to
be, or the offset is not an energy difference: it is what trapezoidal integration
over a fixed grid does to two signals with different frequency content.

`power.draw` is a one-second rolling average whatever rate it is queried at.
`power.draw.instant` is not. So:

    PHYSICAL   both integrals converge as the grid refines; the offset settles;
               `offset / sd` does not move with the sampling rate.
    ARTEFACT   the smooth field is already resolved at 5 Hz and barely moves;
               the sharp one is not, so `energy_j_instant` grows with the rate
               while `energy_j` stays put, and the offset and tau move with it.

Phase E3 varies nothing but the sampler's period. Three intervals over three
rounds with the order rotated, so the interval is not confounded with the part of
the session it ran in -- and this checks that too, because a design is only
balanced if the analysis looks.

THE REQUESTED INTERVAL IS NOT THE ACHIEVED RATE. The sampler queries and then
waits, so the period is the query plus the interval: 0.05 s gives about 14 Hz,
not 20. Everything here is computed against the rate each record actually got,
which is `n_power_samples / sample_span_s`.

  sampling_rate.py results/phase_e3_iv*_r*.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "analysis" / "sampling_rate.txt"


def corr(xs, ys):
    if len(xs) < 3:
        return float("nan")
    mx, my = st.fmean(xs), st.fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")


def round_cv(z):
    """Coefficient of variation across rounds, with the SAMPLE standard deviation.

    This is the noise floor, and the noise floor is the bar: the report below
    says in as many words that "a rate effect on the offset smaller than the
    second number is not an effect". So the direction of any error here is not
    neutral -- understating it lets a rate effect clear a bar that was set too
    low, which is the direction of over-claiming.

    It was `pstdev`. Three rounds are a SAMPLE from the runs this study could
    have made, not the population of them, and pstdev divides by n where the
    sample sd divides by n-1: at n=3 that understates the spread by a factor of
    sqrt(3/2), which is 22 %. Every other analyser in this harness already uses
    `statistics.stdev` for exactly this -- cross_rung, pass_stability, cost_model,
    stats, nvml_polling, exact_forks. The two `pstdev` calls in telemetry.py are
    right, because there the samples collected ARE the population being
    described. This was the only place the two uses were confused.
    """
    if len(z) < 2:
        return float("nan")
    m = st.fmean(z)
    return (st.stdev(z) / m) if m else float("nan")


def load(paths):
    rows, dropped = [], defaultdict(int)
    for p in sorted(paths):
        d = json.loads(Path(p).read_text())
        iv = (d.get("design") or {}).get("power_interval_s")
        rnd = Path(p).stem.rsplit("_r", 1)[-1]
        for r in d.get("records") or []:
            w = r.get("power") or {}
            need = ("energy_j", "energy_j_instant", "power_sd_w",
                    "power_sd_instant_w", "sample_span_s", "n_power_samples")
            # The counter is read EXACTLY TWICE per window, so it is the one
            # reading this experiment cannot move. Optional, because a record
            # without it is still usable for everything else.
            if iv is None or any(w.get(k) is None for k in need) or not w["sample_span_s"]:
                dropped[Path(p).name] += 1
                continue
            rows.append({
                "file": Path(p).name, "arm": r.get("arm", "?"),
                "iv": iv, "round": rnd,
                "hz": w["n_power_samples"] / w["sample_span_s"],
                "e_avg": w["energy_j"], "e_ins": w["energy_j_instant"],
                "off": w["energy_j_instant"] - w["energy_j"],
                "sd": w["power_sd_w"], "sd_i": w["power_sd_instant_w"],
                "span": w["sample_span_s"],
                "nvml": w.get("energy_j_nvml"),
                "tok_s": r.get("decode_tok_s"),
            })
    return rows, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--stdout", action="store_true")
    a = ap.parse_args()
    rows, dropped = load(a.files)
    if not rows:
        raise SystemExit("no usable records; this needs Phase E3's files")

    L: list[str] = []
    W = L.append
    W("=" * 96)
    W("DOES THE OFFSET DEPEND ON HOW OFTEN THE POWER IS SAMPLED?")
    W("=" * 96)
    W(f"{len(rows)} records over {len({r['file'] for r in rows})} invocation(s), "
      f"{len({r['arm'] for r in rows})} arm(s), {len({r['iv'] for r in rows})} intervals.")
    if dropped:
        W("records without the fields this needs, not used: "
          + ", ".join(f"{k} {v}" for k, v in sorted(dropped.items())))
    W("")

    # The requested interval is not the achieved rate, and the difference is not
    # small: the sampler queries and then waits.
    W("  requested interval against the rate it actually achieved")
    W(f"  {'interval s':>11s} {'n':>5s} {'achieved Hz':>13s} {'samples/window':>15s}")
    W("  " + "-" * 48)
    byiv = defaultdict(list)
    for r in rows:
        byiv[r["iv"]].append(r)
    for iv in sorted(byiv):
        v = byiv[iv]
        W(f"  {iv:11.3f} {len(v):5d} {st.fmean(x['hz'] for x in v):13.2f} "
          f"{st.fmean(x['hz'] * x['span'] for x in v):15.1f}")
    W("")

    # The cell is (arm, interval). Records inside one share the arm, the server
    # instance and the session, so they are not independent and the count of
    # cells is the number that matters.
    W("  per arm and interval. THE CELL IS THE UNIT: records inside one share the")
    W("  arm, the server process and the session, so 450 records are not 450")
    W("  independent points and the cell count is what any of this rests on.")
    W("")
    W(f"  {'arm':16s} {'Hz':>6s} {'energy_j':>10s} {'instant':>10s} {'offset J':>9s} "
      f"{'sd':>7s} {'tau=off/sd':>11s} {'tok/s':>7s}")
    W("  " + "-" * 84)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["arm"], r["iv"])].append(r)
    tau_by_arm = defaultdict(dict)
    for (arm, iv), v in sorted(cells.items()):
        hz = st.fmean(x["hz"] for x in v)
        ea = st.fmean(x["e_avg"] for x in v)
        ei = st.fmean(x["e_ins"] for x in v)
        off = st.fmean(x["off"] for x in v)
        sd = st.fmean(x["sd"] for x in v)
        tk = [x["tok_s"] for x in v if x["tok_s"]]
        tau = off / sd if sd else float("nan")
        tau_by_arm[arm][hz] = (tau, ea, ei, off, sd)
        W(f"  {arm[:16]:16s} {hz:6.2f} {ea:10.1f} {ei:10.1f} {off:9.2f} {sd:7.2f} "
          f"{tau:11.3f} {st.fmean(tk) if tk else float('nan'):7.2f}")
    W("")

    # The decisive comparison, stated as the ratio between the extreme rates so
    # that "moves" and "does not move" are numbers rather than adjectives.
    W("  WHAT MOVES WITH THE RATE. Slowest to fastest sampling, per arm:")
    W("")
    W(f"  {'arm':16s} {'Hz lo -> hi':>16s} {'energy_j':>10s} {'instant':>10s} "
      f"{'offset':>10s} {'sd':>9s} {'tau':>9s}")
    W("  " + "-" * 84)
    verdicts = []
    for arm, d in sorted(tau_by_arm.items()):
        hzs = sorted(d)
        if len(hzs) < 2:
            continue
        lo, hi = d[hzs[0]], d[hzs[-1]]
        f = lambda a, b: (b / a) if a else float("nan")   # noqa: E731
        W(f"  {arm[:16]:16s} {f'{hzs[0]:.1f} -> {hzs[-1]:.1f}':>16s} "
          f"{f(lo[1], hi[1]):9.3f}x {f(lo[2], hi[2]):9.3f}x "
          f"{f(lo[3], hi[3]):9.3f}x {f(lo[4], hi[4]):8.3f}x {f(lo[0], hi[0]):8.3f}x")
        verdicts.append((arm, f(lo[1], hi[1]), f(lo[2], hi[2]), f(lo[0], hi[0])))
    W("")
    W("  PHYSICAL predicts every column near 1.000: both integrals converge and")
    W("  the offset is what it is. ARTEFACT predicts energy_j near 1.000 and the")
    W("  instantaneous integral, the offset and tau all moving with the rate.")
    if verdicts:
        W("")
        for arm, ea, ei, tau in verdicts:
            W(f"    {arm[:16]:16s} averaged {ea:.3f}x, instantaneous {ei:.3f}x, "
              f"tau {tau:.3f}x")
    W("")

    # THE NOISE FLOOR, and without it none of the numbers above mean anything.
    #
    # The offset is a DIFFERENCE OF TWO LARGE, NEARLY EQUAL NUMBERS: about 0.5 %
    # of the baseline's energy and 1.4 % of the speculative arm's. So a 0.1 %
    # wobble in either integral -- which is nothing -- moves the offset by 20 %,
    # and a rate-dependence of that size would be indistinguishable from a run
    # having been a run.
    #
    # The rounds measure exactly that. Same arm, same interval, three separate
    # invocations hours apart: the spread across them IS the reproducibility of
    # each integral, and any rate effect has to clear it to be an effect.
    W("  THE NOISE FLOOR. Same arm, same interval, across the three rounds --")
    W("  which is what one integral is reproducible to. The offset is a small")
    W("  difference of two large numbers, so a rate effect must clear this.")
    W("")
    W(f"  {'arm':16s} {'iv':>5s} {'rounds':>7s} {'energy_j CV':>12s} {'instant CV':>11s} "
      f"{'offset CV':>10s} {'offset/energy':>14s}")
    W("  " + "-" * 80)
    floors = []
    rcells = defaultdict(list)
    for r in rows:
        rcells[(r["arm"], r["iv"])].append(r)
    for (arm, iv), v in sorted(rcells.items()):
        byr = defaultdict(list)
        for x in v:
            byr[x["round"]].append(x)
        if len(byr) < 2:
            W(f"  {arm[:16]:16s} {iv:5.2f} {len(byr):7d}   (needs two rounds)")
            continue
        ea = [st.fmean(x["e_avg"] for x in g) for g in byr.values()]
        ei = [st.fmean(x["e_ins"] for x in g) for g in byr.values()]
        of = [st.fmean(x["off"] for x in g) for g in byr.values()]
        cv = round_cv
        frac = st.fmean(of) / st.fmean(ea) if st.fmean(ea) else float("nan")
        floors.append((arm, iv, cv(ea), cv(ei), cv(of), frac))
        W(f"  {arm[:16]:16s} {iv:5.2f} {len(byr):7d} {100 * cv(ea):11.3f} % "
          f"{100 * cv(ei):10.3f} % {100 * cv(of):9.2f} % {100 * frac:13.2f} %")
    if floors:
        W("")
        ecv = [f[2] for f in floors if f[2] == f[2]]
        ocv = [f[4] for f in floors if f[4] == f[4]]
        if ecv and ocv:
            W(f"  An integral reproduces to {100 * st.fmean(ecv):.3f} % across rounds. The offset")
            W(f"  reproduces to {100 * st.fmean(ocv):.1f} %, which is what dividing that wobble by a")
            W("  difference of half a per cent does. A rate effect on the offset")
            W("  smaller than the second number is not an effect.")
        W("")

    # THE TIEBREAKER, and it is the only reading in this experiment that the
    # experiment cannot move. `energy_j_nvml` is the driver's own counter,
    # differenced across the window from exactly two reads, so it does not
    # depend on the sampling rate at all. Whichever integral moves TOWARD it as
    # the grid refines is the one converging on the energy; whichever moves away
    # is the artefact. Correction 44 already established that the counter and
    # the instantaneous integral agree to within 0.15 % at 10 Hz across a 2.75x
    # power range, so the interesting outcome is whether that agreement survives
    # a change of grid or was a coincidence of one.
    nv = [r for r in rows if r.get("nvml") is not None]
    if nv:
        W("  AGAINST THE COUNTER, which is read twice per window and therefore does")
        W("  not move with the sampling rate. Per cent difference from it:")
        W("")
        W(f"  {'arm':16s} {'Hz':>6s} {'n':>4s} {'averaged vs NVML':>18s} "
          f"{'instant vs NVML':>17s}")
        W("  " + "-" * 66)
        nvcells = defaultdict(list)
        for r in nv:
            nvcells[(r["arm"], r["iv"])].append(r)
        drift = defaultdict(list)
        for (arm, iv), v in sorted(nvcells.items()):
            hz = st.fmean(x["hz"] for x in v)
            da = st.fmean(100.0 * (x["e_avg"] - x["nvml"]) / x["nvml"] for x in v)
            di = st.fmean(100.0 * (x["e_ins"] - x["nvml"]) / x["nvml"] for x in v)
            drift[arm].append((hz, da, di))
            W(f"  {arm[:16]:16s} {hz:6.2f} {len(v):4d} {da:+17.3f} % {di:+16.3f} %")
        W("")
        W("  slowest to fastest sampling, how far each integral moved from the counter:")
        for arm, d in sorted(drift.items()):
            d.sort()
            if len(d) < 2:
                continue
            W(f"    {arm[:16]:16s} averaged {d[0][1]:+.3f} -> {d[-1][1]:+.3f} % "
              f"(moved {abs(d[-1][1]) - abs(d[0][1]):+.3f}), "
              f"instant {d[0][2]:+.3f} -> {d[-1][2]:+.3f} % "
              f"(moved {abs(d[-1][2]) - abs(d[0][2]):+.3f})")
        W("")
        W("  A NEGATIVE 'moved' is convergence on the counter. If the instantaneous")
        W("  integral converges while the averaged one does not, the averaged field")
        W("  is the under-resolved reading and the offset is real. If the")
        W("  instantaneous one diverges, it is the grid, and the -36.3 % correction")
        W("  in Correction 44 rests on an artefact.")
        W("")

    # Correlations at record level, reported as such: with three intervals the
    # cell count is six, and a 450-point r would be six points wearing a crowd.
    W("  record-level correlations against the achieved rate, for the shape only:")
    for label, key in (("energy_j", "e_avg"), ("energy_j_instant", "e_ins"),
                       ("offset", "off"), ("power_sd_w", "sd"),
                       ("power_sd_instant_w", "sd_i")):
        W(f"    {label:22s} r = {corr([r['hz'] for r in rows], [r[key] for r in rows]):+.3f}")
    W("")

    # Did the interleaving work? A design is only balanced if someone looks.
    W("  THE ROUNDS, which the rotation exists to make interchangeable. A round")
    W("  effect of the same size as the interval effect would mean the schedule")
    W("  did not do its job.")
    W(f"  {'round':>6s} {'n':>5s} {'offset J':>10s} {'sd':>8s} {'tau':>9s}")
    W("  " + "-" * 42)
    byr = defaultdict(list)
    for r in rows:
        byr[r["round"]].append(r)
    for rn in sorted(byr):
        v = byr[rn]
        off = st.fmean(x["off"] for x in v)
        sd = st.fmean(x["sd"] for x in v)
        W(f"  {rn:>6s} {len(v):5d} {off:10.2f} {sd:8.2f} {off / sd if sd else float('nan'):9.3f}")

    text = "\n".join(L) + "\n"
    if a.stdout:
        sys.stdout.write(text)
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT.with_suffix(".txt.tmp")
        tmp.write_text(text)
        tmp.replace(OUT)
        print(f"   wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
