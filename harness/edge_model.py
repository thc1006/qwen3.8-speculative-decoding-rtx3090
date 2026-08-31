#!/usr/bin/env python3
"""Does the averaged field lose energy at the window's edges, and is the cost a lag?

Phase E3 settled that the offset between integrating `power.draw` and integrating
`power.draw.instant` is a real energy difference rather than an artefact of the grid: the
instantaneous integral does not move across a threefold change of sampling rate and stays
within 0.23 % of the driver's counter, while the averaged one sits below the counter and
drifts further away. What the averaging loses the energy TO was still unmodelled.

THE MODEL. An average smooths and it lags. Smoothing is linear and preserves the integral
of the signal under it, which is why "the offset is the variation the smoothing removed"
could not work and did not (Correction 46). A lag does not preserve it: integrating a signal
delayed by d seconds over [t0, t1] is integrating the undelayed one over [t0 - d, t1 - d],
and the difference is

    offset  =  d * ( p(t1) - p(t0) )

whatever the trace does in between. Per window, not per second. Scaling with the two ENDS,
not with length, mean, spread or total. Free to be NEGATIVE when the window closes lower
than it opened, which `phase_m`'s `dense-draft08b-n4` arm does and no other candidate allows.

WHAT IS TESTED HERE, IN THE ORDER IT DECIDES THINGS.

  1. THE INTERVENTION. `--power-roll S` holds idle inside the window on both sides, so both
     ends sit in one steady state and `p(t1) - p(t0)` goes to zero.
         LAG     the offset collapses toward zero as the roll grows.
         LEAK    it grows, because a per-second loss gets a longer window.
         SPREAD  it is unchanged, because flat idle adds the same to both integrals.
     Three models, three directions, and a result between them is itself a finding.

  2. THE COEFFICIENT. Regress offset on the recorded end-to-end difference of the
     instantaneous field. The model is `offset = d * dp` with NO intercept, so the fit is
     reported WITH one: a large intercept is the pure-lag model failing, and hiding it by
     fitting through the origin would be assuming the answer. One d should serve both arms
     and every roll. The RTX 3090's sensor has a 100 ms update period; the implied d from
     committed data is 0.068 s on the baseline and 0.118 s on `mtp-n2`, which brackets it.

  3. WHERE IN THE WINDOW. A total cannot distinguish a step at the start, a step at the end
     and a drift throughout, and those are three mechanisms. With the traces recorded, the
     running difference between the two integrals is computed directly. The model says
     essentially all of it accrues in the first and last fractions of a second.

  edge_model.py results/phase_e4_roll*_r*.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sampling_rate import round_cv  # noqa: E402  -- the SAMPLE sd; see its docstring

ROOT = Path(__file__).parent.parent
OUT = ROOT / "analysis" / "edge_model.txt"


def through_origin(xs, ys):
    """Least-squares slope with NO intercept, which is the model `offset = d * dp`.

    This is the estimator, and the with-intercept fit below is only a diagnostic. The reason
    is that `dp` barely varies: the window opens near idle and closes at full decode, so it
    is about 380 W on every record and its coefficient of variation is a few per cent. Fitting
    an intercept against an x that does not move puts the whole relationship into the
    intercept and leaves the slope to be estimated from the residual wobble -- on synthetic
    traces built with a planted lag of 0.10 s and 0.25 s, the free fit returned 0.0485 and
    0.0467, which is to say it returned the same wrong number twice and was measuring nothing.
    """
    sxx = sum(x * x for x in xs)
    return (sum(x * y for x, y in zip(xs, ys)) / sxx) if sxx else float("nan")


def ratio_delta(xs, ys):
    """Median of y/x per record: the direct per-window estimate of d, robust to one small x."""
    v = [y / x for x, y in zip(xs, ys) if abs(x) > 1.0]
    return st.median(v) if v else float("nan")


def cv(xs):
    m = st.fmean(xs) if xs else 0.0
    return (st.stdev(xs) / abs(m)) if (len(xs) > 1 and m) else float("nan")


def fit(xs, ys):
    """(slope, intercept, r) by least squares, or (nan, nan, nan) if x does not vary."""
    n = len(xs)
    if n < 3:
        return (float("nan"),) * 3
    mx, my = st.fmean(xs), st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if not sxx:
        return (float("nan"),) * 3
    b = sxy / sxx
    r = sxy / (sxx * syy) ** 0.5 if syy else float("nan")
    return b, my - b * mx, r


def cumulative(ts, ws):
    """Running trapezoid integral of ws over ts, same length as ts, starting at 0."""
    out = [0.0]
    for i in range(1, len(ts)):
        out.append(out[-1] + 0.5 * (ws[i] + ws[i - 1]) * (ts[i] - ts[i - 1]))
    return out


def at(ts, cum, t):
    """Linear interpolation of a cumulative integral at time t, clamped at both ends."""
    if not ts:
        return float("nan")
    if t <= ts[0]:
        return cum[0]
    if t >= ts[-1]:
        return cum[-1]
    lo, hi = 0, len(ts) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ts[mid] <= t:
            lo = mid
        else:
            hi = mid
    f = (t - ts[lo]) / (ts[hi] - ts[lo]) if ts[hi] > ts[lo] else 0.0
    return cum[lo] + f * (cum[hi] - cum[lo])


def boxcar(ts, ws, T, at_t):
    """Mean of the series over [at_t - T, at_t], by trapezoid, or None if not covered."""
    if at_t - T < ts[0] or at_t > ts[-1] or T <= 0:
        return None
    cum = _CUM.get(id(ws))
    if cum is None:
        cum = cumulative(ts, ws)
        _CUM[id(ws)] = cum
    return (at(ts, cum, at_t) - at(ts, cum, at_t - T)) / T


_CUM: dict = {}


def fit_window(ts_i, w_i, ts_a, w_a, grid):
    """The boxcar width T that best reproduces the AVERAGED series from the INSTANT one.

    This is the physical quantity the whole model rests on, measured rather than taken from
    the phrase "a rolling average of about a second". It needs no assumption about the
    window's ends at all: `power.draw` is a filtered version of `power.draw.instant`, so
    applying a boxcar of width T to the second and asking which T reproduces the first is a
    direct read of the filter. Only samples at least T past the start can be compared, since
    the filter's own history before the window is not recorded.

    Returns (best_T, rms_at_best, n_points) or (nan, nan, 0).
    """
    _CUM.clear()
    best, best_err, best_n = float("nan"), float("inf"), 0
    for T in grid:
        errs = []
        for t, want in zip(ts_a, w_a):
            got = boxcar(ts_i, w_i, T, t)
            if got is not None:
                errs.append((got - want) ** 2)
        if len(errs) < 10:
            continue
        rms = (sum(errs) / len(errs)) ** 0.5
        if rms < best_err:
            best, best_err, best_n = T, rms, len(errs)
    _CUM.clear()
    return best, best_err, best_n


def predicted_loss(ts, ws, avg_first_w, T):
    """The boxcar model's closed form for what integrating the averaged field loses.

    For a trailing average of width T over p, integrating it across [t0, t1] weights p by a
    ramp 0->1 across [t0-T, t0], by 1 across the middle, and by 1->0 across [t1-T, t1]. The
    loss against integrating p itself is therefore

        (T/2) * ( mean of p over the last T seconds  -  mean of p over the T BEFORE t0 )

    The second term is not in the window and is not sampled. It does not need to be:
    a T-wide trailing average READ AT t0 is exactly the mean of p over [t0-T, t0], so the
    averaged field's own first sample supplies it. An earlier version used the first T
    seconds INSIDE the window instead and was wrong by a factor of nine on rolled records,
    where the window opens on the request's own ramp -- the two are only the same thing when
    the power is flat across the boundary, which is the case the roll is built to destroy.
    """
    if T <= 0 or len(ts) < 3 or ts[-1] - ts[0] <= T or avg_first_w is None:
        return None
    cum = cumulative(ts, ws)
    tail = (cum[-1] - at(ts, cum, ts[-1] - T)) / T
    return (T / 2.0) * (tail - avg_first_w)


def load(paths):
    rows, notrace = [], defaultdict(int)
    for p in sorted(paths):
        d = json.loads(Path(p).read_text())
        design = d.get("design") or {}
        roll = design.get("power_roll_s")
        if roll is None:
            raise SystemExit(f"{Path(p).name}: design carries no `power_roll_s`. This analysis "
                             f"is about that field; a file without it is not Phase E4 and "
                             f"guessing 0.0 would put an unrolled window in a rolled cell.")
        rnd = Path(p).stem.rsplit("_r", 1)[-1]
        for r in d.get("records") or []:
            w = r.get("power") or {}
            need = ("energy_j", "energy_j_instant", "power_instant_first_w",
                    "power_instant_last_w", "power_first_w", "power_last_w",
                    "sample_span_s", "power_mean_w")
            if any(w.get(k) is None for k in need):
                notrace[Path(p).name] += 1
                continue
            rows.append({
                "file": Path(p).name, "arm": r.get("arm", "?"), "roll": roll, "round": rnd,
                "off": w["energy_j_instant"] - w["energy_j"],
                "dp_inst": w["power_instant_last_w"] - w["power_instant_first_w"],
                "dp_avg": w["power_last_w"] - w["power_first_w"],
                "span": w["sample_span_s"], "P": w["power_mean_w"],
                "e_avg": w["energy_j"], "e_ins": w["energy_j_instant"],
                "nvml": w.get("energy_j_nvml"),
                # The averaged field's FIRST sample, which is the pre-window mean the closed
                # form needs. Named apart from `dp_avg` because it is not an endpoint
                # difference, it is a measurement of something outside the window.
                "avg_first": w.get("power_first_w"),
                "trace": r.get("power_trace"),
            })
    return rows, notrace


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--stdout", action="store_true")
    a = ap.parse_args()
    rows, notrace = load(a.files)
    if not rows:
        raise SystemExit("no usable records; this needs Phase E4's files")

    L: list[str] = []
    W = L.append
    W("=" * 96)
    W("DOES THE AVERAGED FIELD LOSE ENERGY AT THE WINDOW'S EDGES?")
    W("=" * 96)
    W(f"{len(rows)} records over {len({r['file'] for r in rows})} invocation(s), "
      f"{len({r['arm'] for r in rows})} arm(s), rolls "
      + ", ".join(f"{v:g} s" for v in sorted({r['roll'] for r in rows})) + ".")
    if notrace:
        W("records missing a field this needs, not used: "
          + ", ".join(f"{k} {v}" for k, v in sorted(notrace.items())))
    W("")

    # ---- 1. THE INTERVENTION ------------------------------------------------------------
    W("  1. THE INTERVENTION. Idle held inside the window on both sides, so both ends sit in")
    W("     one steady state. LAG predicts the offset collapses, LEAK that it grows, SPREAD")
    W("     that it does not move. The cell is (arm, roll); records inside one share the arm,")
    W("     the server process and the session.")
    W("")
    W(f"  {'arm':16s} {'roll s':>7s} {'n':>4s} {'span s':>7s} {'offset J':>9s} "
      f"{'off/E %':>8s} {'dp inst W':>10s} {'d = off/dp':>11s}")
    W("  " + "-" * 82)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["arm"], r["roll"])].append(r)
    by_arm = defaultdict(dict)
    for (arm, roll), v in sorted(cells.items()):
        off = st.fmean(x["off"] for x in v)
        dp = st.fmean(x["dp_inst"] for x in v)
        ea = st.fmean(x["e_avg"] for x in v)
        by_arm[arm][roll] = off
        W(f"  {arm[:16]:16s} {roll:7.2f} {len(v):4d} {st.fmean(x['span'] for x in v):7.2f} "
          f"{off:9.2f} {100 * off / ea if ea else float('nan'):8.3f} {dp:10.1f} "
          f"{off / dp if dp else float('nan'):11.4f}")
    W("")
    W("  slowest to fastest -- no roll to the longest roll, per arm:")
    for arm, d in sorted(by_arm.items()):
        ks = sorted(d)
        if len(ks) < 2:
            continue
        lo, hi = d[ks[0]], d[ks[-1]]
        W(f"    {arm[:16]:16s} {lo:8.2f} J at roll {ks[0]:g} -> {hi:8.2f} J at roll {ks[-1]:g}"
          f"   ({hi / lo if lo else float('nan'):.3f}x, {hi - lo:+.2f} J)")
    W("")
    W("  A ratio near 0 is LAG. Near 1 is SPREAD. Above 1 is LEAK.")
    W("")

    # ---- 2. THE COEFFICIENT -------------------------------------------------------------
    W("  2. THE COEFFICIENT. offset = d * (p_last - p_first) on the INSTANTANEOUS field, which")
    W("     is the undelayed one. The model has no intercept, so `d origin` and `d ratio` are")
    W("     the estimates and the free fit is a DIAGNOSTIC ONLY: dp is about 380 W on every")
    W("     record because every window opens near idle and closes at full decode, and its own")
    W("     spread is the last column. Fitting an intercept against an x that does not move")
    W("     puts the relationship in the intercept and estimates the slope from the leftovers.")
    W("     On synthetic traces with a lag planted at 0.10 and at 0.25 s the free fit returned")
    W("     0.0485 and 0.0467 -- the same wrong number twice. The two estimators below return")
    W("     the planted value. Read them, not it.")
    W("")
    W(f"  {'subset':26s} {'n':>5s} {'d origin':>9s} {'d ratio':>8s} "
      f"{'free d':>8s} {'free c J':>9s} {'r':>7s} {'dp CV':>7s}")
    W("  " + "-" * 84)
    subsets = [("all records", rows)]
    for arm in sorted({r["arm"] for r in rows}):
        subsets.append((f"arm {arm[:20]}", [r for r in rows if r["arm"] == arm]))
    for roll in sorted({r["roll"] for r in rows}):
        subsets.append((f"roll {roll:g} s", [r for r in rows if r["roll"] == roll]))
    for arm in sorted({r["arm"] for r in rows}):
        for roll in sorted({r["roll"] for r in rows}):
            v = [r for r in rows if r["arm"] == arm and r["roll"] == roll]
            if v:
                subsets.append((f"  {arm[:12]} @ roll {roll:g}", v))
    for label, v in subsets:
        xs = [x["dp_inst"] for x in v]
        ys = [x["off"] for x in v]
        b, c, r = fit(xs, ys)
        W(f"  {label:26s} {len(v):5d} {through_origin(xs, ys):9.4f} {ratio_delta(xs, ys):8.4f} "
          f"{b:8.4f} {c:9.2f} {r:+7.3f} {cv(xs):7.3f}")
    W("")
    W("  The card's sensor has a 100 ms update period. A d near it, the same across arms and")
    W("  rolls, is the model holding. A d that changes with the arm is a per-arm time constant,")
    W("  which nine files already refused once. A d that changes with the ROLL is the model")
    W("  failing outright, because a lag is a property of the instrument and not of the window.")
    W("")

    # ---- 3. WHERE IN THE WINDOW ---------------------------------------------------------
    traced = [r for r in rows if r.get("trace")]

    # The averaging width is fitted FIRST, because both of the sections below are about
    # regions T seconds wide and neither can name its own region without it.
    grid = [round(0.1 + 0.05 * i, 3) for i in range(38)]      # 0.10 to 1.95 s
    fw: dict = defaultdict(list)
    fitted_T: dict = {}
    for r in traced:
        tr = r["trace"]
        ta, wa = tr.get("t_avg_s") or [], tr.get("avg_w") or []
        ti, wi = tr.get("t_instant_s") or [], tr.get("instant_w") or []
        if len(ta) < 20 or len(ti) < 20:
            continue
        T, rms, n = fit_window(ti, wi, ta, wa, grid)
        if n:
            fw[(r["arm"], r["roll"])].append((T, rms))
    for k, v in fw.items():
        fitted_T[k] = st.median([x[0] for x in v])

    if traced:
        W("  3. WHERE IN THE WINDOW the two integrals separate, in JOULES. The boxcar's")
        W("     weight ramps across the first T seconds and the last T, so those are the")
        W("     regions to ask about -- not a round number. T is measured in 3b below and")
        W("     used here; the head window is the first T seconds and the tail the last T.")
        W("")
        W("     Joules, not the fraction of each record's own offset. That fraction is what")
        W("     this printed first and it was unreadable: the roll drives the denominator to")
        W("     near zero ON PURPOSE, so the rows for roll 1.5 and 4 came out at 491.6 % and")
        W("     -299.6 %. A ratio whose denominator the experiment is designed to abolish is")
        W("     not a measurement of anything.")
        W("")
        W(f"  {'arm':16s} {'roll s':>7s} {'n':>4s} {'T s':>5s} {'head J':>8s} {'middle J':>9s} "
          f"{'tail J':>8s} {'total J':>8s}")
        W("  " + "-" * 74)
        tc = defaultdict(list)
        for r in traced:
            tr = r["trace"]
            ta, wa = tr.get("t_avg_s") or [], tr.get("avg_w") or []
            ti, wi = tr.get("t_instant_s") or [], tr.get("instant_w") or []
            if len(ta) < 3 or len(ti) < 3:
                continue
            T = fitted_T.get((r["arm"], r["roll"])) or 1.0
            ca, ci = cumulative(ta, wa), cumulative(ti, wi)
            t_end = min(ta[-1], ti[-1])
            if t_end <= 3 * T:
                continue

            def diff(t):
                return at(ti, ci, t) - at(ta, ca, t)

            total = diff(t_end)
            head = diff(min(T, t_end))
            tail = total - diff(max(t_end - T, 0.0))
            tc[(r["arm"], r["roll"], T)].append((head, total - head - tail, tail, total))
        for k, v in sorted(tc.items()):
            W(f"  {k[0][:16]:16s} {k[1]:7.2f} {len(v):4d} {k[2]:5.2f} "
              f"{st.median([x[0] for x in v]):8.2f} {st.median([x[1] for x in v]):9.2f} "
              f"{st.median([x[2] for x in v]):8.2f} {st.median([x[3] for x in v]):8.2f}")
        W("")
        W("  Medians per column, so they need not sum to the total exactly. The model says")
        W("  the head carries the offset and the middle carries nothing; what the middle")
        W("  carries instead is the residual table 3c also finds.")
        W("")
    else:
        W("  3. WHERE IN THE WINDOW: no record carries a trace. Re-run with --power-trace.")
        W("")

    # ---- 3b. THE FILTER ITSELF ----------------------------------------------------------
    if traced:
        W("  3b. HOW WIDE IS THE AVERAGE? `power.draw` is a filtered `power.draw.instant`, so")
        W("      the width that best reproduces one from the other is a direct read of the")
        W("      filter -- with no assumption about the window's ends, which is what every")
        W("      other line here depends on. The documented figure is 'about a second' and")
        W("      nothing in this repository had measured it. A boxcar of width T behaves at a")
        W("      window edge like a delay of T/2, so this is where the coefficient in table 2")
        W("      should come from.")
        W("")
        W(f"  {'arm':16s} {'roll s':>7s} {'n':>4s} {'median T s':>11s} {'quartiles':>17s} "
          f"{'rms W':>7s} {'at edge':>8s}")
        W("  " + "-" * 72)
        for k, v in sorted(fw.items()):
            Ts = sorted(x[0] for x in v)
            edge = sum(1 for t in Ts if t <= grid[0] or t >= grid[-1])
            W(f"  {k[0][:16]:16s} {k[1]:7.2f} {len(v):4d} {st.median(Ts):11.3f} "
              f"{Ts[len(Ts) // 4]:8.3f} / {Ts[3 * len(Ts) // 4]:<6.3f} "
              f"{st.median([x[1] for x in v]):7.2f} {edge:5d}/{len(Ts):<3d}")
        W("")
        W("  A T that is the same on both arms and at every roll is a property of the driver,")
        W("  which is what it should be. One that moves with the arm is not a filter width and")
        W("  the name would be wrong.")
        W("")
        W("  `at edge` counts records whose best width lands on the grid's first or last value.")
        W("  Those are not evidence of a wider filter, they are records the deconvolution")
        W("  cannot pin -- and a nearly constant trace is exactly that, because a flat signal")
        W("  carries no information about the width of a filter applied to it, so every T fits")
        W("  it about equally well and the argmin follows the noise. Checked rather than")
        W("  asserted, on the 75 unrolled baseline@pw420 records: the 13 that sit at the")
        W("  ceiling already fit no better than the rest -- best rms 1.20 W against 1.18 --")
        W("  and forcing them to 1.00 s costs a median 0.50 W more, where the 62 that do not")
        W("  hit it pay 0.11 W. The two penalties overlap, 0.66 W at worst against 0.60 W, on")
        W("  a 410 W signal. A shallow surface, not a wider filter. Read the median; the")
        W("  count is here so that a cell where MOST records hit an edge cannot be read as a")
        W("  measurement at all -- and every cell with a roll is 0 of 75.")
        W("")

    # ---- 3c. THE MODEL IN CLOSED FORM ---------------------------------------------------
    if traced:
        W("  3c. THE WHOLE MODEL, PREDICTED PER RECORD AND COMPARED. With T measured above,")
        W("      the boxcar's loss has a closed form and no free parameter left:")
        W("")
        W("          loss = (T/2) * ( mean of the last T s inside  -  the averaged field's")
        W("                           own first sample )")
        W("")
        W("      the second term being, by definition of a T-wide trailing average, the mean")
        W("      of the instantaneous field over the T seconds BEFORE the window -- which is")
        W("      not sampled and does not need to be. A ratio near 1.00 at roll 0 is the model")
        W("      accounting for the offset. What survives at the longer rolls is what it does")
        W("      not account for, and that residual is the phase's second result.")
        W("")
        W(f"  {'arm':16s} {'roll s':>7s} {'n':>4s} {'observed J':>11s} {'predicted J':>12s} "
          f"{'ratio':>6s} {'residual J':>11s}")
        W("  " + "-" * 74)
        pred_cells = defaultdict(list)
        for r in traced:
            tr = r["trace"]
            ti, wi = tr.get("t_instant_s") or [], tr.get("instant_w") or []
            if len(ti) < 20:
                continue
            T = fitted_T.get((r["arm"], r["roll"]))
            if not T:
                continue
            pl = predicted_loss(ti, wi, r.get("avg_first"), T)
            if pl is None:
                continue
            pred_cells[(r["arm"], r["roll"])].append((r["off"], pl))
        for k, v in sorted(pred_cells.items()):
            o = st.fmean(x[0] for x in v)
            q = st.fmean(x[1] for x in v)
            W(f"  {k[0][:16]:16s} {k[1]:7.2f} {len(v):4d} {o:11.2f} {q:12.2f} "
              f"{q / o if o else float('nan'):6.2f} {o - q:11.2f}")
        W("")

    # ---- 3d. WHAT THE SURVIVING RESIDUAL IS NOT -----------------------------------------
    if traced:
        W("  3d. WHAT THE SURVIVING RESIDUAL IS NOT. Two things the rolled rows above could")
        W("      mean, separated. The plateau is where the instantaneous field sits above 80 %")
        W("      of its own maximum, trimmed by one second at each end so the filter's ramps")
        W("      are outside it; on that stretch the two fields should agree, because a delayed")
        W("      copy of a flat signal is the same flat signal. Whatever the two fields differ")
        W("      by there is a PER-SECOND term. Everything else is at the edges.")
        W("")
        W(f"  {'arm':16s} {'roll s':>7s} {'n':>4s} {'plateau s':>10s} {'bias W':>8s} "
          f"{'plateau J':>10s} {'edge J':>8s} {'offset J':>9s}")
        W("  " + "-" * 78)
        split = defaultdict(list)
        for r in traced:
            tr = r["trace"]
            ts, wi = tr.get("t_instant_s") or [], tr.get("instant_w") or []
            ta, wa = tr.get("t_avg_s") or [], tr.get("avg_w") or []
            if len(ts) < 20 or len(ta) < 20:
                continue
            thr = 0.8 * max(wi)
            idx = [i for i, w in enumerate(wi) if w >= thr]
            if len(idx) < 10:
                continue
            # `pa`/`pb`, not `a`/`b`: `a` is the argparse namespace in this function, and
            # the first version of this block shadowed it with a float. The report still
            # built -- every table above is computed before the name is used again -- and
            # then died on `a.stdout` at the very end, so the failure was invisible until
            # the artifact was regenerated. Which is the point: a section was added and its
            # artifact was not rebuilt in the same breath.
            pa, pb = ts[idx[0]] + 1.0, ts[idx[-1]] - 1.0
            if pb - pa < 2.0:
                continue
            ci, ca = cumulative(ts, wi), cumulative(ta, wa)
            mi = (at(ts, ci, pb) - at(ts, ci, pa)) / (pb - pa)
            ma = (at(ta, ca, pb) - at(ta, ca, pa)) / (pb - pa)
            split[(r["arm"], r["roll"])].append(
                (pb - pa, mi - ma, (mi - ma) * (pb - pa), r["off"]))
        for k, v in sorted(split.items()):
            pl = st.fmean(x[2] for x in v)
            off = st.fmean(x[3] for x in v)
            W(f"  {k[0][:16]:16s} {k[1]:7.2f} {len(v):4d} {st.fmean(x[0] for x in v):10.2f} "
              f"{st.fmean(x[1] for x in v):8.3f} {pl:10.2f} {off - pl:8.2f} {off:9.2f}")
        W("")
        W("  So the residual is NOT a per-second loss: the plateau carries under a joule of it")
        W("  on an arm whose plateau runs eight seconds and on one whose plateau runs four.")
        W("  It is at the edges, which is where the boxcar model already is -- so what survives")
        W("  is that model being slightly the wrong SHAPE, not a second mechanism beside it.")
        W("")
        W("  And the two arms carrying the same residual says nothing on its own. At the longest")
        W("  roll both windows hold the same excursion, idle to the same 420 W cap and back:")
        W("  section 3's head terms are within 3 % of each other and its middle terms within")
        W("  0.5 %. Anything whose size is set by that excursion is PREDICTED to be equal on the")
        W("  two arms, so the equality distinguishes nothing. What would distinguish it is")
        W("  varying the excursion on purpose, which is Phase E5.")
        W("")

    # ---- 4. THE NOISE FLOOR -------------------------------------------------------------
    W("  4. THE NOISE FLOOR. Same arm, same roll, across the three rounds. The offset is a")
    W("     small difference of two large numbers, so a change smaller than this is not one.")
    W("")
    W(f"  {'arm':16s} {'roll s':>7s} {'rounds':>7s} {'energy_j CV':>12s} {'offset CV':>10s}")
    W("  " + "-" * 58)
    rc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        rc[(r["arm"], r["roll"])][r["round"]].append(r)
    for k, byr in sorted(rc.items()):
        if len(byr) < 2:
            W(f"  {k[0][:16]:16s} {k[1]:7.2f} {len(byr):7d}   (needs two rounds)")
            continue
        ea = [st.fmean(x["e_avg"] for x in g) for g in byr.values()]
        of = [st.fmean(x["off"] for x in g) for g in byr.values()]
        W(f"  {k[0][:16]:16s} {k[1]:7.2f} {len(byr):7d} {100 * round_cv(ea):11.3f} % "
          f"{100 * round_cv(of):9.2f} %")
    W("")

    # ---- 5. THE ROUNDS ------------------------------------------------------------------
    W("  5. THE ROUNDS, which the rotation exists to make interchangeable. A round effect the")
    W("     size of the roll effect would mean the schedule did not do its job.")
    W(f"  {'round':>6s} {'n':>5s} {'offset J':>10s} {'span s':>8s}")
    W("  " + "-" * 34)
    byr = defaultdict(list)
    for r in rows:
        byr[r["round"]].append(r)
    for rn in sorted(byr):
        v = byr[rn]
        W(f"  {rn:>6s} {len(v):5d} {st.fmean(x['off'] for x in v):10.2f} "
          f"{st.fmean(x['span'] for x in v):8.2f}")

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
