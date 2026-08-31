#!/usr/bin/env python3
"""Does the residual that survives Phase E4's roll scale with the step the window straddles?

Phase E4 established that `power.draw` is a boxcar average of `power.draw.instant` about
1.0 s wide, that a closed form with no free parameter accounts for the whole unrolled offset,
and that **5.7 J survives on both arms** at the longest roll. It reported that residual as
arm-independent and offered the equality as a reason to think it a different kind of object.
That inference is withdrawn here and this phase is what replaces it.

WHY THE EQUALITY MEANT NOTHING. At roll 4 s both of E4's arms hold the same excursion -- idle,
up to the same 420 W cap, back to idle. Its own running-difference table says so: head terms
+168.67 J against +173.56, middle terms -158.68 against -159.32. Anything whose size is set by
that excursion is PREDICTED to be equal on the two arms, so the equality distinguished nothing.

WHERE THE RESIDUAL IS NOT. Not on the plateau. Splitting each record where the instantaneous
field sits above 80 % of its own maximum, trimmed a second at each end, the two fields differ
there by 0.11 and 0.15 W -- 0.6 to 0.9 J of the 6.4 -- on an arm whose plateau runs 7.7 s and
one whose runs 4.1. A lag cannot produce a plateau term at all, so the residual is at the edges,
which is where the boxcar model already is. What survives is that model being slightly the wrong
SHAPE, not a second mechanism beside it.

WHAT THE COMMITTED DATA COULD NOT SETTLE. Records vary in step on their own, but only from 175
to 248 W -- a factor of 1.4 -- and regressing the residual on it gives +78 ms on one arm and
-214 ms on the other, at r = +0.12 and -0.19. Not a weak answer: no answer.

THE PREDICTIONS, WRITTEN BEFORE THE RUN. The power cap sets how far the card climbs above its
idle-with-model draw of about 128 W, so it sets the step: about 284 W at 420, 121 at 250, 22 at
150, a factor of 13.

    EDGE   the residual is (a lag asymmetry) x (the step), so it FALLS with the cap and
           `residual / step` is one number in milliseconds at all three. Taking E4's 5.7 J at a
           284 W step, the asymmetry is 20 ms and the predictions are 2.4 J at the 250 W cap
           and 0.44 J at the 150 W cap.
    FIXED  the residual is a per-window quantity and stays near 5.7 J at every cap.

The round-to-round spread on the offset at roll 4 was 12 to 30 %, so about 0.8 to 1.9 J on a
6.4 J offset, and three passes put the standard error near 1 J. A 3.3 J separation at the middle
cap and 5.3 J at the lowest is inside what this design can see.

The span moves 4.5x across the caps as well -- 9.9 s at stock against 44.8 s at 150 W -- so a
per-second term is separated in the same run rather than needing another.

  step_scaling.py results/phase_e5.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from edge_model import at, cumulative, fit, fit_window, predicted_loss  # noqa: E402
from sampling_rate import round_cv  # noqa: E402  -- the SAMPLE sd; see its docstring

ROOT = Path(__file__).parent.parent
OUT = ROOT / "analysis" / "step_scaling.txt"
GRID = [round(0.1 + 0.05 * i, 3) for i in range(38)]      # 0.10 to 1.95 s


def levels(ts, wi, avg_first_w):
    """(idle, load, plateau_start, plateau_end), or None if the window holds no step.

    THE IDLE LEVEL IS NOT TAKEN FROM THE TRACE. `avg_first_w` is the averaged field's first
    sample, which by the definition of a T-wide trailing average IS the mean of the
    instantaneous field over the T seconds BEFORE the window -- the level the filter carried
    in, and exactly the term Phase E4's closed form already uses. Reading it off the window's
    own first samples instead needs there to BE some, and there are not: the card goes from
    idle to full load inside one 0.117 s sample, so the plateau starts at index 1 and a guard
    requiring two idle samples first dropped essentially every record at every cap. Caught on
    the real file while the run was still going, not on the synthetic, where the ramp was drawn
    across several samples and the guard never fired.

    The plateau threshold is halfway between that carried-in level and the trace's 95th
    percentile. Not a fraction of the maximum: this phase shrinks the gap between idle and load
    on purpose, and at the 150 W cap the load is about 150 W against a 128 W idle, so 80 % of
    the maximum is 120.6 W -- BELOW idle -- and every sample would count as plateau. The 95th
    percentile rather than the maximum so that one spike cannot set the scale.
    """
    if len(ts) < 20 or avg_first_w is None:
        return None
    q = sorted(wi)
    top = q[(19 * len(q)) // 20]
    if top - avg_first_w < 8.0:
        return None                     # no step for this window to be straddling
    thr = 0.5 * (avg_first_w + top)
    idx = [i for i, w in enumerate(wi) if w >= thr]
    if len(idx) < 8:
        return None
    a, b = ts[idx[0]] + 1.0, ts[idx[-1]] - 1.0
    if b - a < 1.5:
        return None
    cum = cumulative(ts, wi)
    load = (at(ts, cum, b) - at(ts, cum, a)) / (b - a)
    return avg_first_w, load, a, b


def load_records(paths):
    rows, dropped = [], defaultdict(int)
    for p in sorted(paths):
        d = json.loads(Path(p).read_text())
        design = d.get("design") or {}
        roll = design.get("power_roll_s")
        if roll is None:
            raise SystemExit(f"{Path(p).name}: design carries no `power_roll_s`, so this is not "
                             f"a rolled phase and the residual it reports would be the whole "
                             f"edge term rather than what survives one.")
        for r in d.get("records") or []:
            w = r.get("power") or {}
            tr = r.get("power_trace")
            if not tr or w.get("energy_j") is None or w.get("power_first_w") is None:
                dropped[Path(p).name] += 1
                continue
            ts, wi = tr.get("t_instant_s") or [], tr.get("instant_w") or []
            ta, wa = tr.get("t_avg_s") or [], tr.get("avg_w") or []
            lv = levels(ts, wi, w.get("power_first_w"))
            if lv is None or len(ta) < 20:
                dropped[Path(p).name] += 1
                continue
            idle, load, pa, pb = lv
            T, rms, n = fit_window(ts, wi, ta, wa, GRID)
            if not n:
                dropped[Path(p).name] += 1
                continue
            pred = predicted_loss(ts, wi, w["power_first_w"], T)
            if pred is None:
                dropped[Path(p).name] += 1
                continue
            off = w["energy_j_instant"] - w["energy_j"]
            ca = cumulative(ta, wa)
            cin = cumulative(ts, wi)
            plat = ((at(ts, cin, pb) - at(ts, cin, pa))
                    - (at(ta, ca, pb) - at(ta, ca, pa)))
            rows.append({
                "file": Path(p).name, "arm": r.get("arm", "?"), "roll": roll,
                "pass": r.get("pass"), "cap": (r.get("arm") or "@?").split("@")[-1],
                "idle": idle, "load": load, "step": load - idle,
                "T": T, "off": off, "pred": pred, "resid": off - pred,
                "plateau_j": plat, "plateau_s": pb - pa,
                "span": w.get("sample_span_s"), "e_avg": w["energy_j"],
                "tok_s": r.get("decode_tok_s"),
            })
    return rows, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--stdout", action="store_true")
    a = ap.parse_args()
    rows, dropped = load_records(a.files)
    if not rows:
        raise SystemExit("no usable records; this needs a rolled phase with traces")

    L: list[str] = []
    W = L.append
    W("=" * 96)
    W("DOES THE SURVIVING RESIDUAL SCALE WITH THE STEP THE WINDOW STRADDLES?")
    W("=" * 96)
    W(f"{len(rows)} records over {len({r['file'] for r in rows})} file(s), "
      f"{len({r['arm'] for r in rows})} arm(s), roll "
      + ", ".join(f"{v:g} s" for v in sorted({r['roll'] for r in rows})) + ".")
    if dropped:
        W("records without a usable trace, not used: "
          + ", ".join(f"{k} {v}" for k, v in sorted(dropped.items())))
    W("")

    cells = defaultdict(list)
    for r in rows:
        cells[r["arm"]].append(r)

    # ---- 1. WHAT STEP EACH CAP ACTUALLY PRODUCED ----------------------------------------
    W("  1. THE STEP, MEASURED. The cap is the lever but the step is what the model needs, and")
    W("     it is read off each record: the idle mean before the plateau against the load mean")
    W("     inside it, both trimmed a second clear of the filter's ramps.")
    W("")
    W(f"  {'arm':18s} {'n':>4s} {'idle W':>8s} {'load W':>8s} {'step W':>8s} {'span s':>8s} "
      f"{'plateau s':>10s} {'tok/s':>7s}")
    W("  " + "-" * 80)
    for arm, v in sorted(cells.items(), key=lambda kv: -st.fmean(x["step"] for x in kv[1])):
        tk = [x["tok_s"] for x in v if x["tok_s"]]
        W(f"  {arm[:18]:18s} {len(v):4d} {st.fmean(x['idle'] for x in v):8.1f} "
          f"{st.fmean(x['load'] for x in v):8.1f} {st.fmean(x['step'] for x in v):8.1f} "
          f"{st.fmean(x['span'] for x in v):8.2f} {st.fmean(x['plateau_s'] for x in v):10.2f} "
          f"{st.fmean(tk) if tk else float('nan'):7.2f}")
    W("")

    # ---- 2. THE DECISIVE TABLE ----------------------------------------------------------
    W("  2. THE RESIDUAL AGAINST THE STEP. `predicted` is Phase E4's closed form with the width")
    W("     deconvolved from this record's own trace, so nothing here is fitted. EDGE says the")
    W("     last column is one number at every cap; FIXED says the `resid J` column is.")
    W("")
    W(f"  {'arm':18s} {'n':>4s} {'step W':>8s} {'offset J':>9s} {'pred J':>8s} {'resid J':>9s} "
      f"{'resid/step ms':>14s}")
    W("  " + "-" * 82)
    per_arm = {}
    for arm, v in sorted(cells.items(), key=lambda kv: -st.fmean(x["step"] for x in kv[1])):
        step = st.fmean(x["step"] for x in v)
        res = st.fmean(x["resid"] for x in v)
        per_arm[arm] = (step, res, st.fmean(x["off"] for x in v))
        W(f"  {arm[:18]:18s} {len(v):4d} {step:8.1f} {st.fmean(x['off'] for x in v):9.2f} "
          f"{st.fmean(x['pred'] for x in v):8.2f} {res:9.2f} "
          f"{1000 * res / step if step else float('nan'):14.1f}")
    W("")
    if len(per_arm) >= 2:
        ks = sorted(per_arm, key=lambda k: -per_arm[k][0])
        hi, lo = per_arm[ks[0]], per_arm[ks[-1]]
        W(f"  the step falls {hi[0] / lo[0] if lo[0] else float('nan'):.1f}x from the highest cap "
          f"to the lowest; the residual "
          + (f"falls {hi[1] / lo[1]:.1f}x" if lo[1] and hi[1] / lo[1] > 1
             else f"goes {hi[1]:.2f} -> {lo[1]:.2f} J"))
        W("")
        # THE UNIT IS THE (arm, pass) CELL, NOT THE RECORD. Every record inside one shares the
        # cap, the server process and that pass's position in the session, so 225 records are
        # three steps wearing a crowd: a record-level fit reports an r built from within-cell
        # scatter that has nothing to do with the question, and this repository has already
        # published one correlation that was between-arm structure read as though it were
        # within-arm. Nine cell means, and the count is printed so nobody has to guess.
        cellp = defaultdict(list)
        for r0 in rows:
            cellp[(r0["arm"], r0["pass"])].append(r0)
        xs = [st.fmean(x["step"] for x in v) for v in cellp.values()]
        ys = [st.fmean(x["resid"] for x in v) for v in cellp.values()]
        b, c, r = fit(xs, ys)
        W(f"  residual on step over {len(xs)} (arm, pass) cell means, not {len(rows)} records: "
          f"slope {1000 * b:+.1f} ms, intercept {c:+.2f} J, r {r:+.3f}")
        pb, pc, pr = fit([x["step"] for x in rows], [x["resid"] for x in rows])
        W(f"  the record-level fit, for the shape only: slope {1000 * pb:+.1f} ms, "
          f"intercept {pc:+.2f} J, r {pr:+.3f}")
        W("")
        # THE INTERCEPT NEEDS AN UNCERTAINTY OR "clearly non-zero" IS A JUDGEMENT CALL.
        # It comes from the design rather than from a resampling assumption: the three caps
        # are measured once per pass, so fitting the line separately within each pass gives
        # three independent intercepts, and their spread is what one intercept is worth. The
        # same logic as the round-to-round noise floor everywhere else in this harness.
        by_pass = defaultdict(lambda: defaultdict(list))
        for r0 in rows:
            by_pass[r0["pass"]][r0["arm"]].append(r0)
        per_pass = []
        for pid in sorted(by_pass, key=lambda x: (x is None, x)):
            g = by_pass[pid]
            if len(g) < 3:
                continue
            px = [st.fmean(x["step"] for x in v) for v in g.values()]
            py = [st.fmean(x["resid"] for x in v) for v in g.values()]
            sb, sc, _ = fit(px, py)
            if sb == sb:
                per_pass.append((pid, sb, sc))
        if len(per_pass) >= 2:
            W("")
            W("  the same fit inside each pass, which is where the uncertainty comes from --")
            W("  three caps measured once per pass, so these are independent lines:")
            W(f"  {'pass':>6s} {'slope ms':>10s} {'intercept J':>13s}")
            W("  " + "-" * 32)
            for pid, sb, sc in per_pass:
                W(f"  {str(pid):>6s} {1000 * sb:10.1f} {sc:13.2f}")
            ic = [x[2] for x in per_pass]
            sl = [1000 * x[1] for x in per_pass]
            W(f"  intercept {st.fmean(ic):+.2f} J, spread {max(ic) - min(ic):.2f}, "
              f"sd {round_cv(ic) * abs(st.fmean(ic)) if st.fmean(ic) else float('nan'):.2f}")
            W(f"  slope     {st.fmean(sl):+.1f} ms, spread {max(sl) - min(sl):.1f}")
            W("  An intercept whose spread across passes covers zero is not a fixed component")
            W("  this design can see, whatever the pooled fit says.")
        W("")
        W("  EDGE is a slope near the per-cap ratio with an intercept near zero. FIXED is a")
        W("  slope near zero with an intercept near the residual. BOTH is what a slope and an")
        W("  intercept that are each clearly non-zero would mean, and the intercept is then the")
        W("  part of the residual that a step of zero would still produce -- which is the")
        W("  quantity this phase exists to put a number on.")
        W("")

    # ---- 3. THE PLATEAU, WHERE THE SPAN ALSO MOVES --------------------------------------
    W("  3. THE PER-SECOND TERM, separated in the same run. The caps move the span 4.5x as well,")
    W("     so if any of the residual accrues per second rather than per edge it has to show up")
    W("     as a plateau term that grows with the plateau.")
    W("")
    W(f"  {'arm':18s} {'n':>4s} {'plateau s':>10s} {'bias W':>8s} {'plateau J':>10s} "
      f"{'edge J':>8s} {'offset J':>9s}")
    W("  " + "-" * 76)
    for arm, v in sorted(cells.items(), key=lambda kv: -st.fmean(x["step"] for x in kv[1])):
        pj = st.fmean(x["plateau_j"] for x in v)
        ps = st.fmean(x["plateau_s"] for x in v)
        off = st.fmean(x["off"] for x in v)
        W(f"  {arm[:18]:18s} {len(v):4d} {ps:10.2f} {pj / ps if ps else float('nan'):8.3f} "
          f"{pj:10.2f} {off - pj:8.2f} {off:9.2f}")
    W("")

    # ---- 4. THE WIDTH, WHICH MUST NOT MOVE ----------------------------------------------
    W("  4. THE AVERAGING WIDTH, which is a property of the driver and must not move with the")
    W("     cap. If it does, the deconvolution is reading the workload rather than the filter,")
    W("     and every number above rests on it.")
    W("")
    W(f"  {'arm':18s} {'n':>4s} {'median T s':>11s} {'quartiles':>17s} {'at edge':>9s}")
    W("  " + "-" * 62)
    for arm, v in sorted(cells.items(), key=lambda kv: -st.fmean(x["step"] for x in kv[1])):
        Ts = sorted(x["T"] for x in v)
        edge = sum(1 for t in Ts if t <= GRID[0] or t >= GRID[-1])
        W(f"  {arm[:18]:18s} {len(v):4d} {st.median(Ts):11.3f} "
          f"{Ts[len(Ts) // 4]:8.3f} / {Ts[3 * len(Ts) // 4]:<6.3f} {edge:5d}/{len(Ts):<3d}")
    W("")

    # ---- 5. THE PASSES ------------------------------------------------------------------
    W("  5. THE PASSES, which the rotation exists to make interchangeable. One pass per arm, so")
    W("     `rot = (p_idx - 1) % len(arms)` closes and each cap visits each order position")
    W("     exactly once; a pass effect the size of the cap effect would mean it did not work.")
    W(f"  {'pass':>5s} {'n':>5s} {'resid J':>9s} {'offset J':>9s}")
    W("  " + "-" * 32)
    byp = defaultdict(list)
    for r in rows:
        byp[r["pass"]].append(r)
    for k in sorted(byp, key=lambda x: (x is None, x)):
        v = byp[k]
        W(f"  {str(k):>5s} {len(v):5d} {st.fmean(x['resid'] for x in v):9.2f} "
          f"{st.fmean(x['off'] for x in v):9.2f}")
    W("")
    W("  and the noise floor, per arm across the passes:")
    W(f"  {'arm':18s} {'passes':>7s} {'energy_j CV':>12s} {'offset CV':>10s} {'resid CV':>10s}")
    W("  " + "-" * 60)
    for arm, v in sorted(cells.items(), key=lambda kv: -st.fmean(x["step"] for x in kv[1])):
        g = defaultdict(list)
        for x in v:
            g[x["pass"]].append(x)
        if len(g) < 2:
            W(f"  {arm[:18]:18s} {len(g):7d}   (needs two passes)")
            continue
        ea = [st.fmean(x["e_avg"] for x in q) for q in g.values()]
        of = [st.fmean(x["off"] for x in q) for q in g.values()]
        rs = [st.fmean(x["resid"] for x in q) for q in g.values()]
        W(f"  {arm[:18]:18s} {len(g):7d} {100 * round_cv(ea):11.3f} % "
          f"{100 * round_cv(of):9.2f} % {100 * round_cv(rs):9.2f} %")

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
