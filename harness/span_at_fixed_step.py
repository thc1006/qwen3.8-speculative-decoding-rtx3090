#!/usr/bin/env python3
"""Does the residual follow the step or the span? Phase E5 could not tell; this can.

Phase E5's only lever was the power cap, and it moves the step the window straddles and the
generation rate together -- Spearman -0.917 across its nine cells. Three models fit those
points about equally and give intercepts of +3.56, +1.38 and +10.00 J, so the split Correction
49 published is a property of the model chosen (Correction 50).

Phase E6 holds the cap and moves the generation length instead: 200, 400 and 800 tokens at the
stock 420 W limit, with Phase E4's 4.0 s roll. The step is whatever the card does between its
shelf and the cap and does not move; the span goes about 12.9 -> 27.5 s.

THE TWO PREDICTIONS, WRITTEN BEFORE THE RUN, using E5's own fitted coefficients and nothing
else. They are not free parameters here -- they were fitted on a different phase.

    STEP-SCALED   residual = 3.56 + 0.0197 x step, and with the step held that is one number
                  at every length: about 9.2 J.
    SPAN-SCALED   residual = 1.38 + 101.14 / span, which falls: about 9.2, 7.1 and 5.1 J.

They agree at the short end by construction, because that is where E5's top cap sat. They
differ by about 4.1 J at the long end, against a round-to-round spread on this residual of
16.6 % at this cap -- about 1.5 J on a 9 J value, so about 0.9 J of standard error on three
rounds.

A THIRD OUTCOME IS POSSIBLE AND IS NOT A FAILURE. If the residual falls but by less than the
span model predicts, both are wrong and the truth is some mixture; the fitted slope on 1/span
here, with the step held, is then the honest estimate of the span-dependent part.

  span_at_fixed_step.py results/phase_e6_tok*_r*.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from edge_model import fit  # noqa: E402
from sampling_rate import round_cv  # noqa: E402  -- the SAMPLE sd
from step_scaling import load_records  # noqa: E402  -- same rows, same levels(), same closed form

ROOT = Path(__file__).parent.parent
OUT = ROOT / "analysis" / "span_at_fixed_step.txt"

# Phase E5's fitted coefficients, PINNED so the prediction is pre-registered and cannot be
# quietly refitted to whatever this phase turns out to show.
E5_STEP_C0, E5_STEP_C1 = 3.56, 0.0197        # J, s   (residual = c0 + c1 x step)
E5_SPAN_C0, E5_SPAN_C1 = 1.38, 101.14        # J, J.s (residual = c0 + c1 / span)
# The check that these still match Phase E5 lives in the test suite, not here. Recomputing
# them would mean reading a 1.5 MB file and deconvolving 225 traces on every invocation --
# seconds of CPU inside an analyser that the gate regenerates and that may run while a
# measurement holds the card. A guard that runs where no card is busy is the right place for
# it: `ThePinnedE5CoefficientsMustStillBeWhatPhaseE5Says`.


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--stdout", action="store_true")
    a = ap.parse_args()
    rows, dropped = load_records(a.files)
    if not rows:
        raise SystemExit("no usable records; this needs Phase E6's files")

    for r in rows:
        r["tok"] = r["file"].split("_tok")[-1].split("_")[0]
        r["round"] = r["file"].rsplit("_r", 1)[-1].split(".")[0]

    L: list[str] = []
    W = L.append
    W("=" * 96)
    W("DOES THE RESIDUAL FOLLOW THE STEP OR THE SPAN? THE STEP IS HELD HERE")
    W("=" * 96)
    W("predictions from Phase E5's nine cell means, pinned in the source so this phase cannot "
      "be scored")
    W("against a moving target, and checked against `analysis/step_scaling.txt` by the test "
      "suite.")
    W("")
    W(f"{len(rows)} records over {len({r['file'] for r in rows})} invocation(s), "
      f"{len({r['arm'] for r in rows})} arm, "
      f"{len({r['tok'] for r in rows})} generation lengths, "
      f"{len({r['round'] for r in rows})} rounds.")
    if dropped:
        W("records without a usable trace, not used: "
          + ", ".join(f"{k} {v}" for k, v in sorted(dropped.items())))
    W("")

    cells = defaultdict(list)
    for r in rows:
        cells[r["tok"]].append(r)

    # A length whose records were mostly dropped is a length that contributes nothing, and a
    # table with one fewer row does not look like a failure. Twice tonight an entire cell fell
    # out of an analysis behind prose that went on describing it, so this says so first.
    # A FRACTION, not a count. The first version flagged any file keeping fewer than 20
    # records and said "each invocation holds 25" -- a number written into the analyser,
    # which fires on every legitimate smaller run and says nothing about a file that dropped
    # four fifths of a larger one. What matters is the share lost, and both halves of it are
    # already here: kept plus dropped is what the file held.
    per_file = defaultdict(int)
    for r in rows:
        per_file[r["file"]] += 1
    bad = []
    for f in sorted(set(per_file) | set(dropped)):
        kept, lost = per_file.get(f, 0), dropped.get(f, 0)
        if kept + lost and lost / (kept + lost) > 0.20:
            bad.append(f"{f} lost {lost} of {kept + lost}")
    if bad:
        W("  *** RECORDS ARE MISSING: " + "; ".join(bad))
        W("  *** A length that lost most of its records is not measured, and a table with one")
        W("  *** short row does not look like a failure. The cells below are not the design.")
        W("")

    W("  1. THE STEP HELD AND THE SPAN MOVED, both measured per record rather than assumed.")
    W("")
    W(f"  {'tokens':>7s} {'n':>4s} {'idle W':>8s} {'load W':>8s} {'step W':>8s} {'span s':>8s} "
      f"{'plateau s':>10s} {'T s':>6s}")
    W("  " + "-" * 76)
    for tok in sorted(cells, key=int):
        v = cells[tok]
        W(f"  {tok:>7s} {len(v):4d} {st.fmean(x['idle'] for x in v):8.1f} "
          f"{st.fmean(x['load'] for x in v):8.1f} {st.fmean(x['step'] for x in v):8.1f} "
          f"{st.fmean(x['span'] for x in v):8.2f} {st.fmean(x['plateau_s'] for x in v):10.2f} "
          f"{st.median(x['T'] for x in v):6.3f}")
    steps = [st.fmean(x["step"] for x in cells[t]) for t in cells]
    spans = [st.fmean(x["span"] for x in cells[t]) for t in cells]
    W("")
    step_pct = 100 * (max(steps) - min(steps)) / st.fmean(steps)
    span_x = max(spans) / min(spans)
    W(f"  the step moves {max(steps) - min(steps):.1f} W across the three, "
      f"{step_pct:.1f} % of its own mean; the span moves {span_x:.2f}x.")
    # A VERDICT, not a number to be read past. The whole phase is the step being held while
    # the span moves, and if that did not happen nothing below separates anything -- which is
    # a thing the report has to say for itself rather than leave to whoever reads the column.
    if step_pct > 8.0 or span_x < 1.6:
        W("")
        W("  *** THE MANIPULATION DID NOT HAPPEN. The step had to stay inside a few per cent")
        W(f"  *** and the span had to move; it moved {step_pct:.1f} % and {span_x:.2f}x. Every")
        W("  *** comparison below is between cells that differ in both, which is the confound")
        W("  *** Phase E5 already had and this phase exists to remove. Do not read them.")
    else:
        W("  Held to under 8 % with the span moved over 1.6x, so the two are separated here.")
    W("")

    W("  2. THE RESIDUAL AGAINST WHAT E5'S TWO MODELS PREDICT FOR IT. Neither is fitted here:")
    W("     both use coefficients from Phase E5 and this phase's own measured step and span.")
    W("")
    W(f"  {'tokens':>7s} {'n':>4s} {'span s':>8s} {'observed J':>11s} {'step model':>11s} "
      f"{'span model':>11s} {'obs-step':>9s} {'obs-span':>9s}")
    W("  " + "-" * 82)
    obs = {}
    for tok in sorted(cells, key=int):
        v = cells[tok]
        o = st.fmean(x["resid"] for x in v)
        sp = st.fmean(x["span"] for x in v)
        stp = st.fmean(x["step"] for x in v)
        pstep = E5_STEP_C0 + E5_STEP_C1 * stp
        pspan = E5_SPAN_C0 + E5_SPAN_C1 / sp
        obs[tok] = (sp, o)
        W(f"  {tok:>7s} {len(v):4d} {sp:8.2f} {o:11.2f} {pstep:11.2f} {pspan:11.2f} "
          f"{o - pstep:+9.2f} {o - pspan:+9.2f}")
    W("")
    if len(obs) >= 2:
        ks = sorted(obs, key=lambda t: obs[t][0])
        lo, hi = obs[ks[0]], obs[ks[-1]]
        W(f"  short to long: the span goes {lo[0]:.1f} -> {hi[0]:.1f} s and the residual "
          f"{lo[1]:.2f} -> {hi[1]:.2f} J, a change of {hi[1] - lo[1]:+.2f}.")
        W(f"  The step model predicts no change. The span model predicts "
          f"{(E5_SPAN_C0 + E5_SPAN_C1 / hi[0]) - (E5_SPAN_C0 + E5_SPAN_C1 / lo[0]):+.2f}.")
        W("")

    W("  3. FITTED HERE, with the step held, so a slope on 1/span is not confounded with it:")
    W("")
    rc = defaultdict(list)
    for r in rows:
        rc[(r["tok"], r["round"])].append(r)
    xs = [1.0 / st.fmean(x["span"] for x in v) for v in rc.values()]
    ys = [st.fmean(x["resid"] for x in v) for v in rc.values()]
    b, c, rr = fit(xs, ys)
    W(f"  residual on 1/span over {len(xs)} (length, round) cell means: "
      f"slope {b:+.2f} J.s, intercept {c:+.2f} J, r {rr:+.3f}")
    W(f"  E5's own 1/span fit, for comparison: slope {E5_SPAN_C1:+.2f}, "
      f"intercept {E5_SPAN_C0:+.2f} -- but there the span was moved BY the cap, which moved")
    W("  the step with it. Here it is not.")
    W("")
    # WHAT ELSE MOVED WITH THE LENGTH. A longer generation is a hotter card, so temperature
    # co-varies with the manipulation exactly as the span does, and nothing in section 2
    # separates them. The within-cell correlation does: inside one length the span is fixed
    # and the card still warms from the first record to the last.
    W("  3c. THE CONFOUND THIS PHASE INTRODUCES. Generating for longer makes the card hotter,")
    W("      so temperature moves with the length just as the span does. Between cells the two")
    W("      cannot be told apart. WITHIN a cell the span is fixed and the card still warms,")
    W("      which is where to look.")
    W("")
    W(f"  {'tokens':>7s} {'n':>4s} {'temp max C':>11s} {'SM MHz':>8s} {'resid J':>9s} "
      f"{'r(res,temp)':>12s} {'r(res,ordinal)':>15s} {'r(temp,ord)':>12s}")
    W("  " + "-" * 88)
    for tok in sorted(cells, key=int):
        v = [x for x in cells[tok] if x.get("temp") is not None]
        if not v:
            W(f"  {tok:>7s} {0:4d}   (no temperature recorded)")
            continue
        o = [x for x in v if x.get("ordinal") is not None]
        _, _, rt = fit([x["temp"] for x in v], [x["resid"] for x in v])
        ro = fit([x["ordinal"] for x in o], [x["resid"] for x in o])[2] if o else float("nan")
        rto = fit([x["ordinal"] for x in o], [x["temp"] for x in o])[2] if o else float("nan")
        clk = [x["sm_clock"] for x in v if x["sm_clock"]]
        W(f"  {tok:>7s} {len(v):4d} {st.fmean(x['temp'] for x in v):11.1f} "
          f"{st.fmean(clk) if clk else float('nan'):8.0f} "
          f"{st.fmean(x['resid'] for x in v):9.2f} {rt:+12.3f} {ro:+15.3f} {rto:+12.3f}")
    W("")
    W("  THE LAST COLUMN IS WHY THE OTHER TWO CANNOT BE READ AS A TEMPERATURE EFFECT. The card")
    W("  warms monotonically through a pass, so inside one cell the temperature and the")
    W("  position in the pass are very nearly the same variable, and a correlation with one is")
    W("  a correlation with the other. This is the same shape of confound Phase E5 had between")
    W("  the step and the span, and naming it is all that can be done with this design.")
    W("")
    W("  What the columns DO settle: a value near zero rules out a strong effect of either,")
    W("  because a real one would have to show up in a correlation with both. A large value")
    W("  settles nothing about which, and would mean section 2's between-cell change cannot be")
    W("  read as a span effect until the two are separated -- which needs the length varied")
    W("  with the thermal history held, and no phase here has done that.")
    W("")

    # THE DECISIVE NUMBER, COMPUTED HERE RATHER THAN IN PROSE. The first write-up of this
    # phase took its error from the spread of the round means POOLED OVER THE THREE LENGTHS,
    # which is the precision of a round mean and not of a length CONTRAST: averaging three
    # lengths cancels the round scatter, and differencing two cells adds their variances. It
    # made the error four times too small and turned t = 2.5 into "5.5 standard errors".
    # The number that decides the phase belongs in the artifact.
    ks = sorted(cells, key=lambda t: st.fmean(x["span"] for x in cells[t]))
    if len(ks) >= 2:
        short, long_ = ks[0], ks[-1]
        rounds = sorted({r["round"] for r in rows})
        per = []
        for rd in rounds:
            # `lo_v`/`hi_v`, not `a`/`b`: `a` is the argparse namespace in this function.
            # edge_model.py had exactly this shadowing in its section 3d, it was fixed with
            # a comment explaining it, and it came back here in the same session. Renaming
            # one instance does not stop the next; the guard that does is
            # `NoAnalyserMayShadowItsArgparseNamespace`.
            lo_v = [x["resid"] for x in cells[short] if x["round"] == rd]
            hi_v = [x["resid"] for x in cells[long_] if x["round"] == rd]
            if lo_v and hi_v:
                per.append(st.fmean(hi_v) - st.fmean(lo_v))
        W("  4. THE CONTRAST, PAIRED WITHIN A ROUND, which is what the models disagree about.")
        W("     Each round holds all three lengths, so differencing inside one removes the")
        W("     session drift; the spread of those differences is the error, and it is larger")
        W("     than the spread of the round means because a difference adds two variances.")
        W("")
        W(f"  {long_} tokens minus {short}, per round: "
          + ", ".join(f"{x:+.2f}" for x in per))
        if len(per) >= 2:
            m, sd = st.fmean(per), st.stdev(per)
            sem = sd / math.sqrt(len(per))
            W(f"  mean {m:+.2f} J, sd {sd:.2f}, sem {sem:.2f} on {len(per)} rounds "
              f"(df = {len(per) - 1})")
            W("")
            sp_lo = st.fmean(x["span"] for x in cells[short])
            sp_hi = st.fmean(x["span"] for x in cells[long_])
            for label, pred in (
                    ("step model, no change", 0.0),
                    ("span model", (E5_SPAN_C0 + E5_SPAN_C1 / sp_hi)
                     - (E5_SPAN_C0 + E5_SPAN_C1 / sp_lo))):
                W(f"    against the {label:22s} predicting {pred:+6.2f} J: "
                  f"t = {(m - pred) / sem if sem else float('nan'):+.2f}")
            W("")
            need = (3.0 * sd / abs(pred)) ** 2 if pred else float("nan")
            W(f"  At this scatter, refusing an effect of {abs(pred):.2f} J at t = 3 would take")
            W(f"  about {need:.0f} rounds. This phase ran {len(per)}. A t of 2.5 on two degrees")
            W("  of freedom is a lean, not a refusal, and saying so is the result.")
            W("")

    W("  5. THE ROUNDS, and the noise floor the difference above has to clear.")
    W(f"  {'round':>6s} {'n':>5s} {'resid J':>9s} {'span s':>8s}")
    W("  " + "-" * 32)
    byr = defaultdict(list)
    for r in rows:
        byr[r["round"]].append(r)
    for k in sorted(byr):
        v = byr[k]
        W(f"  {k:>6s} {len(v):5d} {st.fmean(x['resid'] for x in v):9.2f} "
          f"{st.fmean(x['span'] for x in v):8.2f}")
    W("")
    W(f"  {'tokens':>7s} {'rounds':>7s} {'energy_j CV':>12s} {'resid CV':>10s}")
    W("  " + "-" * 40)
    for tok in sorted(cells, key=int):
        g = defaultdict(list)
        for x in cells[tok]:
            g[x["round"]].append(x)
        if len(g) < 2:
            W(f"  {tok:>7s} {len(g):7d}   (needs two rounds)")
            continue
        ea = [st.fmean(x["e_avg"] for x in q) for q in g.values()]
        rs = [st.fmean(x["resid"] for x in q) for q in g.values()]
        W(f"  {tok:>7s} {len(g):7d} {100 * round_cv(ea):11.3f} % {100 * round_cv(rs):9.2f} %")

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
