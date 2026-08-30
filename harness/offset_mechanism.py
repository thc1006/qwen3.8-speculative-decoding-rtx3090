#!/usr/bin/env python3
"""What is the averaged field's offset, if it is any of the things left?

Correction 44 established what it is NOT, over 89 file-arm cells and 6075
windows: not proportional to the energy (r = +0.120 against `P x span`), not
window length (-0.060), not power fluctuation as measured by SM-clock spread
(-0.096), and not a per-window constant -- a reading with an arm-dependent time
constant was refused by nine files and by the negative offsets on `phase_m`'s
`moe-draft08b-*` arms. Mean power gives +0.542, which is the strongest of a weak
set and cannot be the whole story: within every power cap the arm drawing LESS
power carries the LARGER offset.

Every one of those candidates was tested against a proxy, because the record did
not carry the thing itself. `power_max_w - power_mean_w` looked like a spread and
is not: while the card sits at its limit, max IS the cap, so that difference
measures how far below the cap the mean sits. On the first record of the
2026-08-30 dry run the true spread was 16.11 W and `max - mean` was 7.20 -- the
proxy understated it by a factor of two, and it was the proxy that produced an
r of +0.97 and the appearance of a mechanism.

`power_sd_w` and `power_sd_instant_w` are recorded from 2026-08-30, so the
candidate that could never be tested can be:

    THE OFFSET IS THE VARIATION THE SMOOTHING REMOVED.

`power.draw` is a one-second rolling average and `power.draw.instant` is not, so
`power_sd_instant_w - power_sd_w` is, directly, how much of the trace's movement
the averaging discarded. If integrating a smoothed signal loses energy in
proportion to what the smoothing took out, that difference predicts the offset
and the others do not.

WHAT WOULD FALSIFY IT. A near-zero correlation against that difference, or one
no better than mean power's +0.542, means this candidate joins the other three.
The script prints every competitor beside it rather than the winner alone,
because a single correlation reported on its own is how the first three got as
far as they did.

AND A CORRELATION IS STILL NOT A MECHANISM. If the difference does predict the
offset, the prediction that follows is quantitative: the energy lost should be
about that difference times the window, since a spread in watts integrated over
seconds is joules. That ratio is printed. A number near 1 is a mechanism; a
number that varies with arm or cap is a correlation wearing one.

  offset_mechanism.py results/phase_e2.json [more.json ...]
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
OUT = ROOT / "analysis" / "offset_mechanism.txt"


def corr(xs, ys):
    if len(xs) < 3:
        return float("nan")
    mx, my = st.fmean(xs), st.fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")


def load(paths):
    rows, skipped = [], defaultdict(int)
    for p in paths:
        d = json.loads(Path(p).read_text())
        for r in d.get("records") or []:
            w = r.get("power") or {}
            need = ("energy_j", "energy_j_instant", "power_mean_w",
                    "sample_span_s", "power_sd_w", "power_sd_instant_w")
            if any(w.get(k) is None for k in need):
                skipped[Path(p).name] += 1
                continue
            rows.append({
                "file": Path(p).name, "arm": r.get("arm", "?"),
                "offset_j": w["energy_j_instant"] - w["energy_j"],
                "sd": w["power_sd_w"], "sd_i": w["power_sd_instant_w"],
                "sd_lost": w["power_sd_instant_w"] - w["power_sd_w"],
                "p": w["power_mean_w"], "span": w["sample_span_s"],
                "energy": w["power_mean_w"] * w["sample_span_s"],
            })
    return rows, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--stdout", action="store_true")
    a = ap.parse_args()

    rows, skipped = load(a.files)
    if not rows:
        raise SystemExit("no record carries both spreads; this needs a run made "
                         "after 2026-08-30, when power_sd_w was first recorded")

    L: list[str] = []
    W = L.append
    W("=" * 92)
    W("IS THE OFFSET THE VARIATION THE SMOOTHING REMOVED?")
    W("=" * 92)
    W(f"{len(rows)} records over {len({r['file'] for r in rows})} file(s).")
    if skipped:
        # Records that predate power_sd_w are not silently dropped: a shrinking
        # denominator is how a partial sweep reads as a complete one.
        W("records without both spreads, not used: "
          + ", ".join(f"{k} {v}" for k, v in sorted(skipped.items())))
    W("")

    off = [r["offset_j"] for r in rows]
    cands = [
        ("spread the smoothing removed  (sd_instant - sd)", "sd_lost"),
        ("spread of the instantaneous field  (sd_instant)", "sd_i"),
        ("spread of the averaged field  (sd)", "sd"),
        ("mean power  -- the best of the old set, +0.542", "p"),
        ("total energy  P x span -- a proportional error", "energy"),
        ("window length", "span"),
    ]
    W("  offset in joules against each candidate, over every record:")
    W(f"  {'candidate':52s} {'r':>8s}")
    W("  " + "-" * 62)
    best = None
    for label, key in cands:
        r = corr([x[key] for x in rows], off)
        W(f"  {label:52s} {r:+8.3f}")
        # nan-safe. `abs(nan) > abs(nan)` is False, so seeding `best` with a nan
        # candidate would have pinned it there and reported "strongest: nan".
        if r == r and (best is None or abs(r) > abs(best[1])):
            best = (label, r)
    W("")
    if best is None:
        W("  no candidate produced a usable correlation at all.")
    else:
        W(f"  strongest: {best[0]} at r = {best[1]:+.3f}")
    W("  A candidate that does not beat mean power's +0.542 has not explained")
    W("  anything the previous three did not, and joins them.")
    W("")
    # POOLED correlations can be produced entirely by differences BETWEEN arms
    # while nothing holds within any of them -- and the arms here differ
    # systematically in both quantities, so that is not a remote risk. The
    # within-arm correlation is the one that survives that reading, and it is
    # the check the first three candidates never got.
    W("  the same correlations WITHIN each arm, where between-arm differences")
    W("  cannot produce them. A pooled r that is not reproduced here is a")
    W("  statement about how the arms differ, not about what the offset is.")
    W("")
    W(f"  {'candidate':52s} {'median r':>9s} {'range':>18s}")
    W("  " + "-" * 82)
    _byarm = defaultdict(list)
    for r in rows:
        _byarm[(r["file"], r["arm"])].append(r)
    for label, key in cands:
        rs = [corr([x[key] for x in v], [x["offset_j"] for x in v])
              for v in _byarm.values() if len(v) >= 3]
        rs = [x for x in rs if x == x]
        if not rs:
            W(f"  {label:52s} {'--':>9s} {'(too few records)':>18s}")
            continue
        W(f"  {label:52s} {st.median(rs):+9.3f} "
          f"{f'{min(rs):+.3f} .. {max(rs):+.3f}':>18s}")
    W("")

    # The quantitative prediction. A spread in watts integrated over a window in
    # seconds is joules, so if the offset IS the discarded variation, the ratio
    # below is about 1 and does not move with arm or cap. A ratio that moves is
    # a correlation wearing a mechanism's clothes -- which is what
    # `max - mean` did at r = +0.97.
    W("  THE PREDICTION, WHICH IS WHERE A CORRELATION EARNS THE WORD MECHANISM")
    W("  offset_J / (sd_lost x span): near 1 and flat across arms is a mechanism.")
    W("")
    W(f"  {'arm':22s} {'n':>4s} {'offset J':>9s} {'sd_lost':>8s} {'span s':>7s} {'ratio':>8s}")
    W("  " + "-" * 66)
    by = defaultdict(list)
    for r in rows:
        # keyed by FILE and arm. Two runs of this matrix use the same arm names,
        # and merging them would average an arm across two sessions hours apart
        # while calling it one cell.
        by[(r["file"], r["arm"])].append(r)
    ratios = []
    for (_f, arm), rs in sorted(by.items()):
        o = st.fmean(x["offset_j"] for x in rs)
        sl = st.fmean(x["sd_lost"] for x in rs)
        sp = st.fmean(x["span"] for x in rs)
        ratio = o / (sl * sp) if sl and sp else float("nan")
        if ratio == ratio:
            ratios.append(ratio)
        W(f"  {arm[:22]:22s} {len(rs):4d} {o:9.2f} {sl:8.3f} {sp:7.2f} {ratio:8.3f}")
    if len(ratios) > 1:
        W("")
        W(f"  ratio spans {min(ratios):.3f} to {max(ratios):.3f}, a factor of "
          f"{max(ratios) / min(ratios):.1f} across arms.")
        W("  A factor near 1 is the mechanism. A large one means the correlation")
        W("  above is carried by something the ratio does not hold fixed, and the")
        W("  candidate is not the answer either.")

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
