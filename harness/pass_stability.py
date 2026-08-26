#!/usr/bin/env python3
"""Which arm-passes disagree with their own repeats, and whether anything explains it.

Written because the analysis behind PREREGISTRATION.md Corrections 15 and 16 was done by hand:
object-file timestamps compared against server-log timestamps, then per-arm mean deltas, then --
after the first reading turned out to be wrong -- per-prompt deltas. That is a lot of manual work
to answer a question the result file can answer itself.

The unit is the arm-pass. For each one, every prompt is compared against the SAME PROMPT in the
arm's other passes, so prompt-to-prompt differences cancel and what is left is how much that
arm-pass disagreed with its own repeats.

Two dispersions, and Correction 16 exists because they are not the same quantity:

  * the MEAN deviation of an arm-pass, one number per arm-pass. Its spread across arm-passes is
    what Correction 15 measured, and one bad arm-pass moves it.
  * the SD of the per-prompt deviations WITHIN an arm-pass, 25 paired values. This is the one that
    says whether an arm-pass was noisy, and on the data that prompted this it showed no group
    effect where the first measure appeared to.

Both are printed. Neither is a verdict on its own.

What the signature does NOT settle is which arms get it. The obvious guess -- that arm-passes doing
more, shorter GPU operations per second leave more room for gaps -- was tested and fails:
corr(within-pass sd, target forwards per second) is +0.08 on Phase M, +0.25 on NMAX, +0.11 on A and
**-0.26** on R2, weak and inconsistent in sign. Throughput does no better (+0.10, +0.43, +0.36,
-0.29). It was a story fitted to which arms happened to be noisy and it is recorded here as
refuted, so it does not get invented again.

Where a run recorded `arm_pass_host_load` -- added after that incident, so not present in files
written before it -- the competing CPU at arm entry is joined on, and an arm-pass that is both an
outlier and contended is named as such rather than left to a timestamp comparison.
"""
from __future__ import annotations

import argparse
import json
import os as _os
import statistics
import sys as _sys
from collections import defaultdict
from pathlib import Path

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import analyze as AN  # noqa: E402


def _corr(x, y):
    """Pearson r, or None when either side has no spread."""
    if len(x) != len(y) or len(x) < 3:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return num / den if den else None


def deviations(result: dict, metric: str = "decode_tok_s"):
    """-> {(arm, pass): {"mean": %, "sd": %, "max": %, "n": prompts}} against the other passes."""
    vals: dict = defaultdict(dict)
    pw: dict = defaultdict(dict)
    clk: dict = defaultdict(dict)
    for rec in result["records"]:
        v = rec.get(metric)
        if v:
            vals[(rec["arm"], rec["pass"])][rec["prompt"]] = float(v)
        p = rec.get("power") or {}
        if p.get("power_mean_w") is not None:
            pw[(rec["arm"], rec["pass"])][rec["prompt"]] = float(p["power_mean_w"])
        if p.get("sm_clock_mean_mhz") is not None:
            clk[(rec["arm"], rec["pass"])][rec["prompt"]] = float(p["sm_clock_mean_mhz"])

    passes_of = defaultdict(list)
    for arm, p in vals:
        passes_of[arm].append(p)

    out = {}
    for (arm, p), series in vals.items():
        others = [q for q in passes_of[arm] if q != p]
        if not others:
            continue
        deltas, dw, dc = [], [], []
        for tag, v in series.items():
            # the same prompt in the arm's other passes, so the comparison is paired
            ref = [vals[(arm, q)][tag] for q in others if tag in vals[(arm, q)]]
            if not ref:
                continue
            deltas.append(100.0 * (v / statistics.fmean(ref) - 1.0))
            # The same pairing over power and clock. Where an arm-pass is noisy, which of these
            # moves with it says what kind of noise it is: power at constant clock is the GPU
            # spending less time idle on identical work, which is not a thing the thermal gate
            # can see.
            for src, dst in ((pw, dw), (clk, dc)):
                cur = src.get((arm, p), {}).get(tag)
                r = [src[(arm, q)][tag] for q in others
                     if tag in src.get((arm, q), {})]
                dst.append((cur - statistics.fmean(r)) if (cur is not None and r) else 0.0)
        if len(deltas) < 2:
            continue
        out[(arm, p)] = {"mean": statistics.fmean(deltas),
                         "sd": statistics.stdev(deltas),
                         "max": max(abs(d) for d in deltas),
                         "n": len(deltas),
                         "r_power": _corr(deltas, dw) if len(dw) == len(deltas) else None,
                         "r_clock": _corr(deltas, dc) if len(dc) == len(deltas) else None}
    return out


def report(result: dict) -> None:
    dev = deviations(result)
    if not dev:
        print("no arm has more than one pass yet, so nothing can be compared against its repeats")
        return

    load = result.get("arm_pass_host_load") or {}
    settle = result.get("arm_pass_settle") or {}

    sds = sorted(d["sd"] for d in dev.values())
    med = statistics.median(sds)
    # Three times the median within-arm-pass scatter, but never below one percent. Descriptive,
    # not a test: there is no null here, the point is to name candidates for a human.
    #
    # The floor is load-bearing rather than cosmetic. Guarding the zero-median case with infinity
    # inverted the intent: a matrix where most arm-passes are perfectly stable has a median of
    # zero, and that is exactly when one wild arm-pass should stand out -- instead nothing could
    # ever be flagged. The floor also stops a very quiet matrix from flagging scatter too small
    # to matter.
    cut = max(3.0 * med, 1.0)

    print("=" * 100)
    print("PASS STABILITY   each arm-pass against the same prompts in its own other passes")
    print("=" * 100)
    print(f"median within-arm-pass sd {med:.2f} %, so an arm-pass is flagged above {cut:.2f} %\n")
    print(f"{'arm':22} {'pass':>4} {'mean':>8} {'sd':>7} {'worst':>8} {'r pwr':>6} "
          f"{'r clk':>6} {'host':>8}  note")

    flagged = []
    for (arm, p), d in sorted(dev.items(), key=lambda kv: -kv[1]["sd"]):
        tag = f"pass{p:02d}_{arm}"
        hl = load.get(tag) or {}
        st = settle.get(tag) or {}
        host = (f"{hl['competing_pct']:.0f}%" if hl.get("competing_pct") is not None else "-")
        temp = (f"{st['entry_temp_c']:.0f}" if st.get("entry_temp_c") is not None else "-")
        # Three reasons to say more about an arm-pass, and only the first is about being unusual.
        # A contended host that did NOT hurt is a finding. And scatter that tracks power rather
        # than clock says what kind of noise it is, which matters at any size -- a median-relative
        # rule cannot flag anything in a matrix of two arm-passes, but the signature is still
        # readable there.
        powerish = ((d.get("r_power") or 0) > 0.7 and abs(d.get("r_clock") or 0) < 0.5
                    and d["sd"] >= 1.0)
        note = ""
        if d["sd"] > cut:
            note = "OUTLIER"
        if hl.get("contended"):
            note = (note + " + host contended").strip(" +")
        if powerish and not note:
            note = "power-bound"
        if note:
            flagged.append((arm, p, d, hl))
        rp = f"{d['r_power']:+.2f}" if d.get("r_power") is not None else "-"
        rc = f"{d['r_clock']:+.2f}" if d.get("r_clock") is not None else "-"
        print(f"{arm:22} {p:>4} {d['mean']:+7.2f}% {d['sd']:6.2f}% {d['max']:7.2f}% "
              f"{rp:>6} {rc:>6} {host:>8}  {note}")

    if not flagged:
        print("\n  No arm-pass exceeds the threshold. Nothing here is evidence that any pass ran "
              "under different conditions from its repeats.")
        return

    print("")
    for arm, p, d, hl in flagged:
        how = "OUTLIER" if d["sd"] > cut else "within the normal range"
        print(f"  {arm} pass {p}: {how}, within-pass scatter {d['sd']:.2f} % against a median of "
              f"{med:.2f} %, worst prompt {d['max']:.2f} %.")
        if hl.get("contended"):
            names = ", ".join(f"{c['comm']} {c['pcpu']:.0f}%" for c in hl.get("competing", []))
            print(f"      the host was contended at arm entry: {names}. That is a recorded cause, "
                  f"not an inference.")
        elif hl:
            print(f"      the host was NOT contended at arm entry "
                  f"({hl.get('competing_pct', 0):.0f}% competing), so the scatter is the arm's own.")
        else:
            # Third branch of the HOST-LOAD chain. It used to be the else of the power-correlation
            # test below, so any arm-pass whose scatter did not track power was told that nothing
            # was recorded about the host -- two lines after this same function had named the
            # process that contended for it. A tool contradicting itself in consecutive sentences.
            print(f"      this file predates arm_pass_host_load, so there is nothing recorded "
                  f"about the host and the cause cannot be settled from the result alone.")
        if (d.get("r_power") or 0) > 0.7 and abs(d.get("r_clock") or 0) < 0.5:
            # r_clock is None when the clock never moved -- which phase_r2 arranges deliberately
            # with nvidia-smi -lgc. That is the stronger case, not a missing one, so it gets a
            # sentence rather than a format specifier that would have crashed on those files.
            clk = (f"not clock (r={d['r_clock']:+.2f})" if d.get("r_clock") is not None
                   else "not clock, which did not move at all")
            print(f"      the scatter tracks POWER (r={d['r_power']:+.2f}) and {clk}. Identical "
                  f"work at the same clock drawing more watts is the GPU idling less, which the "
                  f"thermal gate cannot see.")
    print("\n  An outlier is a candidate, not a verdict. What settles it is another pass: an "
          "arm-pass that\n  is noisy under known-clean conditions was noisy for its own reasons.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("result")
    a = ap.parse_args()
    report(AN.load(Path(a.result)))


if __name__ == "__main__":
    main()
