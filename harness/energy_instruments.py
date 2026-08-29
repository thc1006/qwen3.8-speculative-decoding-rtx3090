#!/usr/bin/env python3
"""Three instruments read the same windows. Which of them disagree, and in what shape?

Every energy number this repository has published comes from `energy_j`: a trapezoid over
nvidia-smi's `power.draw`, which on Ampere is a rolling average of about a second. Two other
readings of the identical window exist in the newer files -- `energy_j_instant` over
`power.draw.instant`, and `energy_j_nvml` from the driver's own cumulative counter -- and nothing
had compared all three across conditions.

`docs/ENERGY.md` already records that the averaged and instantaneous integrals "agree to 0.00-0.34 %
on the baselines and differ by 0.58-1.97 % on the speculative arms". That is true and it is a range
of PERCENTAGES, which is the wrong unit for the thing: this script's first job is to show that the
underlying quantity is a fixed number of JOULES per window, so the percentage is that quantity
divided by however long the window happened to be. A 400-token cap and a 1600-token cap then give
different percentages for the same instrument error, which is what made two correct measurements
look like they contradicted each other.

WHAT WOULD FALSIFY THE PER-WINDOW READING. If the offset were a proportional gain error it would
scale with total energy, so `offset_J` would track `P x span`. If it is a per-window edge effect it
tracks `P` alone and is flat in `span`. Both correlations are printed. A third possibility -- that
the offset tracks how much the power FLUCTUATES -- is tested against SM-clock spread, which unlike
`power_max - power_mean` is not pinned by the power cap.

  energy_instruments.py            all result files that carry the instantaneous field
  energy_instruments.py --files a.json b.json
  energy_instruments.py --stdout
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "analysis" / "energy_instruments.txt"
PROBE_BYTES = 300_000        # cheap prefix test, so 61 MB of JSON is not parsed to find 14 files


def carries_instant(p: Path) -> bool:
    with p.open("rb") as fh:
        return b"energy_instant_vs_average_pct" in fh.read(PROBE_BYTES)


def is_baseline(arm: str) -> bool:
    return arm.split("@")[0].startswith("baseline")


def _tokens(arm):
    """Split an arm name into tokens on the separators this repository uses.

    Token equality, not substring containment. `"pr" in name` matched `baseline@prefill` and any
    other word with those two letters in it; `"pr" in _tokens(name)` matches the tree marker and
    nothing else. Every wrong number this file was written to check came from loose matching.
    """
    out, cur = [], ""
    for ch in arm.lower():
        if ch.isalnum():
            cur += ch
        else:
            if cur:
                out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def _tree(arm):
    """'pr' if the arm is on PR #27342's tree, 'master' if on master, None if unmarked."""
    t = _tokens(arm)
    if "pr" in t or any("27342" in x for x in t) or any(x.startswith("pr27342") for x in t):
        return "pr"
    if "master" in t:
        return "master"
    return None


def _family(arm):
    """'dense', 'moe', or None -- the base model, which the arm name states."""
    t = _tokens(arm)
    if "dense" in t:
        return "dense"
    if "moe" in t:
        return "moe"
    return None


def pair_baseline(sp_row, bases):
    """Which baseline is this speculative arm's control? What the name states, then power.

    Nearest mean power alone gets this wrong in the two files where it matters. In `phase_m.json`
    it paired `dense-draft08b-*` against `baseline-moe`, because those arms run 335-356 W and the
    MoE baseline sits at 357 W while the dense one is at 415 W -- a dense arm compared against a
    different model's baseline. In `phase_a_cap1600.json` the two baselines are 0.1 W apart, so
    "nearest" chose between the master and PR trees on noise, which is Correction 43 section 6's
    confound reintroduced by the script written to check for it. In `phase_l_98304.json` it put
    `mtp-n2@96k` against `baseline@96k-pr` -- an MTP arm, which runs master, against the PR tree.

    The arm name states the model family and, where the file marks it, the tree. Both are used
    before power, and which rule fired is printed beside every row.
    """
    fam, tree = _family(sp_row["arm"]), None
    stem = sp_row["arm"].split("@")[0].lower()
    if stem.startswith("dflash"):
        tree = "pr"
    elif stem.startswith("mtp") or "mtp" in _tokens(sp_row["arm"]):
        tree = "master"

    pool, rule = bases, "nearest power"
    if fam:
        same = [b for b in pool if _family(b["arm"]) == fam]
        if same:
            pool, rule = same, f"same {fam}"
    if tree:
        # Files mark the DEVIATION, not both sides. `phase_l_98304.json` has `baseline@96k` and
        # `baseline@96k-pr`: the PR one is marked and the master one is bare. Requiring an explicit
        # `master` token left every MTP arm there falling through to nearest power, which chose the
        # PR baseline by 1.4 W. So when some baseline in the pool is marked and others are not, the
        # unmarked ones are the other tree.
        marked = {b["arm"]: _tree(b["arm"]) for b in pool}
        if any(v is not None for v in marked.values()):
            same = [b for b in pool if marked[b["arm"]] == tree]
            if not same and any(v is not None and v != tree for v in marked.values()):
                same = [b for b in pool if marked[b["arm"]] is None]
            if same:
                pool = same
                rule = f"{rule}, {tree} tree" if fam else f"{tree} tree"
    return min(pool, key=lambda r: abs(r["w"] - sp_row["w"])), rule


def corr(xs, ys):
    if len(xs) < 3:
        return float("nan")
    mx, my = st.fmean(xs), st.fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")


def load(paths):
    """(file, arm) -> list of power dicts that carry both integrals."""
    cells: dict[tuple[str, str], list[dict]] = {}
    dropped: dict[str, int] = {}
    derived: dict[str, int] = {}
    for p in paths:
        try:
            d = json.loads(p.read_text())
        except Exception as exc:                                   # noqa: BLE001
            print(f"   skipping {p.name}: {exc}", file=sys.stderr)
            continue
        for r in d.get("records") or []:
            pw = r.get("power") or {}
            if not (pw.get("energy_j") and pw.get("energy_j_instant") and pw.get("power_mean_w")):
                dropped[p.name] = dropped.get(p.name, 0) + 1
                continue
            if not pw.get("sample_span_s"):
                # `sample_span_s` postdates the four shorter phase_l runs, and a first version of
                # this script dropped all 720 of their records for want of it -- which silently
                # removed four of the five context lengths from a sweep that then described itself
                # as spanning five. None of the three quantities that matter needs the span: the
                # offset in joules, the offset in per cent and tau are all computed without it. It
                # is used for one display column and one correlation, and the trapezoid divided by
                # the mean power recovers it -- checked against 2730 records in four files that
                # carry both, where the median error is 0.06 % and the worst is 0.65 %.
                pw = dict(pw)
                pw["sample_span_s"] = pw["energy_j"] / pw["power_mean_w"]
                pw["_span_derived"] = True
                derived[p.name] = derived.get(p.name, 0) + 1
            cells.setdefault((p.name, r.get("arm", "?")), []).append(pw)
    return cells, dropped, derived


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
        skipped = []
    else:
        every = sorted(Path(ROOT / "results").glob("*.json"))
        paths = [p for p in every if carries_instant(p)]
        skipped = [p.name for p in every if p not in paths]
    cells, dropped, derived = load(paths)
    if not cells:
        raise SystemExit("no records carry both integrals")

    L: list[str] = []
    W = L.append
    W("=" * 100)
    W("THREE INSTRUMENTS ON THE SAME WINDOWS")
    W("=" * 100)
    W(f"{len(cells)} file-arm cells over {len({f for f, _ in cells})} files, "
      f"{sum(len(v) for v in cells.values())} records.")
    if derived:
        W(f"{sum(derived.values())} records predate `sample_span_s` and have it recovered as "
          "energy_j / power_mean_w, which is the trapezoid over its own mean: "
          + ", ".join(f"{k} {v}" for k, v in sorted(derived.items())))
    if dropped:
        W(f"{sum(dropped.values())} records carry both integrals but no mean power and are not "
          "used: " + ", ".join(f"{k} {v}" for k, v in sorted(dropped.items())))
    if skipped:
        # A prefix probe, so a file that gained the field partway through a run reads as not
        # having it. Naming the skipped files is the difference between a scope and a silent gap.
        W(f"{len(skipped)} result files carry no instantaneous field in their first "
          f"{PROBE_BYTES // 1000} kB and are not read: " + ", ".join(skipped))
    W("")
    W("tau = (instant - averaged) / mean_power: the offset expressed as SECONDS of the window's own")
    W("power. If the offset is a per-window edge effect, tau is a property of the arm and does not")
    W("move when the window length does.")
    W("")
    W(f"  {'file':30s} {'arm':18s} {'n':>4s} {'span s':>7s} {'W':>6s} "
      f"{'offset J':>9s} {'offset %':>9s} {'tau s':>7s}")
    W("  " + "-" * 96)
    rows = []
    for (fn, arm), ps in sorted(cells.items()):
        sp = st.fmean(p["sample_span_s"] for p in ps)
        mw = st.fmean(p["power_mean_w"] for p in ps)
        dj = st.fmean(p["energy_j_instant"] - p["energy_j"] for p in ps)
        dp = st.fmean(p["energy_instant_vs_average_pct"] for p in ps)
        tau = dj / mw if mw else float("nan")
        rows.append({"file": fn, "arm": arm, "n": len(ps), "span": sp, "w": mw,
                     "dj": dj, "dp": dp, "tau": tau, "base": is_baseline(arm)})
        W(f"  {fn[:30]:30s} {arm[:18]:18s} {len(ps):4d} {sp:7.2f} {mw:6.1f} "
          f"{dj:9.2f} {dp:+9.3f} {tau:7.4f}")
    W("")

    b = [r["tau"] for r in rows if r["base"]]
    s = [r["tau"] for r in rows if not r["base"]]
    if b and s:
        W("TAU BY ARM TYPE")
        W(f"  baseline arms     n={len(b):3d}  tau {min(b):.4f} to {max(b):.4f} s  "
          f"median {st.median(b):.4f}")
        W(f"  speculative arms  n={len(s):3d}  tau {min(s):.4f} to {max(s):.4f} s  "
          f"median {st.median(s):.4f}")
        W(f"  separation: {'the ranges do not overlap' if max(b) < min(s) or max(s) < min(b) else 'THE RANGES OVERLAP'}")
        W("")

    # Is tau a property of the arm, or of the window length? The same arm name appearing in files
    # with different token caps is the only place this can be separated, because a speculative arm
    # is always faster than its baseline and so always has the shorter window within one file.
    W("IS TAU A PROPERTY OF THE ARM OR OF THE WINDOW? Same arm at the SAME POWER, different files.")
    W("")
    W("  The power has to be held. Grouping `baseline@pw150` with `baseline@pw420` as one arm would")
    W("  put a 2.8x power range inside what is being called a window-length comparison, and the")
    W("  conclusion would be an artefact of the grouping. Rows are paired only when their mean")
    W("  power is within 5 %, which leaves the comparison this can actually make: the same arm at")
    W("  one power under two different token caps.")
    W("")
    byarm: dict[str, list[dict]] = {}
    for r in rows:
        byarm.setdefault(r["arm"].split("@")[0], []).append(r)
    made = 0
    for arm, rs in sorted(byarm.items()):
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                if a["file"] == b["file"]:
                    continue
                if not a["w"] or abs(a["w"] - b["w"]) / a["w"] > 0.05:
                    continue
                lo, hi = (a, b) if a["span"] < b["span"] else (b, a)
                sr = hi["span"] / lo["span"] if lo["span"] else float("nan")
                tr = (hi["tau"] / lo["tau"]) if lo["tau"] > 0 and hi["tau"] > 0 else float("nan")
                tail = (f"tau x{tr:.2f}" if tr == tr
                        else "tau ratio undefined, one tau is not positive")
                W(f"  {arm:12s} {lo['file'][:22]:22s} {lo['span']:6.2f} s "
                  f"{lo['w']:5.1f} W  tau {lo['tau']:+.4f}")
                W(f"  {'':12s} {hi['file'][:22]:22s} {hi['span']:6.2f} s "
                  f"{hi['w']:5.1f} W  tau {hi['tau']:+.4f}   -> span x{sr:.1f}, {tail}")
                made += 1
    if not made:
        W("  No arm appears at one power in two files. The comparison cannot be made from these")
        W("  files, and nothing below should be read as though it had been.")
    else:
        W("")
        W("  A span multiple far larger than the tau multiple is what a per-window offset looks")
        W("  like. A tau that tracked span would show the two multiples together.")
    W("")

    # The three shapes the offset could have, at record level.
    allp = [p for ps in cells.values() for p in ps]
    dj = [p["energy_j_instant"] - p["energy_j"] for p in allp]
    W(f"WHAT THE OFFSET TRACKS, over all {len(allp)} records:")
    W(f"  offset J  vs  mean power P        r = {corr([p['power_mean_w'] for p in allp], dj):+.3f}"
      "   <- per-window edge effect predicts this")
    W(f"  offset J  vs  total energy P*span r = "
      f"{corr([p['power_mean_w'] * p['sample_span_s'] for p in allp], dj):+.3f}"
      "   <- proportional gain error predicts this")
    W(f"  offset J  vs  window length span  r = "
      f"{corr([p['sample_span_s'] for p in allp], dj):+.3f}")
    clk = [p for p in allp if p.get("sm_clock_mean_mhz") and p.get("sm_clock_min_mhz")]
    if len(clk) > 2:
        W(f"  offset %  vs  SM-clock spread     r = "
          f"{corr([p['sm_clock_mean_mhz'] - p['sm_clock_min_mhz'] for p in clk], [p['energy_instant_vs_average_pct'] for p in clk]):+.3f}"
          "   <- power-fluctuation story predicts this")
    W("  SM-clock spread is the fluctuation proxy that the power cap does NOT pin; power_max minus")
    W("  power_mean is (cap minus mean) whenever the card is at its limit, which is most of a run.")
    W("")

    # The counter, wherever it was read.
    nv = [p for p in allp if p.get("energy_j_nvml") and p.get("n_nvml_samples", 0) >= 2
          and p.get("nvml_vs_integral_pct") is not None]
    if nv:
        W(f"THE DRIVER'S OWN COUNTER, on the {len(nv)} records that carry it")
        W("  Read exactly twice per window. Against BOTH integrals of the same window:")
        W("")
        W(f"  {'file':24s} {'arm':18s} {'n':>4s} {'W':>6s} {'NVML-avg %':>11s} {'NVML-inst %':>12s}")
        W("  " + "-" * 80)
        bycell: dict[tuple[str, str], list[dict]] = {}
        for (fn, arm), ps in cells.items():
            keep = [p for p in ps if p.get("energy_j_nvml") and p.get("n_nvml_samples", 0) >= 2
                    and p.get("nvml_vs_integral_pct") is not None]
            if keep:
                bycell[(fn, arm)] = keep
        for (fn, arm), ps in sorted(bycell.items()):
            na = st.fmean(p["nvml_vs_integral_pct"] for p in ps)
            ni = st.fmean(100.0 * (p["energy_j_nvml"] - p["energy_j_instant"]) / p["energy_j_instant"]
                          for p in ps)
            W(f"  {fn[:24]:24s} {arm[:18]:18s} {len(ps):4d} "
              f"{st.fmean(p['power_mean_w'] for p in ps):6.1f} {na:+11.3f} {ni:+12.3f}")
        ni_all = [100.0 * (p["energy_j_nvml"] - p["energy_j_instant"]) / p["energy_j_instant"]
                  for p in nv]
        na_all = [p["nvml_vs_integral_pct"] for p in nv]
        W("")
        W(f"  counter vs INSTANTANEOUS integral: {st.fmean(ni_all):+.3f} % mean, "
          f"{min(ni_all):+.3f} to {max(ni_all):+.3f}")
        W(f"  counter vs AVERAGED     integral: {st.fmean(na_all):+.3f} % mean, "
          f"{min(na_all):+.3f} to {max(na_all):+.3f}")
        W("  These are two READOUT PATHS over one sensor, not two sensors: a one-second rolling")
        W("  average, a less-smoothed instantaneous reading, and a counter the firmware")
        W("  accumulates. So their agreement bounds the processing and says nothing about the")
        W("  sensor's own calibration -- the proportional +-5 %, bidirectional per board, that")
        W("  the sensor literature reports and only an external meter can resolve. What it does")
        W("  establish is that `power.draw` is the path that departs, and by how much.")
        W("")
        W(f"  regression, NVML-vs-averaged on instant-vs-averaged: "
          f"r = {corr([p['energy_instant_vs_average_pct'] for p in nv], na_all):+.3f}")
        W("  A slope near 1 with r near 1 says the counter's apparent disagreement with `energy_j`")
        W("  IS the averaged field's offset, seen through a second instrument.")
        W("")

    # What it does to a ratio, which is the only thing the published figures are.
    W("DOES IT CANCEL IN A RATIO? A same-board multiplicative gain does. A per-window offset does")
    W("not, and its relative size grows as the window shortens, so the FASTER arm carries more of")
    W("it -- which is the arm every speedup claim here puts in the numerator.")
    W("")
    byfile: dict[str, list[dict]] = {}
    for r in rows:
        byfile.setdefault(r["file"], []).append(r)
    for fn, rs in sorted(byfile.items()):
        bases = [r for r in rs if r["base"]]
        specs = [r for r in rs if not r["base"]]
        if not bases or not specs:
            continue
        for sp_r in specs:
            bs, rule = pair_baseline(sp_r, bases)
            shift = (1.0 + sp_r["dp"] / 100.0) / (1.0 + bs["dp"] / 100.0)
            W(f"  {fn[:26]:26s} {sp_r['arm'][:14]:14s} vs {bs['arm'][:16]:16s} [{rule}]  "
              f"{sp_r['dp']:+6.3f} % vs {bs['dp']:+6.3f} %  ->  energy ratio understated by "
              f"{100.0 * (shift - 1.0):5.3f} %")
    W("")
    W("A -37.1 % saving is a ratio of 0.629. Multiply it by the shift for the matching row above")
    W("to get the ratio the instantaneous field would have reported for the same windows.")

    text = "\n".join(L) + "\n"
    if args.stdout:
        # write, not print: `text` already ends in a newline, and print adds a second one,
        # so --stdout differed from the committed file by one blank line and section 9 -- which
        # compares the two -- called the artifact stale on every run.
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
