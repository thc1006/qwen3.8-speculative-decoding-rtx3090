#!/usr/bin/env python3
"""The Phase M replication anchor, computed with the estimator its band was calibrated on.

Phase M swaps the target for the 35B-A3B MoE so that the predecessor study's headline claim can
be re-measured on this harness. Nothing else in Phase M means anything about the predecessor
until that one arm is shown to reproduce, so the check is a gate rather than a line in a report.

WHY THIS IS A SEPARATE FILE. The gate first shipped inline in `run_remaining.sh` as a heredoc,
and it compared a POOLED MEDIAN against a band calibrated on a CLASS-STRATIFIED figure. The two
are not the same quantity. On the first pass of the 2026-08-26 run they differ by 6.6 points
(-73.0 % pooled-median against -66.4 % class-stratified), and the inline gate's own header names
both predecessor figures -- "-10.8 % raw, -21.5 % class-stratified" -- while its band, -12 % to
-32 %, brackets only the second. A perfect replication of the raw number would have FAILED that
gate, and a borderline arm could have been passed or failed on the estimator rather than on the
data. The band is right; the estimator was not.

So the anchor lives here, uses `stats.paired_cluster_bootstrap` exactly as `analyze.py`'s primary
endpoint does, and prints the pooled estimators beside it rather than instead of it, so the size
of the estimator gap is visible instead of being a choice made silently upstream.

A point estimate alone cannot say "the penalty reproduces": an arm can land inside the band with
an interval that also contains zero, which is not evidence of a penalty at all. The gate is
therefore three conditions, reported separately so a near miss can be read rather than guessed at.
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
import stats as ST  # noqa: E402


# The registered anchor, in one place so a test can check the estimator against the band.
#
# `band` was calibrated on `predecessor_stratified`, so `estimator` names the quantity that may
# be compared against it. `predecessor_raw` is carried only to be printed: it is the same effect
# measured a different way, it lies OUTSIDE the band, and that is exactly the trap this file
# exists to close.
ANCHOR = {
    "arm": "moe-draft08b-n8",
    "baseline": "baseline-moe",
    "predecessor_stratified": -21.5,
    "predecessor_raw": -10.8,
    "band": (-32.0, -12.0),
    "estimator": "class-stratified paired mean (mean of per-class mean per-prompt ratios)",
    "source": "PREREGISTRATION.md, Correction 9",
}


def _pooled(records, arm):
    return [r["decode_tok_s"] for r in records
            if r["arm"] == arm and r.get("decode_tok_s")]


def _paired_ratios(series, arm, base, prompt_class):
    """{class: [per-prompt ratio-1]} over the prompts both arms actually cover."""
    a, b = AN._balanced(series[arm], series[base])
    per_class = defaultdict(list)
    for tag in sorted(a):
        av, bv = statistics.fmean(a[tag]), statistics.fmean(b[tag])
        if bv:
            per_class[prompt_class[tag]].append(av / bv - 1.0)
    return per_class, a, b


def verdict(result: dict, anchor: dict = ANCHOR) -> dict:
    """-> a dict carrying every number the report prints, so tests can assert on it."""
    series, prompt_class, excluded, _flagged = AN.build_series(result)
    arm, base = anchor["arm"], anchor["baseline"]
    lo, hi = anchor["band"]

    out = {
        "arm": arm, "baseline": base, "band": (lo, hi),
        "estimator": anchor["estimator"], "source": anchor["source"],
        "predecessor_stratified": anchor["predecessor_stratified"],
        "predecessor_raw": anchor["predecessor_raw"],
        "holds": False, "reason": "", "interval": None,
        "n_excluded": sum(len(v) for v in excluded.values()),
    }
    if arm not in series or base not in series:
        out["reason"] = (f"{arm!r} or {base!r} has no usable record; the anchor cannot be "
                         f"evaluated, which is not the same as failing it")
        return out

    per_class, a_s, b_s = _paired_ratios(series, arm, base, prompt_class)
    if not per_class:
        out["reason"] = "no prompt is covered by both arms; the pairing is empty"
        return out

    out["per_class"] = {k: 100 * statistics.fmean(v) for k, v in sorted(per_class.items())}
    out["n_prompts"] = sum(len(v) for v in per_class.values())
    out["point"] = 100 * ST.stratified_mean(per_class)

    # NOTE argument order: (baseline, arm). Reversed, every sign here inverts.
    iv = ST.paired_cluster_bootstrap(b_s, a_s, prompt_class, relative=True)
    out["interval"] = (iv.point, iv.lo, iv.hi)
    out["spans_zero"] = iv.spans_zero
    out["near_zero"] = iv.near_zero
    out["margin_half_widths"] = iv.margin_half_widths
    out["singleton_classes"] = tuple(iv.singleton_classes)
    out["interval_inside_band"] = (lo <= iv.lo and iv.hi <= hi)
    out["interval_overlaps_band"] = (iv.lo <= hi and lo <= iv.hi)

    # The pooled estimators, printed so the estimator gap is visible rather than assumed small.
    pb, pa = _pooled(result["records"], base), _pooled(result["records"], arm)
    if pb and pa:
        mb, ma = statistics.median(pb), statistics.median(pa)
        out["pooled_median"] = (ma - mb) / mb * 100 if mb else None
        rb, ra = statistics.fmean(pb), statistics.fmean(pa)
        out["pooled_mean"] = (ra - rb) / rb * 100 if rb else None
        out["baseline_tok_s"], out["arm_tok_s"] = mb, ma

    point = out["point"]
    if iv.spans_zero:
        out["reason"] = ("the interval contains zero, so this arm does not show a penalty at "
                         "all, whatever the point estimate lands on")
    elif iv.hi - iv.lo <= 0:
        # Every prompt returned the same ratio, so the resampling found nothing to vary and the
        # interval collapsed onto the point. That prints as perfect precision and means precision
        # was not estimated at all -- the opposite of what it looks like.
        out["reason"] = ("the interval has zero width: every prompt returned the same ratio, so "
                         "no precision was estimated. A collapsed interval is not a tight one")
    elif iv.near_zero:
        # An interval that clears zero by a fraction of a half-width is inside the undercoverage
        # this repo measured at n = 25 (88.0-90.9 % actual against a nominal 95 %). Without this
        # branch an arm at -20 % with an interval of [-80, -1] would have "held": the point sits
        # in the band and the interval technically excludes zero, while the data cannot tell a
        # 1 % penalty from an 80 % one. The same rule analyze.py applies to every other verdict.
        out["reason"] = (f"the interval clears zero by only {iv.margin_half_widths:.2f} "
                         f"half-widths, which is inside the undercoverage measured at this "
                         f"sample size; the penalty is not established well enough to call a "
                         f"replication")
    elif not (lo < point < hi):
        out["reason"] = (f"the class-stratified effect is {point:+.1f} %, outside the registered "
                         f"{lo:+.0f} % to {hi:+.0f} % band")
    else:
        out["holds"] = True
        out["reason"] = "the penalty reproduces on this harness, at the registered magnitude"
    return out


def render(v: dict) -> str:
    lo, hi = v["band"]
    L = [f"REPLICATION ANCHOR  {v['arm']} vs {v['baseline']}",
         f"  registered band   {lo:+.0f} % to {hi:+.0f} %   ({v['source']})",
         f"  estimator         {v['estimator']}",
         f"  predecessor       {v['predecessor_stratified']:+.1f} % class-stratified, "
         f"{v['predecessor_raw']:+.1f} % raw"]
    if v["interval"] is None:
        L += ["", f"  NOT EVALUABLE. {v['reason']}"]
        return "\n".join(L)

    p, ilo, ihi = v["interval"]
    if "baseline_tok_s" in v:
        L += ["", f"  {v['baseline']:22s} {v['baseline_tok_s']:7.1f} tok/s  (pooled median)",
              f"  {v['arm']:22s} {v['arm_tok_s']:7.1f} tok/s  (pooled median)"]
    L += ["", f"  PRIMARY   {v['point']:+.1f} %  [{ilo:+.1f}, {ihi:+.1f}]  "
          f"n={v['n_prompts']} prompts"]
    L.append("  per class " + "  ".join(f"{k} {x:+.1f} %" for k, x in v["per_class"].items()))
    if v.get("pooled_median") is not None:
        gap = v["pooled_median"] - v["point"]
        L += ["", f"  estimator gap: pooled-median {v['pooled_median']:+.1f} %, "
              f"pooled-mean {v['pooled_mean']:+.1f} %, primary {v['point']:+.1f} % "
              f"({gap:+.1f} pt between the first and the primary)",
              "  Only the primary may be compared against the band; the band was calibrated on "
              "that estimator."]
    if v["near_zero"]:
        L.append(f"  COVERAGE: the interval clears zero by only "
                 f"{v['margin_half_widths']:.2f} half-widths, and the percentile bootstrap "
                 f"undercovers at this n. This verdict should not be leaned on.")
    if v["singleton_classes"]:
        L.append(f"  WIDTH UNDERSTATED: single-prompt classes {', '.join(v['singleton_classes'])} "
                 f"contribute no variance.")
    if v["n_excluded"]:
        L.append(f"  {v['n_excluded']} record(s) excluded by the quality rule before this fit.")

    L.append("")
    if v["holds"]:
        L.append(f"  ANCHOR HOLDS. {v['reason']}")
        if not v["interval_inside_band"]:
            L.append("  The point is inside the band but the interval is not contained by it, "
                     "so the magnitude is consistent with the registration rather than pinned "
                     "to it.")
    else:
        L += [f"  ANCHOR DOES NOT HOLD. {v['reason']}.",
              "  Nothing else in Phase M may be read as a statement about the predecessor until "
              "this is understood. The MoE target is kept so it can be chased.",
              "  It remains a valid statement about THIS harness and these configurations."]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("result", nargs="?", default="results/phase_m.json")
    ap.add_argument("--marker", default="results/phase_m_anchor_ok",
                    help="written only when the anchor holds; removed when it does not, so a "
                         "stale marker from an earlier run cannot gate a later one")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    v = verdict(AN.load(Path(args.result)))
    print(json.dumps(v, indent=2, default=str) if args.json else render(v))

    marker = Path(args.marker)
    if v["holds"]:
        marker.write_text(f"{v['point']:+.2f}\n")
    elif marker.exists():
        marker.unlink()
    raise SystemExit(0 if v["holds"] else 1)


if __name__ == "__main__":
    main()
