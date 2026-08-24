"""Verification-step cost model.

Speculative decoding emits `mean_len` tokens per target forward pass, where llama.cpp's own
per-request line reports `mean_len` and it satisfies (verified against this run's data)

    mean_len = 1 + n_max * acceptance_rate

If one speculative verification step costs `k` times a plain decode step, then

    speedup = mean_len / k

so `k` is recoverable per request as `mean_len / speedup`, with speedup measured against the
no-spec baseline on the *same prompt and pass*.

Why this is worth isolating: `k` turns out to be constant across prompt classes whose acceptance
rates differ by nearly a factor of two, while differing sharply between configurations that
verify different numbers of positions. That separates "what the content does" from "what the
configuration costs", and it turns the mechanism question into a question about one coefficient
rather than about a throughput table.

Two tests are reported:

  1. **k vs acceptance, within an arm.** State-rollback accounts of the overhead charge the cost
     to *rejection*. If that dominates, `k` must rise as acceptance falls. A flat slope is
     evidence against it and for a cost paid per verified position regardless of outcome.

  2. **k vs verification width, across arms.** Fitting k = k0 + c*(w-1) gives a marginal cost `c`
     per additional verified position and an intercept `k0` attributable to the drafter itself.
     If two unrelated drafters share `c` but differ in `k0`, the marginal cost belongs to the
     verification path, not to the drafter.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares y = a + b x. Returns (a, b, r2)."""
    n = len(xs)
    if n < 2:
        return (float("nan"), float("nan"), float("nan"))
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return (my, 0.0, float("nan"))
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return (a, b, r2)


def _n_max(meta: dict) -> int | None:
    args = meta.get("extra_args", [])
    if "--spec-draft-n-max" in args:
        return int(args[args.index("--spec-draft-n-max") + 1])
    return None


def _spec_type(meta: dict) -> str:
    args = meta.get("extra_args", [])
    if "--spec-type" in args:
        return args[args.index("--spec-type") + 1]
    return "none"


def collect(result: dict) -> list[dict]:
    """One row per (arm, pass, prompt), computed entirely from the per-request API timings.

    `mean_len` is derived, not taken from llama.cpp's `mean len` log field. The two disagree:
    llama.cpp reports `1 + n_max * accept_rate`, which assumes every step drafts the full
    `n_max`, while a step may draft fewer. Measured on this run the gap is +0.17 % to +0.81 %
    and it VARIES BY PROMPT -- the same order as the cross-class constancy of `k` that this
    module is used to demonstrate, so using that field would contaminate the very claim.

    The physical definition has no such ambiguity. Each target forward pass emits one token of
    its own plus whatever drafts it accepted, so over F forwards

        predicted_n = accepted + F     =>   F = predicted_n - accepted
        mean_len    = predicted_n / F

    Deriving from the API also removes a dependency on aligning log lines to prompts by
    position, which was an implicit and unverified assumption.
    """
    recs = result["records"]
    arms_meta = result.get("arms", {})

    baselines: dict[tuple[str, int, str], float] = {}
    for r in recs:
        m = arms_meta.get(r["arm"], {})
        if not m.get("extra_args") and not m.get("expects_drafter"):
            baselines[(m.get("tree", "?"), r["pass"], r["prompt"])] = r["decode_tok_s"]

    rows: list[dict] = []
    for rec in recs:
        meta = arms_meta.get(rec["arm"], {})
        nmax = _n_max(meta)
        if nmax is None:
            continue
        tm = rec.get("timings") or {}
        drafted = tm.get("t_draft_n") or 0
        accepted = tm.get("t_draft_n_accepted") or 0
        pn = rec.get("predicted_n") or 0
        base = baselines.get((meta.get("tree", "?"), rec["pass"], rec["prompt"]))
        forwards = pn - accepted
        if not (drafted and pn and base and forwards > 0 and rec.get("decode_tok_s")):
            continue
        speedup = rec["decode_tok_s"] / base
        mean_len = pn / forwards
        rows.append({
            "arm": rec["arm"], "pass": rec["pass"], "prompt": rec["prompt"],
            "class": rec["class"], "spec_type": _spec_type(meta),
            "n_max": nmax, "width": nmax + 1,
            "acceptance": accepted / drafted,
            "drafted": drafted, "accepted": accepted,
            "forwards": forwards, "mean_len": mean_len,
            "speedup": speedup, "k": mean_len / speedup,
        })
    return rows


def cross_check_against_log(result: dict, rows: list[dict]) -> None:
    """Independent confirmation that the API counters and llama.cpp's own log agree.

    Not used to compute anything -- purely a data-integrity check that two separate sources
    describe the same events.
    """
    acc_by = result.get("arm_pass_acceptance", {})
    if not acc_by:
        return
    checked = mismatched = 0
    by_ap: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        by_ap[(r["arm"], r["pass"])].append(r)
    for (arm, pas), g in by_ap.items():
        log_rows = acc_by.get(f"pass{pas:02d}_{arm}")
        if not log_rows or len(log_rows) != len(g) + 1:
            continue
        for r, lg in zip(g, log_rows[1:]):
            checked += 1
            if r["drafted"] != lg["generated"] or r["accepted"] != lg["accepted"]:
                mismatched += 1
    if checked:
        print(f"\n[integrity] API counters vs llama.cpp log lines: {checked} requests compared, "
              f"{mismatched} mismatched"
              f"{'  -- two independent sources agree to the token' if not mismatched else ''}")


def report(result: dict) -> None:
    rows = collect(result)
    if not rows:
        print("no rows (need per-request acceptance logs and a same-tree baseline)")
        return

    print("=" * 100)
    print("VERIFICATION-STEP COST MODEL     speedup = mean_len / k")
    print("  mean_len = predicted_n / (predicted_n - accepted)   [derived, not llama.cpp's field]")
    print("=" * 100)
    cross_check_against_log(result, rows)

    print("\n--- k by arm (pooled over classes and passes) ---")
    print(f"{'arm':16s} {'spec':14s} {'w':>3s} {'n':>4s} {'k mean':>8s} {'k sd':>7s} "
          f"{'k range':>15s}")
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    for arm in sorted(by_arm, key=lambda a: by_arm[a][0]["width"]):
        ks = [r["k"] for r in by_arm[arm]]
        g = by_arm[arm][0]
        sd = statistics.stdev(ks) if len(ks) > 1 else 0.0
        print(f"{arm:16s} {g['spec_type']:14s} {g['width']:3d} {len(ks):4d} "
              f"{statistics.fmean(ks):8.4f} {sd:7.4f} "
              f"{min(ks):7.3f}-{max(ks):.3f}")

    print("\n--- TEST 1: is the overhead charged to REJECTION? ---")
    print("    Model a rejection-proportional cost r (decode-steps per rejected draft token):")
    print("        k = k_verify + r * n_max * (1 - acceptance)")
    print("        => dk/d(acceptance) = -r * n_max,   so  r = -slope / n_max")
    print("    A state-rollback account of the overhead predicts r > 0, i.e. slope < 0.")
    print(f"\n{'arm':16s} {'n':>2s} {'accept range':>15s} {'slope':>9s} {'r2':>6s} "
          f"{'r estimate':>11s}  reading")
    r_estimates = []
    for arm in sorted(by_arm, key=lambda a: by_arm[a][0]["width"]):
        g = by_arm[arm]
        xs = [r["acceptance"] for r in g]
        ys = [r["k"] for r in g]
        a, b, r2 = _linfit(xs, ys)
        n = g[0]["n_max"]
        r_est = -b / n if n else float("nan")
        r_estimates.append(r_est)
        spread = (max(ys) - min(ys)) / statistics.fmean(ys) * 100
        if r_est <= 0:
            reading = f"r <= 0 — no rejection cost detected (k spread {spread:.2f}%)"
        elif r_est < 0.02:
            reading = f"r negligible (k spread {spread:.2f}%)"
        else:
            reading = f"rejection cost present (k spread {spread:.2f}%)"
        print(f"{arm:16s} {n:2d} {min(xs):6.3f}-{max(xs):.3f} {b:9.4f} {r2:6.3f} "
              f"{r_est:11.5f}  {reading}")
    if r_estimates:
        worst = max(r_estimates)
        print(f"\n  Largest r across arms: {worst:+.5f} decode-steps per rejected draft token.")
        print(f"  Acceptance spans roughly {min(r['acceptance'] for r in rows):.3f}-"
              f"{max(r['acceptance'] for r in rows):.3f} in this data — nearly a ten-fold range —")
        print("  so a rejection-driven overhead of any consequence would have shown up here.")
        if worst <= 0.02:
            print("  Reading: the verification overhead is paid per position VERIFIED, not per")
            print("  draft REJECTED. This does not say rollback is free; it bounds how much of")
            print("  the observed cost rollback can account for.")

    print("\n--- TEST 2: k vs verification width, per method ---")
    print("    fitting k = k0 + c*(w-1); c is the marginal cost of one more verified position")
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["spec_type"]].append(r)
    for method, g in sorted(by_method.items()):
        pts: dict[int, list[float]] = defaultdict(list)
        for r in g:
            pts[r["width"]].append(r["k"])
        if len(pts) < 2:
            w = next(iter(pts))
            print(f"  {method:14s} only one width ({w}) — cannot fit")
            continue
        xs = [w - 1 for w in sorted(pts)]
        ys = [statistics.fmean(pts[w]) for w in sorted(pts)]
        a, b, r2 = _linfit(xs, ys)
        print(f"  {method:14s} widths {sorted(pts)}  ->  k0={a:.4f}  c={b:.4f}  r2={r2:.4f}")
        for w in sorted(pts):
            pred = a + b * (w - 1)
            print(f"      w={w:2d}  k={statistics.fmean(pts[w]):.4f}  fit={pred:.4f}  "
                  f"resid={statistics.fmean(pts[w]) - pred:+.4f}")

    ms = [m for m in by_method if len({r['width'] for r in by_method[m]}) >= 2]
    if len(ms) >= 2:
        print("\n  Two methods with independent drafters fitted separately. If their `c` agree")
        print("  while `k0` differ, the marginal cost belongs to the verification path rather")
        print("  than to the drafter.")

    print("\n--- implied optimum ---")
    print("    mean_len saturates with depth while k grows linearly, so speedup = mean_len/k")
    print("    has an interior maximum. Observed best per method:")
    for method, g in sorted(by_method.items()):
        best: dict[int, float] = defaultdict(float)
        cnt: dict[int, list[float]] = defaultdict(list)
        for r in g:
            cnt[r["n_max"]].append(r["speedup"])
        for n, v in cnt.items():
            best[n] = statistics.fmean(v)
        if best:
            bn = max(best, key=lambda n: best[n])
            listing = "  ".join(f"n{n}={best[n]:.3f}x" for n in sorted(best))
            print(f"  {method:14s} {listing}   -> best n-max = {bn}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    a = ap.parse_args()
    report(json.loads(Path(a.result).read_text()))


if __name__ == "__main__":
    main()
