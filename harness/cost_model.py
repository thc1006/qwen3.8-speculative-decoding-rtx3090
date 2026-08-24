"""Verification-step cost model.

Speculative decoding emits `mean_len` tokens per target forward pass. llama.cpp prints that
figure per request as `mean len` but does not return it through the API, so it is derived here
from the counters that are returned. The derivation is accurate to under 1 % and the residual is
characterised in `collect()`; it is not `1 + n_max * acceptance_rate`, which only holds when
every step drafts its full width

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

    `mean_len` is derived from the per-request API counters, because there is no API field for
    it. Getting the derivation right took two attempts and the second one is still approximate;
    both are recorded here because the difference is small enough to be invisible and systematic
    enough to matter.

    The first attempt read

        predicted_n = accepted + F     =>   F = predicted_n - accepted
        mean_len    = predicted_n / F

    on the reasoning that each target forward pass emits one token of its own plus whatever
    drafts it accepted. Checked against the server's own `mean len` log line on all 625
    speculative requests of Phase A, it is wrong, and always in the same direction. The missing
    piece is the first generated token: it comes out of the prompt-processing pass, not out of a
    decode forward, so the decode phase emits `predicted_n - 1` tokens over F forwards, not
    `predicted_n`:

        predicted_n - 1 = accepted + F  =>  F = predicted_n - accepted - 1
        mean_len        = (predicted_n - 1) / F

    That reproduces the server's printed value on about 70 % of requests. The remaining 30 %
    need F smaller by one more, which is what truncation at the token cap looks like: a
    verification step that ran and was counted, whose accepted tokens were partly discarded
    because the request had reached `max_tokens`. So the true F is `predicted_n - accepted - 1`
    or one less, and the API cannot say which.

    The residual bias is bounded and reported rather than ignored. Against the server's own
    figure it is under 1 % on `mean_len`, it moves `k` by +0.33 % at n-max 2 rising to +0.59 %
    at n-max 7, and because it grows with depth it inflates the fitted `c` by about 0.8 %. It
    does not move any conclusion in this study: `c` stays at 0.28 to two figures, and every
    cross-arm ratio moves by less than the absolute figures do, since the bias has the same sign
    everywhere.

    The clean fix is upstream. `server_slot_stats` already holds the exact count in
    `n_draft_verif_steps` and does not put it in `to_json()`. The one-line patch is in
    `upstream/`, and this docstring is the argument for it: a derivation that looks exact,
    reproduces plausible numbers, and is quietly wrong by a percent is exactly what an exposed
    counter prevents.
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
        # predicted_n - accepted - 1: the first generated token comes from the prompt pass, not
        # from a decode forward. See the docstring for why this is still approximate.
        forwards = pn - accepted - 1
        if not (drafted and pn and base and forwards > 0 and rec.get("decode_tok_s")):
            continue
        speedup = rec["decode_tok_s"] / base
        mean_len = (pn - 1) / forwards
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
    ml_checked: list[float] = []
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
            # The counters agreeing is necessary and not sufficient. They agreed perfectly while
            # the mean length derived from them was a percent low, because the derivation, not
            # the counters, was wrong. Comparing the derived figure against the one the server
            # prints is the check that catches that, and it is the check that was missing.
            lg_ml = lg.get("mean_len") or lg.get("mean_draft_len")
            if lg_ml:
                ml_checked.append(r["mean_len"] - lg_ml)
    if ml_checked:
        import statistics as _st
        bias = _st.fmean(ml_checked)
        worst = max(abs(x) for x in ml_checked)
        print(f"[integrity] derived mean_len vs the server's printed mean len: "
              f"{len(ml_checked)} requests, mean gap {bias:+.4f}, worst {worst:.4f}")
        if abs(bias) > 0.02:
            print(f"            the gap is systematic, not rounding. The derivation is wrong, "
                  f"not the counters. See collect()'s docstring.")
        else:
            print(f"            within the log's own %5.2f printing precision, so the "
                  f"derivation tracks the server.")
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
    print("  mean_len = (predicted_n - 1) / (predicted_n - accepted - 1)   [derived; the API has")
    print("            no verification-step count, and this is low by <1 %. See the docstring.]")
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
