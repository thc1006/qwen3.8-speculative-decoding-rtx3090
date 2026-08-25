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
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import completeness as _CO  # noqa: E402
import random
import statistics
from collections import defaultdict

import speclen  # noqa: E402
import stats as ST  # noqa: E402

# ggml/src/ggml-cuda/mmvq.cu: a wider batch is dispatched to a different kernel.
MMVQ_MAX_BATCH_SIZE = 8
from pathlib import Path



def recorded_mmvq_max(d, fallback=MMVQ_MAX_BATCH_SIZE):
    """The dispatch limit this result was actually produced under, if the file records it.

    Falls back to the constant above and says so, because a run from before harness/kernel_facts.py
    existed carries no such record and silently using today's value is how an analyser starts
    describing a build it never saw.
    """
    facts = ((d.get("design") or {}).get("kernel_facts") or {})
    seen = {t.get("mmvq", {}).get("mmvq_max_batch_size") for t in facts.values()}
    seen.discard(None)
    if len(seen) == 1:
        return next(iter(seen)), True
    return fallback, False


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


def _fit_prompts(g: list[dict], on_path: list[int]):
    """{prompt -> {width -> [k]}} and {prompt -> class}, restricted to the fitted widths."""
    by_prompt: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    prompt_class: dict[str, str] = {}
    for r in g:
        if r["width"] in on_path:
            by_prompt[r["prompt"]][r["width"]].append(r["k"])
            prompt_class[r["prompt"]] = r["class"]
    return by_prompt, prompt_class


def _fit_on(by_prompt, prompt_class, tags, on_path, xs):
    """Refit k = k0 + c*(w-1) over one draw of prompts, class-stratified as everywhere else."""
    per_class: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for t in tags:
        for w, ks in by_prompt[t].items():
            per_class[prompt_class[t]][w].extend(ks)
    ys = []
    for w in on_path:
        means = [statistics.fmean(d[w]) for d in per_class.values() if d.get(w)]
        if not means:
            return None
        ys.append(statistics.fmean(means))
    return _linfit(xs, ys)


def fit_ci(g: list[dict], on_path: list[int], *, n_boot: int = 4000, alpha: float = 0.05,
           seed: int = 20260825):
    """Percentile intervals on k0 and c, resampling whole prompts within class.

    The fit runs over per-width means, so the widths are not the sample. Every prompt supplies a k
    at every width and the widths share it, which is why a replicate redraws prompts rather than
    rows. Same estimand and the same known undercoverage as the rest of this repo: 88.0-90.9 %
    actual against a nominal 95 % at n = 25.

    `c` was reported as a bare point estimate until now, which left H6b and the whole Phase Q
    quantization question with no uncertainty to compare a difference against.
    """
    by_prompt, prompt_class = _fit_prompts(g, on_path)
    if not prompt_class:
        return None
    classes: dict[str, list[str]] = defaultdict(list)
    for tag, cls in prompt_class.items():
        classes[cls].append(tag)
    xs = [w - 1 for w in on_path]

    point = _fit_on(by_prompt, prompt_class, sorted(prompt_class), on_path, xs)
    if point is None:
        return None

    rng = random.Random(seed)
    k0s, cs = [], []
    for _ in range(n_boot):
        draw = []
        for tags in classes.values():
            draw.extend(rng.choices(tags, k=len(tags)))
        got = _fit_on(by_prompt, prompt_class, draw, on_path, xs)
        if got:
            k0s.append(got[0])
            cs.append(got[1])
    if len(cs) < n_boot // 2:
        return None

    def pct(v, q):
        v = sorted(v)
        return v[max(0, min(len(v) - 1, int(round(q * (len(v) - 1)))))]

    singles = tuple(sorted(c for c, t in classes.items() if len(t) < 2))
    return {
        "k0": ST.Interval(point[0], pct(k0s, alpha / 2), pct(k0s, 1 - alpha / 2),
                          len(prompt_class), singles),
        "c": ST.Interval(point[1], pct(cs, alpha / 2), pct(cs, 1 - alpha / 2),
                         len(prompt_class), singles),
        "n_prompts": len(prompt_class),
    }


def delta_c_ci(ga: list[dict], on_a: list[int], gb: list[dict], on_b: list[int],
               *, n_boot: int = 4000, alpha: float = 0.05, seed: int = 20260825):
    """Interval on c(a) - c(b), redrawing the same prompts for both methods.

    Two separate marginal intervals cannot answer whether the coefficients differ: both are
    estimated on the same 25 prompts, so they move together. Pairing the draw removes that shared
    movement, which is the comparison H6b and Phase Q actually need.
    """
    bpa, pca = _fit_prompts(ga, on_a)
    bpb, pcb = _fit_prompts(gb, on_b)
    shared = sorted(set(pca) & set(pcb))
    if len(shared) < 2:
        return None
    classes: dict[str, list[str]] = defaultdict(list)
    for tag in shared:
        classes[pca[tag]].append(tag)
    xa, xb = [w - 1 for w in on_a], [w - 1 for w in on_b]

    fa = _fit_on(bpa, pca, shared, on_a, xa)
    fb = _fit_on(bpb, pcb, shared, on_b, xb)
    if fa is None or fb is None:
        return None

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        draw = []
        for tags in classes.values():
            draw.extend(rng.choices(tags, k=len(tags)))
        a = _fit_on(bpa, pca, draw, on_a, xa)
        b = _fit_on(bpb, pcb, draw, on_b, xb)
        if a and b:
            diffs.append(a[1] - b[1])
    if len(diffs) < n_boot // 2:
        return None
    diffs.sort()

    def pct(q):
        return diffs[max(0, min(len(diffs) - 1, int(round(q * (len(diffs) - 1)))))]

    singles = tuple(sorted(c for c, t in classes.items() if len(t) < 2))
    return ST.Interval(fa[1] - fb[1], pct(alpha / 2), pct(1 - alpha / 2), len(shared), singles)


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
        # Derived in speclen.py, which is also where the docstring above now lives: this file
        # had a copy that never consulted draft_n_verif_steps, so it would have parted company
        # with analyze.py the moment llama.cpp #27676 landed and that counter began arriving.
        forwards = speclen.forwards(rec)
        mean_len = speclen.mean_len(rec)
        if not (drafted and pn and base and forwards and mean_len and rec.get("decode_tok_s")):
            continue
        speedup = rec["decode_tok_s"] / base
        rows.append({
            "arm": rec["arm"], "pass": rec["pass"], "prompt": rec["prompt"],
            "class": rec["class"], "spec_type": _spec_type(meta),
            # Phase M runs two targets in one matrix, so the method alone no longer identifies a
            # fit. Without this, k values from two different models pool into one line and the
            # slope that comes out is not either model's.
            "model": meta.get("model") or (result.get("env") or {}).get("model"),
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
    skipped: list[tuple] = []
    ml_checked: list[float] = []
    by_ap: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        by_ap[(r["arm"], r["pass"])].append(r)
    for (arm, pas), g in by_ap.items():
        log_rows = acc_by.get(f"pass{pas:02d}_{arm}")
        # The log carries one extra line per arm-pass: the drafter-evidence request that runs
        # before the measured ones. Requiring exactly one extra is what keeps the zip aligned.
        # Skipping quietly when it is not would turn this check off without saying so, which is
        # the failure mode it exists to catch.
        if not log_rows:
            continue
        if len(log_rows) != len(g) + 1:
            skipped.append((arm, pas, len(log_rows), len(g) + 1))
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
    if skipped:
        print(f"\n[integrity] the log cross-check was SKIPPED for {len(skipped)} arm-pass(es) whose "
              f"line count did not match:")
        for arm, pas, got, want in skipped[:5]:
            print(f"              pass{pas:02d}_{arm}: {got} log lines against {want} expected")
        print(f"            That check is off for those, not passing for them.")
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
            reading = f"r <= 0 - no rejection cost detected (k spread {spread:.2f}%)"
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
              f"{max(r['acceptance'] for r in rows):.3f} in this data - nearly a ten-fold range -")
        print("  so a rejection-driven overhead of any consequence would have shown up here.")
        if worst <= 0.02:
            print("  Reading: the verification overhead is paid per position VERIFIED, not per")
            print("  draft REJECTED. This does not say rollback is free; it bounds how much of")
            print("  the observed cost rollback can account for.")

    _CO.warn_if_incomplete(result)
    mmvq_max, from_record = recorded_mmvq_max(result)
    print("\n--- TEST 2: k vs verification width, per method ---")
    print("    MMVQ dispatch limit %d, %s" % (
        mmvq_max, "read from this run's own record" if from_record
        else "from the analyser's constant: this file predates harness/kernel_facts.py"))
    print("    fitting k = k0 + c*(w-1). `c` is the marginal cost of one more verified")
    print("    position in serial-decode-step equivalents, over the whole cycle rather than")
    print("    attributed to any one component; see the note under the fits.")
    def _label(key):
        """method alone while the file holds one model, method and model once it holds two."""
        return key[0] if n_models < 2 else f"{key[0]} @ {str(key[1]).rsplit('/', 1)[-1][:22]}"

    by_method: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_method[(r["spec_type"], r.get("model"))].append(r)
    n_models = len({k[1] for k in by_method})
    fits: dict[tuple, tuple] = {}
    for key, g in sorted(by_method.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        # one model in the file: print what this always printed
        method = _label(key)
        pts: dict[int, list[float]] = defaultdict(list)
        for r in g:
            pts[r["width"]].append(r["k"])
        if len(pts) < 2:
            w = next(iter(pts))
            print(f"  {method:14s} only one width ({w}) - cannot fit")
            continue
        # A batch wider than MMVQ_MAX_BATCH_SIZE never reaches MMVQ and takes a different kernel
        # family, so a single line through both sides is a line through two regimes. On phase_nmax
        # that dragged the MTP coefficient from 0.2915 to 0.2215 and the fit quality from
        # r2 = 0.9959 to 0.8304, and the width-9 point sits 26 % below what the MMVQ line predicts.
        on_path = [w for w in sorted(pts) if w <= mmvq_max]
        off_path = [w for w in sorted(pts) if w > mmvq_max]
        if len(on_path) < 2:
            print(f"  {method:14s} fewer than two widths on the MMVQ path - cannot fit")
            continue
        xs = [w - 1 for w in on_path]
        ys = [statistics.fmean(pts[w]) for w in on_path]
        a, b, r2 = _linfit(xs, ys)
        print(f"  {method:14s} MMVQ widths {on_path}  ->  k0={a:.4f}  c={b:.4f}  r2={r2:.4f}")
        ci = fit_ci(g, on_path)
        fits[key] = (g, on_path)
        if ci:
            print(f"      {'':14s} k0 {ci['k0'].lo:.4f} to {ci['k0'].hi:.4f}   "
                  f"c {ci['c'].lo:.4f} to {ci['c'].hi:.4f}   "
                  f"nominal 95 %, {ci['n_prompts']} prompts resampled within class")
        for w in on_path:
            pred = a + b * (w - 1)
            print(f"      w={w:2d}  k={statistics.fmean(pts[w]):.4f}  fit={pred:.4f}  "
                  f"resid={statistics.fmean(pts[w]) - pred:+.4f}")
        for w in off_path:
            obs = statistics.fmean(pts[w])
            pred = a + b * (w - 1)
            rel = 100.0 * (obs - pred) / pred if pred else float("nan")
            print(f"      w={w:2d}  k={obs:.4f}  OFF THE MMVQ PATH (> {MMVQ_MAX_BATCH_SIZE}); the "
                  f"MMVQ line would predict {pred:.4f}, so it sits {rel:+.1f} %")
            print(f"            excluded from the fit: it is a different kernel, not a residual")

    # Same-trajectory check. Once a speculative arm diverges from its baseline it is no longer
    # decoding the same tokens, so acceptance and cost after the fork are measured on a different
    # history. A record that came out byte-identical shares the whole trajectory, and fitting `c`
    # on those alone is the comparison the rest of this file cannot make. If the two coefficients
    # disagree, the fit above is describing two token sequences rather than two widths.
    div = {}
    for r in result["records"]:
        v = r.get("divergence")
        if v is not None:
            div[(r["arm"], r["prompt"], r["pass"])] = not v.get("identical")
    print("\n--- the same fit on requests that never diverged from their baseline ---")
    for key, g in sorted(by_method.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        method = _label(key)
        pts_all: dict[int, list[float]] = defaultdict(list)
        pts_same: dict[int, list[float]] = defaultdict(list)
        for r in g:
            if r["width"] > mmvq_max:
                continue
            pts_all[r["width"]].append(r["k"])
            if div.get((r["arm"], r["prompt"], r["pass"])) is False:
                pts_same[r["width"]].append(r["k"])
        ws = sorted(w for w in pts_same if len(pts_same[w]) >= 2 and w in pts_all)
        if len(ws) < 3:
            print(f"  {method:14s} only {len(ws)} widths keep two or more non-diverging records; "
                  f"not fitted")
            continue
        _, c_all, _ = _linfit([w - 1 for w in ws], [statistics.fmean(pts_all[w]) for w in ws])
        _, c_same, _ = _linfit([w - 1 for w in ws], [statistics.fmean(pts_same[w]) for w in ws])
        n_same = sum(len(pts_same[w]) for w in ws)
        n_all = sum(len(pts_all[w]) for w in ws)
        gap = 100.0 * (c_same - c_all) / c_all if c_all else float("nan")
        flag = "" if abs(gap) < 5 else "   <-- the fit above is describing trajectories, not widths"
        print(f"  {method:14s} c on all {n_all:4d} = {c_all:.4f}   c on the {n_same:3d} that "
              f"never diverged = {c_same:.4f}   {gap:+.1f} %{flag}")

    # This used to assert that the two coefficients agree, whatever they were. On Phase A they
    # did, 0.2829 against 0.2784, and the shared-slope reading followed. On the completed ladder
    # they are 0.2904 against 0.2481, so the reading has to be decided by the interval instead.
    # With one model this is the two methods. With two models the pairing that answers H6b is the
    # same method on each, so the comparison is chosen rather than assumed to be whatever is left.
    if n_models >= 2:
        by_meth: dict[str, list] = defaultdict(list)
        for (meth, mdl), v in fits.items():
            by_meth[meth].append((mdl, v))
        cmps = [(f"{m} @ {str(a[0]).rsplit('/', 1)[-1][:22]}", a[1],
                 f"{m} @ {str(b[0]).rsplit('/', 1)[-1][:22]}", b[1])
                for m, v in sorted(by_meth.items()) if len(v) == 2
                for a, b in [sorted(v, key=lambda x: str(x[0]))]]
    elif len(fits) == 2:
        (ka, va), (kb, vb) = sorted(fits.items())
        cmps = [(_label(ka), va, _label(kb), vb)]
    else:
        cmps = []

    for ma, (ga, oa), mb, (gb, ob) in cmps:
        d = delta_c_ci(ga, oa, gb, ob)
        same_method = ma.split(" @ ")[0] == mb.split(" @ ")[0]
        if same_method:
            print("\n  One method on two targets, fitted separately. Everything about the cycle is")
            print("  held except the model, so a difference in the slope is a difference in what a")
            print("  verified position costs on the two architectures. That is H6b.")
        else:
            print("\n  Two methods with independent drafters, fitted separately. What `k` measures is")
            print("  the whole speculative cycle: target verification, the drafter's own forwards,")
            print("  sampling, launch and synchronisation, output extraction and any per-step state")
            print("  management. The two methods share every one of those except the drafter.")
        if d is None:
            print("  The two fits do not share enough prompts to be compared.")
        else:
            print(f"\n  c({ma}) - c({mb}) = {d.point:+.4f}  [{d.lo:+.4f}, {d.hi:+.4f}]"
                  f"   nominal 95 %, {d.n_clusters} shared prompts, paired on them")
            if d.spans_zero:
                print("  The difference does not clear zero. A shared slope would put the marginal")
                print("  cost in the machinery both methods have in common, though it would still")
                print("  not say which part of it.")
            else:
                print("  The difference clears zero, so the marginal cost is NOT shared: the two")
                print("  methods pay different amounts per verified position. Whatever `c` is")
                print("  charging for, part of it moves with the drafter, and the shared-machinery")
                print("  reading that Phase A's two-point fit supported does not survive the")
                print("  completed ladder.")
        print("\n  Separating the components needs per-context CUDA-event timing, a replay that")
        print("  skips drafter compute, or a profiler decomposition - none of which this file has.")

    print("\n--- implied optimum ---")
    print("    mean_len saturates with depth while k grows linearly, so speedup = mean_len/k")
    print("    has an interior maximum in principle. Over the widths measured here it falls")
    print("    monotonically, so what follows is the best TESTED point, not a fitted optimum:")
    for key, g in sorted(by_method.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        method = _label(key)
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
