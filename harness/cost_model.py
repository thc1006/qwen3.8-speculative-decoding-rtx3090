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


def _corr(xs, ys):
    """Pearson r, or nan when either side has no spread. Used only to describe a confound."""
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


def _slope_se(xs, ys):
    """Textbook standard error of the slope, from the fit's own residuals across WIDTHS.

    fit_ci resamples PROMPTS. That captures prompt-to-prompt variation in `k` and captures none
    of the model's misfit across widths: it redraws which prompts contribute to each width's mean
    and never asks whether a line through those means is the right shape. When a width sits off
    the line -- Phase M's MoE arm at w = 4 misses by 0.137, 4.7 % of `k` -- the prompt bootstrap
    reports an interval built entirely from the wrong source of variation, and reports it narrow.

    Both are printed. Neither subsumes the other: the prompt interval answers "would other
    prompts have given this slope", this answers "is a line the right shape for these widths".
    """
    n = len(xs)
    if n < 3:
        return None                      # with two points the line is exact and has no residual
    a, b, _ = _linfit(xs, ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    xbar = statistics.fmean(xs)
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 0:
        return None
    return ((ss_res / (n - 2)) / sxx) ** 0.5


# Two-sided 95 % t, by residual degrees of freedom. Needed because a fit over three widths has one
# residual degree of freedom and the normal quantile is wrong by a factor of six there. Values are
# the standard table; anything past 30 uses the normal limit.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 22: 2.074, 24: 2.064,
        26: 2.056, 28: 2.048, 30: 2.042}


def _t95(dof):
    """Two-sided 95 % t, rounded to the SMALLER tabulated dof, which is the larger t.

    Rounding up the dof rounds the critical value down and makes the test liberal: at 21 dof the
    table skips to 22 and returns 2.074 against the true 2.080. Conservative is the only safe
    direction for a value that decides whether a difference is reported.
    """
    if dof >= 30:
        return 1.960
    d = max(1, int(dof))
    while d not in _T95 and d > 1:
        d -= 1
    return _T95.get(d, 12.706)


def _welch_dof(s1, n1, s2, n2):
    """Welch-Satterthwaite dof for the difference of two slopes, from residual dof n1-2, n2-2."""
    d1, d2 = max(1, n1 - 2), max(1, n2 - 2)
    v1, v2 = s1 * s1, s2 * s2
    den = v1 * v1 / d1 + v2 * v2 / d2
    return ((v1 + v2) ** 2 / den) if den > 0 else 1.0


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


def rejection_slope_ci(g: list[dict], *, n_boot: int = 4000, alpha: float = 0.05,
                       seed: int = 20260826):
    """Interval on `r` in k = k_verify + r*n_max*(1 - acceptance), resampling prompts in class.

    TEST 1 reported `r` as a bare point estimate and chose its verdict string from that point's
    sign alone. `r2` was computed, printed, and never consulted, so an arm whose fit explained
    13 % of the variance in `k` was announced as "rejection cost present" in the same words as
    one that explained 99 %. The summary line then took `max(r_estimates)` across arms -- the
    maximum of a set of noisy point estimates, which is biased upward by construction -- and used
    it to gate whether TEST 1's actual conclusion got printed at all. On the Phase M data one
    arm's noise (r2 = 0.134) suppressed that paragraph.

    An interval fixes both: a verdict comes from whether it excludes zero, and a bound comes from
    the largest upper limit rather than the largest point.
    """
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    prompt_class: dict[str, str] = {}
    for r in g:
        by_prompt[r["prompt"]].append(r)
        prompt_class[r["prompt"]] = r["class"]
    if len(prompt_class) < 3:
        return None
    # The DIVISOR is the realised draft length per forward pass, not the requested n_max. They
    # are the same on every MTP and DFlash arm in this study (within 0.6 %), and they are not the
    # same on `dflash2-n8` (6.94) or the 0.8B draft-simple arms (4.20): the server reuses a
    # surviving draft tail without re-drafting, so a cycle can cost a forward pass and generate
    # nothing. Dividing by 8 where 6.94 tokens were actually at risk understates r, which is the
    # wrong direction for something reported as an upper bound.
    widths = [r["draft_per_forward"] for r in g if r.get("draft_per_forward")]
    w = statistics.fmean(widths) if widths else g[0]["n_max"]
    if not w:
        return None
    classes: dict[str, list[str]] = defaultdict(list)
    for tag, cls in prompt_class.items():
        classes[cls].append(tag)

    def fit(tags):
        xs, ys = [], []
        for t in tags:
            for r in by_prompt[t]:
                xs.append(r["acceptance"])
                ys.append(r["k"])
        if len(set(xs)) < 2:
            return None
        return _linfit(xs, ys)

    point = fit(sorted(prompt_class))
    if point is None:
        return None
    rng = random.Random(seed)
    rs = []
    for _ in range(n_boot):
        draw = []
        for tags in classes.values():
            draw.extend(rng.choices(tags, k=len(tags)))
        got = fit(draw)
        if got:
            rs.append(-got[1] / w)
    if len(rs) < n_boot // 2:
        return None
    rs.sort()

    def pct(q):
        return rs[max(0, min(len(rs) - 1, int(round(q * (len(rs) - 1)))))]

    singles = tuple(sorted(c for c, t in classes.items() if len(t) < 2))
    return ST.Interval(-point[1] / w, pct(alpha / 2), pct(1 - alpha / 2),
                       len(prompt_class), singles), point[2]


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

    # The model is part of the key. Phase M runs a dense baseline and an MoE baseline in one
    # matrix and both declare tree "master", so keying on the tree alone made the second overwrite
    # the first on all 25 prompts: every MoE arm would have been divided by the dense baseline,
    # 41.6 against 147.8 tok/s, inflating its speedup about 3.5x and shrinking its k by the same
    # factor. That k is what H6b compares. Nothing would have errored.
    env_model = (result.get("env") or {}).get("model")

    def _model_of(meta):
        return meta.get("model") or env_model

    baselines: dict[tuple, float] = {}
    for r in recs:
        m = arms_meta.get(r["arm"], {})
        if not m.get("extra_args") and not m.get("expects_drafter"):
            baselines[(_model_of(m), m.get("tree", "?"), r["pass"], r["prompt"])] = r["decode_tok_s"]

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
        base = baselines.get((_model_of(meta), meta.get("tree", "?"), rec["pass"], rec["prompt"]))
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
            "model": _model_of(meta),
            "n_max": nmax, "width": nmax + 1,
            "acceptance": accepted / drafted,
            "drafted": drafted, "accepted": accepted,
            "forwards": forwards, "mean_len": mean_len,
            # Draft tokens the DRAFTER generated per target forward pass. Not the same as
            # `width`: the server reuses a surviving draft tail without re-drafting
            # (server-context.cpp:2893, "we have a previous (partial) draft to reuse"), so a
            # reused cycle costs a forward pass and generates nothing. On this run the MTP arms
            # sit on n_max to three decimals (0.998, 1.990, 2.987) and the 0.8B draft-simple arm
            # sits at 4.20 against an n_max of 8. TEST 1 needs that difference, because its model
            # treats n_max as a constant multiplier.
            "draft_per_forward": drafted / forwards,
            # Carried so `c` can also be reported in milliseconds. `c` is in serial-decode-step
            # equivalents of the arm's OWN target, and two targets whose decode steps differ by
            # 3.5x do not have comparable steps. Comparing their `c` directly answers "which pays
            # more relative to itself", which is not the question a per-position cost is usually
            # asked.
            "baseline_tok_s": base,
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
    print("        k = k_verify + r * w * (1 - acceptance),  w = realised draft len / forward")
    print("        => dk/d(acceptance) = -r * w,   so  r = -slope / w")
    print("    A state-rollback account of the overhead predicts r > 0, i.e. slope < 0.")
    print("    The model holds w constant within an arm. Where the server reuses a surviving")
    print("    draft tail instead of re-drafting, w moves WITH acceptance and the slope picks")
    print("    that up rather than any rejection cost - biased positive, which reads as r < 0,")
    print("    which is this test's own conclusion. Arms whose w varies are fitted and shown,")
    print("    and excluded from the bound.")
    print(f"\n{'arm':18s} {'n':>2s} {'accept range':>15s} {'w':>6s} {'w cv%':>6s} {'r2':>6s} "
          f"{'r [95 %]':>26s}  reading")
    bounds, confounded = [], []
    for arm in sorted(by_arm, key=lambda a: by_arm[a][0]["width"]):
        g = by_arm[arm]
        xs = [r["acceptance"] for r in g]
        n = g[0]["n_max"]
        dpf = [r["draft_per_forward"] for r in g if r.get("draft_per_forward")]
        mean_dpf = statistics.fmean(dpf) if dpf else float("nan")
        cv = (statistics.stdev(dpf) / mean_dpf * 100) if len(dpf) > 1 and mean_dpf else 0.0
        # What breaks the model is draft length VARYING WITH ACCEPTANCE, not draft length simply
        # differing from the requested n_max: a length that is constant at 6.94 rather than 8 only
        # rescales r, and rejection_slope_ci divides by the realised length for exactly that
        # reason. A length that moves with acceptance puts the regressor inside the response.
        # The threshold is not doing fine work: every arm in this study is either under 0.7 %
        # (every MTP and DFlash arm) or over 8 % (both draft-simple arms), a 13-fold gap.
        pinned = bool(dpf) and cv < 2.0
        got = rejection_slope_ci(g)
        if got is None:
            print(f"{arm:18s} {n:2d} {min(xs):6.3f}-{max(xs):.3f} {mean_dpf:6.2f} "
                  f"{cv:6.2f} {'-':>6s} {'not estimable':>26s}  too few prompts to resample")
            continue
        iv, r2 = got
        # What the estimate would account for, if taken at face value: r*n_max*(1 - acceptance)
        # against this arm's own k. An r that clears zero and still explains under a percent of
        # the cycle is not the overhead this test is looking for, and saying so is more useful
        # than a bare "present".
        share = (iv.point * mean_dpf * (1 - statistics.fmean(xs))
                 / statistics.fmean([r["k"] for r in g]) * 100)
        if not pinned:
            reading = (f"CONFOUNDED: draft/fwd varies {cv:.1f}% and tracks acceptance at "
                       f"r={_corr(xs, dpf):+.2f}; the regressor is inside the response")
            confounded.append(arm)
        elif iv.spans_zero:
            reading = "no rejection cost detected (interval contains zero)"
        elif iv.near_zero:
            reading = (f"clears zero by only {iv.margin_half_widths:.2f} half-widths - inside "
                       f"the known undercoverage; would be {share:+.1f}% of k")
        elif iv.point > 0:
            reading = f"rejection cost present, {share:+.1f}% of k at mean acceptance"
        else:
            reading = "k FALLS with rejection - not a rollback account"
        if pinned:
            # The share of k that the UPPER limit of r would account for. `r` alone is not
            # comparable across arms: it is per rejected draft token, so an arm with w = 1 and an
            # arm with w = 8 can carry the same total cost at eight times the r. The share is the
            # quantity a reader needs, and it is what the bound below is stated in.
            hi_share = (iv.hi * mean_dpf * (1 - statistics.fmean(xs))
                        / statistics.fmean([r["k"] for r in g]) * 100)
            bounds.append((arm, iv, hi_share))
        print(f"{arm:18s} {n:2d} {min(xs):6.3f}-{max(xs):.3f} {mean_dpf:6.2f} "
              f"{cv:6.2f} {r2:6.3f} {str(iv):>26s}  {reading}")

    if bounds:
        arm, iv, _ = max(bounds, key=lambda kv: kv[1].hi)
        sarm, _, share = max(bounds, key=lambda kv: kv[2])
        print(f"\n  Largest upper 95 % limit on r, over the {len(bounds)} arm(s) where the model "
              f"applies: {iv.hi:+.5f}")
        print(f"  decode-steps per rejected draft token ({arm}). The bound is an upper confidence")
        print("  limit, not the largest point estimate: the maximum of several noisy points is")
        print("  biased upward and is not a bound on anything.")
        print(f"  At that limit the rejection term accounts for at most {share:+.2f} % of the "
              f"cycle cost ({sarm}),")
        print("  which is the comparable figure: r is per rejected draft token, so arms at "
              "different widths")
        print("  carry the same total cost at different r.")
        print(f"  Acceptance spans roughly {min(r['acceptance'] for r in rows):.3f}-"
              f"{max(r['acceptance'] for r in rows):.3f} in this data - nearly a ten-fold range -")
        print("  so a rejection-driven overhead of any consequence would have shown up here.")
        # A verdict that clears zero by a fraction of a half-width is inside the undercoverage
        # this repo measured at n = 25 (88.0-90.9 % actual against a nominal 95 %), so it does not
        # get to overturn the conclusion on its own. stats.Interval.near_zero carries the rule.
        established = [b for b in bounds
                       if not b[1].spans_zero and b[1].point > 0 and not b[1].near_zero]
        if not established:
            print("  Reading: the verification overhead is paid per position VERIFIED, not per")
            print("  draft REJECTED. This does not say rollback is free; it bounds how much of")
            print("  the observed cost rollback can account for.")
            falls = [b for b in bounds if not b[1].spans_zero and b[1].point < 0]
            if falls:
                print("  " + ", ".join(b[0] for b in falls) + ": r is significantly NEGATIVE, "
                      "the opposite of a rollback account. k rises with acceptance at constant")
                print("  draft width, which is what a cost paid per position verified looks like "
                      "near saturation: mean_len climbs toward w+1 while the cycle cost does not.")
            marginal = [b for b in bounds
                        if not b[1].spans_zero and b[1].point > 0 and b[1].near_zero]
            if marginal:
                print("  Not established, and not treated as overturning the reading: "
                      + ", ".join(f"{b[0]} ({b[1].margin_half_widths:.2f} half-widths)"
                                  for b in marginal))
        else:
            print("  Reading: at least one arm shows an r that clears zero by more than the "
                  "known undercoverage: "
                  + ", ".join(f"{b[0]} {str(b[1])}" for b in established))
    else:
        print("\n  No arm in this file holds its draft length at n_max, so none of them can")
        print("  estimate r under this model and no bound is reported. That is a limit of the")
        print("  design, not a finding about rollback.")
    if confounded:
        print(f"  Excluded from the bound as confounded: {', '.join(confounded)}.")

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
        fits[key] = (g, on_path, a, b)
        if ci:
            print(f"      {'':14s} k0 {ci['k0'].lo:.4f} to {ci['k0'].hi:.4f}   "
                  f"c {ci['c'].lo:.4f} to {ci['c'].hi:.4f}   "
                  f"nominal 95 %, {ci['n_prompts']} prompts resampled within class")
        # A decode step is a different amount of time on each target, so `c` in step-equivalents
        # is not comparable across models even though both columns print as a plain number. The
        # millisecond figure is, and on Phase M it reverses the ordering: the MoE's `c` is the
        # larger of the two relative to its own step and the smaller in wall time, because its
        # step is roughly 3.5x shorter. Anything that reads a higher `c` as "this architecture
        # pays more per verified position" needs the second column.
        # k at zero draft depth has a known LOWER BOUND, not a known value. A cycle at w = 1 is a
        # plain decode step plus whatever the drafter costs, and the drafter cost is at least
        # zero, so k(1) >= 1.0. An earlier version of this block called it "the exact 1.0", which
        # is the baseline arm's k -- and the baseline runs no drafter, so it is a different
        # configuration from the one the line extrapolates to.
        #
        # The bound is still a free falsification test, and the only one available: no fit sees
        # w = 1, so r2 over the measured widths cannot supply it. Refitting with k(1) pinned at
        # the bound shows how much the intercept was carrying.
        xa, ya = [0] + xs, [1.0] + ys
        aa, ba, r2a = _linfit(xa, ya)
        if a >= 1.0:
            print(f"      {'':14s} k(w=1) extrapolates to {a:.4f}, above the floor of 1.0 that a "
                  f"zero-depth cycle must cost. Consistent; the excess is an upper bound on a "
                  f"fixed per-cycle cost, not a measurement of one.")
        else:
            print(f"      {'':14s} k(w=1) extrapolates to {a:.4f}, BELOW the floor of 1.0: the "
                  f"line implies a cycle cheaper than a plain decode step at zero draft depth, "
                  f"which no configuration can be.")
            print(f"      {'':14s} so k(w) is concave here and the FIRST extra position costs "
                  f"more than c. Refit with k(1) pinned at the floor: k0={aa:.4f}  c={ba:.4f}  "
                  f"r2={r2a:.4f}  ({abs(ba - b) / b * 100:.1f} % change in c)")
            if abs(ba - b) / b < 0.05 and r2a > 0.97:
                print(f"      {'':14s} under 5 % and r2 still above 0.97, so the line describes "
                      f"the measured widths well. What it does not support is reading k0 as a "
                      f"fixed overhead, on this method or any other.")
            else:
                print(f"      {'':14s} the pin moves the fit materially, so the linear form does "
                      f"not reach zero depth and neither coefficient is a mechanism.")
        se = _slope_se(xs, ys)
        if se is not None and ci:
            half = (ci["c"].hi - ci["c"].lo) / 2.0
            # LACK OF FIT, not a standard error on c. The residual across widths is dominated by
            # curvature in k(w), which is deterministic and largely shared between methods on the
            # same card -- on phase_nmax the two arms' residuals over widths 3, 5 and 7 are
            # +0.0209/-0.0418/+0.0209 and +0.0210/-0.0420/+0.0210, the same number twice. Treating
            # that as independent noise on each fit and adding it in quadrature inflates the
            # uncertainty on a DIFFERENCE by more than an order of magnitude and biases every
            # comparison toward a null. The difference is compared paired instead, below.
            #
            # Note also that with three equally spaced widths the residual vector is forced to be
            # proportional to [1, -2, 1] -- it has to be orthogonal to the constant and linear
            # directions -- so its shape carries nothing and only its magnitude is informative.
            print(f"      {'':14s} lack of fit across widths: residual se on the slope "
                  f"{se:.4f} ({len(xs)} widths, {len(xs) - 2} dof), against the prompt "
                  f"bootstrap's half-width {half:.4f}")
            if half <= 0:
                # Every prompt gave the same k, so the resampling found nothing to vary and the
                # interval collapsed onto the point. That prints as perfect precision and means
                # the design could not estimate precision at all -- the same failure
                # stats.Interval.width_understated names for single-prompt classes, reached a
                # different way. Skipping the line here would report the collapse as agreement.
                print(f"      {'':14s} the prompt bootstrap's interval has ZERO width: every "
                      f"prompt returned the same k, so it estimated no precision at all.")
            elif se > 1.5 * half:
                print(f"      {'':14s} the line is a {se / half:.1f}x poorer description of these "
                      f"widths than prompt-to-prompt scatter alone would suggest, so `c` is a "
                      f"chord over the widths fitted and not a constant marginal cost.")
        base_rates = [r["baseline_tok_s"] for r in g if r.get("baseline_tok_s")]
        if base_rates:
            step_ms = 1000.0 / statistics.fmean(base_rates)
            print(f"      {'':14s} baseline {statistics.fmean(base_rates):7.1f} tok/s "
                  f"-> one decode step {step_ms:.3f} ms; "
                  f"k0 = {a * step_ms:.3f} ms, c = {b * step_ms:.3f} ms per extra position"
                  + (f"  [{ci['c'].lo * step_ms:.3f}, {ci['c'].hi * step_ms:.3f}]" if ci else ""))
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

    for ma, (ga, oa, *_), mb, (gb, ob, *_) in cmps:
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
        # RESTRICT BOTH FITS TO THE WIDTHS THEY SHARE. k(w) is curved, so a slope over widths
        # 3 to 7 and a slope over 2 to 8 are chords of different arcs and are not estimates of
        # the same quantity. On phase_nmax matching the ranges moves the difference from -0.0424
        # to -0.0473, which is a sixth of the effect being compared.
        shared = sorted(set(oa) & set(ob))
        d = delta_c_ci(ga, shared, gb, shared) if len(shared) >= 2 else None
        if d is None:
            print(f"  The two fits share {len(shared)} width(s) and cannot be compared on a "
                  f"common range.")
        else:
            print(f"\n  c({ma}) - c({mb}) = {d.point:+.4f}  [{d.lo:+.4f}, {d.hi:+.4f}]"
                  f"   nominal 95 %, {d.n_clusters} shared prompts, paired on them,"
                  f" both fitted on the shared widths {shared}")
            # The bootstrap above redraws prompts and is the sampling uncertainty. What it
            # cannot see is whether a straight line is the right shape. Checking that on each fit
            # separately and adding the two in quadrature was wrong: the residual is mostly
            # curvature in k(w), it is shared between arms measured on the same card, and adding
            # it twice inflates a DIFFERENCE that it largely cancels from. The check belongs on
            # the difference itself.
            ka = {w: statistics.fmean([r["k"] for r in ga if r["width"] == w]) for w in shared}
            kb = {w: statistics.fmean([r["k"] for r in gb if r["width"] == w]) for w in shared}
            xd = [w - 1 for w in shared]
            yd = [ka[w] - kb[w] for w in shared]
            sed = _slope_se(xd, yd)
            if sed is None:
                print("  Two shared widths, so the difference is a line through two points and "
                      "has no residual. The interval above is the only uncertainty available.")
                shape_ok, bound = True, None
            else:
                a0, b0, _ = _linfit(xd, yd)
                res = [y - (a0 + b0 * x) for x, y in zip(xd, yd)]
                worst = max(abs(r) for r in res)
                dof = len(xd) - 2
                tcrit = _t95(dof)
                t = abs(b0) / sed if sed else float("inf")
                print(f"  Shape check on the DIFFERENCE across widths {shared}: residuals "
                      + " ".join(f"{r:+.5f}" for r in res)
                      + f", se(slope) {sed:.5f} on {dof} dof.")
                shape_ok = t >= tcrit
                bound = None if shape_ok else tcrit * sed
                if shape_ok:
                    # A residual at the floating-point floor makes t astronomically large and
                    # printing it as a number ("3602879701896390 standard errors") is noise, not
                    # evidence. Say what the number means instead.
                    scale = statistics.fmean([abs(y) for y in yd]) or 1.0
                    how = ("to numerical precision" if worst < 1e-9 * scale
                           else f"to within {worst:.5f}")
                    strength = ("exactly" if worst < 1e-9 * scale
                                else f"{t:,.0f} standard errors from zero, against a 95 % point "
                                     f"of {tcrit:.2f},")
                    print(f"  The two k(w) curves differ by a straight line {how}, so whatever "
                          f"curvature they carry is shared and cancels. The slope of that "
                          f"difference is {strength} "
                          + ("so the shape check imposes no penalty and the interval above "
                             "decides." if worst < 1e-9 * scale else "so it is not curvature."))
                else:
                    print(f"  The difference is not itself linear across these widths, so the "
                          f"curvature does NOT cancel. Its slope is {t:.2f} se against a 95 % "
                          f"point of {tcrit:.2f}, which bounds the comparison at "
                          f"+/-{bound:.4f} -- wider than the interval above, and binding.")

            if shape_ok and not d.spans_zero:
                print("  VERDICT: the marginal costs differ. `k` is the whole speculative cycle "
                      "and the two configurations share all of it except what is named at the "
                      "head of this section, so part of the marginal cost moves with that.")
            elif shape_ok:
                print("  VERDICT: no difference in marginal cost is established; the interval "
                      "contains zero.")
            elif bound is not None and abs(d.point) >= bound:
                print("  VERDICT: the difference survives even the wider shape-based bound.")
            else:
                print(f"  VERDICT: not resolved. The point estimate is {d.point:+.4f} against a "
                      f"shape-based bound of {bound:.4f}.")
                print("  The reason is not a shortage of prompts. The two k(w) curves are not "
                      "parallel over these widths, so one number for the difference in slope is "
                      "not a summary of them: whatever separates the two configurations is "
                      "itself width-dependent. More prompts would narrow the interval above and "
                      "change nothing here.")
        print("\n  Separating the components needs per-context CUDA-event timing, a replay that")
        print("  skips drafter compute, or a profiler decomposition - none of which this file has.")

    # ---------------------------------------------------------------- matched acceptance
    # The question this study keeps running into is whether a speculative path loses because its
    # drafts are bad or because running the drafter costs more than the drafts save. Acceptance
    # separates the two, and Phase M happens to contain pairs where it is matched almost exactly
    # across methods on the same target: the 0.8B draft-simple arm at n-max 4 accepts 38.7 % and
    # the built-in MTP head at n-max 5 accepts 38.6 %, and they land 76 points apart.
    #
    # Those two do NOT verify at the same width -- 3.32 columns against 5.97 -- so the pair is
    # flagged. The flag is not the end of it: the arm verifying 2.6 more columns is the faster
    # one, so the width difference works against the gap rather than explaining it, and every
    # pair prints which way its confound runs.
    #
    # Pairs are found rather than named, so this says nothing when a file holds no such pair.
    by_model_arm: dict[tuple, dict] = {}
    for r in rows:
        key = (r.get("model"), r["arm"])
        e = by_model_arm.setdefault(key, {"drafted": 0, "accepted": 0, "sp": [],
                                          "spec": r["spec_type"], "n_max": r["n_max"]})
        e["drafted"] += r["drafted"]
        e["accepted"] += r["accepted"]
        e["sp"].append(r["speedup"])
        e.setdefault("dpf", []).append(r["draft_per_forward"])
    for e in by_model_arm.values():
        e["acc"] = e["accepted"] / e["drafted"] if e["drafted"] else None
        e["speedup"] = statistics.fmean(e["sp"])
        # Effective verification width, one plus what the drafter actually proposed. `n_max` is
        # what was asked for. Two arms at the same n_max can verify at different widths -- on
        # phase_nmax DFlash2 fills 87 % of an n_max of 8 and MTP fills 99 %, so they run at 7.94
        # and 8.93 columns, one inside MMVQ_MAX_BATCH_SIZE and one past it.
        e["width_eff"] = statistics.fmean(e["dpf"]) + 1 if e.get("dpf") else None

    TOL = 0.02      # 2 acceptance points; the pairs this finds are inside 0.1
    pairs = []
    keys = sorted(by_model_arm)
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            a, b = by_model_arm[ka], by_model_arm[kb]
            if ka[0] != kb[0] or a["spec"] == b["spec"]:
                continue                      # same target, different method
            if a["acc"] is None or b["acc"] is None or abs(a["acc"] - b["acc"]) > TOL:
                continue
            pairs.append((ka, kb, a, b))
    if pairs:
        print("\n--- matched acceptance: is it draft quality, or the cost of drafting? ---")
        print("    Two methods on the SAME target whose acceptance agrees to within "
              f"{100 * TOL:.0f} points.")
        print("    Acceptance is what draft quality buys, so a pair that matches on it and")
        print("    separates on throughput separates on something else - and what the two")
        print("    methods do not share is the drafter's own forward passes.")
        print("    A pair sits at ONE width, so it speaks to the total cost of a cycle there and")
        print("    not to the slope: it cannot say whether the difference is in k0 or in c.")
        for ka, kb, a, b in pairs:
            m = str(ka[0]).rsplit("/", 1)[-1][:22] if n_models > 1 else ""
            head = f"  {m + '  ' if m else ''}"
            print(f"{head}{ka[1]} ({a['spec']}, n-max {a['n_max']}) acceptance "
                  f"{100 * a['acc']:.1f} %  ->  {a['speedup']:.3f}x")
            print(f"{' ' * len(head)}{kb[1]} ({b['spec']}, n-max {b['n_max']}) acceptance "
                  f"{100 * b['acc']:.1f} %  ->  {b['speedup']:.3f}x")
            gap = (a["speedup"] - b["speedup"]) * 100
            wa, wb = a.get("width_eff"), b.get("width_eff")
            print(f"{' ' * len(head)}acceptance differs by "
                  f"{abs(100 * (a['acc'] - b['acc'])):.1f} points, throughput by "
                  f"{abs(gap):.0f} points of baseline.")
            if wa and wb:
                # A quarter column, the same threshold width_groups.py uses before it will score
                # a pair. Beyond that the two arms are not verifying the same shape, and if they
                # straddle MMVQ_MAX_BATCH_SIZE they are not even in the same kernel.
                line = f"{' ' * len(head)}effective width {wa:.2f} against {wb:.2f} columns"
                if abs(wa - wb) <= 0.25:
                    print(line)
                else:
                    straddle = ((wa <= mmvq_max) != (wb <= mmvq_max))
                    print(line + "  -- CONFOUNDED: more than a quarter column apart"
                          + (f", and they straddle the MMVQ limit of {mmvq_max}"
                             if straddle else "")
                          + ", so this pair separates on verification shape as well as on the "
                            "drafter.")
                    # A flag with no direction is close to useless. Extra verified positions cost
                    # more, so if the arm that verifies WIDER is also the faster one, the width
                    # difference works against the gap rather than explaining it, and correcting
                    # for it would make the gap larger. Say which case this is, and price it with
                    # the fitted c when one is available for that method and model.
                    wider, faster = (ka if wa > wb else kb), (ka if a["speedup"] > b["speedup"]
                                                              else kb)
                    cs = [v[3] for kk, v in fits.items()
                          if kk[1] == ka[0] and len(v) >= 4]
                    price = ""
                    if cs:
                        price = (f" -- about {abs(wa - wb) * statistics.fmean(cs):.2f} "
                                 f"decode-steps per cycle at the fitted c")
                    if wider == faster:
                        print(f"{' ' * len(head)}The confound runs AGAINST the gap: the faster "
                              f"arm is also the one verifying {abs(wa - wb):.2f} more "
                              f"columns{price}. Correcting for it widens the gap; it cannot "
                              f"explain it.")
                    else:
                        print(f"{' ' * len(head)}The confound runs WITH the gap: the faster arm "
                              f"verifies {abs(wa - wb):.2f} fewer columns{price}. Part of the "
                              f"separation is verification shape, and this pair cannot say how "
                              f"much.")

    print("\n--- implied optimum ---")
    print("    mean_len saturates with depth while k grows linearly, so speedup = mean_len/k has")
    print("    an interior maximum in principle. Whether the measured ladder shows one is read")
    print("    off the data below rather than asserted. An earlier version of this line stated")
    print("    that the ladder falls monotonically over every width measured. That was true of")
    print("    the dense phases it was written against and false the moment Phase M ran an MTP")
    print("    ladder that peaks at n-max 2 on both targets, and nothing would have caught it.")
    print("    These are best TESTED points. Whether a peak clears noise is analyze.py's paired")
    print("    intervals, not this table.")
    for key, g in sorted(by_method.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        method = _label(key)
        cnt: dict[int, list[float]] = defaultdict(list)
        for r in g:
            cnt[r["n_max"]].append(r["speedup"])
        best = {n: statistics.fmean(v) for n, v in cnt.items()}
        if not best:
            continue
        ns = sorted(best)
        seq = [best[n] for n in ns]
        bn = max(best, key=lambda n: best[n])
        if len(ns) < 2:
            shape = "one width only"
        elif all(x >= y for x, y in zip(seq, seq[1:])):
            shape = "falls monotonically"
        elif all(x <= y for x, y in zip(seq, seq[1:])):
            shape = "rises monotonically; the peak may be beyond the widest tested"
        elif bn not in (ns[0], ns[-1]):
            shape = f"interior maximum at n-max {bn}"
        else:
            shape = "not monotone, and the best point is an endpoint"
        listing = "  ".join(f"n{n}={best[n]:.3f}x" for n in ns)
        print(f"  {method:14s} {listing}")
        print(f"  {'':14s} -> best tested n-max = {bn}   ({shape})")
        # NOT independent evidence, and it must not be read as any. It reuses each width's own
        # measured mean_len and only replaces the measured k with the fitted one, so all it asks
        # is whether smoothing k through a straight line preserves the argmax. It covers only the
        # MMVQ widths the fit was made on. A disagreement is informative; an agreement mostly says
        # the residuals are small enough not to move the peak.
        f = fits.get(key)
        if not f:
            continue
        _g, on_path, k0, c = f
        ml: dict[int, list[float]] = defaultdict(list)
        for r in _g:
            if r["width"] in on_path:
                ml[r["width"]].append(r["mean_len"])
        pred = {w - 1: statistics.fmean(ml[w]) / (k0 + c * (w - 1)) for w in on_path if ml.get(w)}
        if len(pred) < 2:
            continue
        pn = max(pred, key=lambda n: pred[n])
        agree = ("preserves it" if pn == bn
                 else f"DISAGREES with the tested argmax ({bn})")
        print(f"  {'':14s} -> smoothing k through the fit over MMVQ widths {on_path} puts the "
              f"peak at n-max {pn} ({agree}; same mean_len, so not independent evidence)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    a = ap.parse_args()
    report(json.loads(Path(a.result).read_text()))


if __name__ == "__main__":
    main()
