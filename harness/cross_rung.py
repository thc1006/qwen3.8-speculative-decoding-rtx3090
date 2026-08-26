"""Phase Q -- comparing the marginal cost `c` across rungs of the quantization ladder.

Why this is a separate file. `cost_model.py` reports one result file at a time and a rung IS one
result file, so its `n_models >= 2` branch -- the one that pairs the same method on two targets --
never fires on Phase Q. The statistics it needs already exist there and are imported rather than
reimplemented: a second definition of `c` in this repo would be a second thing to keep correct.

Two numbers, two questions, and they can point opposite ways.

    `c` dimensionless   slope of k(w) = k0 + c*(w-1), where k is in serial-decode-step
                        equivalents OF THE ARM'S OWN TARGET. Answers "relative to its own
                        unspeculated decode, what does this target pay per extra verified
                        position?" This is the quantity H2' is stated in, because their
                        6.7 / 14.5 / 23.4 % are also per-extra-token costs relative to the
                        same model's own throughput.

    `c` milliseconds    the same slope multiplied by that target's decode step. Answers
                        "what does a verified position cost in wall time?"

On this ladder the two targets' decode steps differ by about 16 % (Q4 41.6 tok/s -> 24.05 ms,
Q5 35.9 tok/s -> 27.87 ms). A dimensionless `c` that falls by 8 % therefore RISES in
milliseconds by about 7 %. Neither is wrong; they are answers to different questions, and a
report that gave only one would let a reader take the wrong one for the other. Both are printed.

What this can and cannot test. PR #27342's account (H2' in PREREGISTRATION.md) puts the
per-extra-token cost at 6.7 % for BF16, 14.5 % for Q8_0 and 23.4 % for Q4_K_M, measured with
`llama-batched-bench`. `c` here is a slope fitted over widths inside the MMVQ path from
server timings. The two are not the same estimator and nothing here derives one from the other,
so no interpolation onto their scale is attempted: the check is ORDINAL -- does `c` move with
quantization in the direction their account requires, and by a magnitude of the same rough
order. Two rungs about one bit apart is a weak instrument for that; `phase_qsmall` spans four
times the bit range and is the better one. This file is the honest version of the weak test,
not a substitute for the strong one.

The pairing. Both rungs run the SAME 25 prompts, so a bootstrap replicate redraws one set of
prompts and refits both rungs on it. That removes prompt difficulty, which is shared. It does
NOT remove session drift: the rungs ran hours apart, and that is reported as a confound rather
than folded into the interval, because nothing in this design can separate it from the effect.
The pass-to-pass spread of `c` WITHIN each rung is printed next to the difference as the
yardstick for how much drift this instrument shows on its own.

Usage:
    python3 harness/cross_rung.py results/phase_q_UD-Q4_K_XL.json results/phase_q_UD-Q5_K_XL.json
"""
from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cost_model as CM  # noqa: E402
import stats as ST  # noqa: E402


def load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def rung_view(result: dict, path: str) -> dict:
    """Everything about one rung that the comparison needs, plus what it needs to refuse."""
    rows = CM.collect(result)
    spec = [r for r in rows if r["spec_type"] and r["spec_type"] != "none"]
    mmvq_max, from_record = CM.recorded_mmvq_max(result)
    widths = sorted({r["width"] for r in spec})
    on_path = [w for w in widths if w <= mmvq_max]
    env = result.get("env") or {}
    arms = result.get("arms") or {}
    # The shape of the run, not just its size. A rung that died after one pass still produces a
    # perfectly fittable `c`; what it cannot produce is any estimate of how much `c` moves
    # between sessions, which is the quantity the cross-rung difference has to clear.
    shape = collections.Counter((r["arm"], r["pass"]) for r in (result.get("records") or []))
    # Intent, written by bench.py before the run rather than inferred from what survived it.
    # Inferring the expected pass count from the passes present makes every truncated run look
    # complete, which is the whole failure this guard exists to catch.
    design = result.get("design") or {}
    # One entry per tree, each with the sha of the binaries it ran. Two rungs measured with
    # different binaries are not a quantization contrast; this repo's own rule is that only
    # within-host, same-binary deltas compare.
    bins = {t: (f.get("binaries") or {}) for t, f in (design.get("kernel_facts") or {}).items()}
    base_arms = [a for a, m in arms.items()
                 if not m.get("extra_args") and not m.get("expects_drafter")]
    return {
        "path": path,
        # Trimmed: the label is printed in fixed-width columns and a long filename silently
        # breaks every alignment below it.
        "label": Path(path).stem.replace("phase_q_", "")[:18],
        "rows": spec,
        "on_path": on_path,
        "off_path": [w for w in widths if w > mmvq_max],
        "mmvq_max": mmvq_max,
        "mmvq_from_record": from_record,
        "model": env.get("model"),
        "sha256": (env.get("model_sha256") or "")[:12],
        "captured_at": env.get("captured_at"),
        "host": env.get("host"),
        "prompts": sorted({r["prompt"] for r in spec}),
        "passes": sorted({r["pass"] for r in spec}),
        "baseline_arms": base_arms,
        "incidents": result.get("incidents") or [],
        "n_records": len(result.get("records") or []),
        "shape": shape,
        "all_arms": sorted(arms),
        "n_passes_expected": design.get("passes"),
        "n_prompts_expected": design.get("n_prompts"),
        "prompt_tags": sorted(design.get("prompt_tags") or []),
        "binaries": {t: {n: (v or {}).get("sha256_16") for n, v in b.items()}
                     for t, b in bins.items()},
        "max_tokens": design.get("max_tokens"),
        "common_args": design.get("common_args"),
        "baseline_tok_s": statistics.fmean([r["baseline_tok_s"] for r in spec]) if spec else None,
        "records_raw": result.get("records") or [],
        "settle": result.get("arm_pass_settle") or {},
        "host_load": result.get("arm_pass_host_load") or {},
        "gpu": result.get("arm_pass_gpu") or {},
    }


def per_pass_c(v: dict, on_path: list[int]) -> dict[int, float]:
    """`c` fitted from each pass alone. The spread across passes bounds within-run drift.

    A cross-rung difference smaller than this spread is not evidence about quantization whatever
    its bootstrap interval says, because the bootstrap only covers prompt sampling and every pass
    here uses all prompts. What it does NOT do is estimate the between-rung drift: these passes
    are minutes apart inside one invocation and the rungs are hours apart, so this is a lower
    bound on the nuisance, and clearing it is necessary rather than sufficient.
    """
    out = {}
    xs = [w - 1 for w in on_path]
    for p in v["passes"]:
        g = [r for r in v["rows"] if r["pass"] == p]
        by_prompt, prompt_class = CM._fit_prompts(g, on_path)
        if not prompt_class:
            continue
        fit = CM._fit_on(by_prompt, prompt_class, sorted(prompt_class), on_path, xs)
        if fit:
            out[p] = fit[1]
    return out


def divergence_by_family(v: dict) -> tuple[dict, dict, dict]:
    """{family -> {prompt -> [0/1 per pass]}}, plus prompt classes and per-family fork depths.

    `divergence` is computed by bench.py at measurement time against THAT rung's own baseline
    text (see `divergence_baseline_map` in the result), so a rung's identical-rate is already
    the right quantity; what this does is line the rungs up on the arm family, since the arm
    names carry the rung as a suffix and would otherwise never match.

    The baseline arm is dropped. It is compared against itself and is identical by
    construction, so including it would dilute every rate toward 1 by a fixed amount and make
    the two rungs look more alike than they are.
    """
    per_fam: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    forks: dict[str, list[float]] = collections.defaultdict(list)
    prompt_class: dict[str, str] = {}
    for rec in v["records_raw"]:
        d = rec.get("divergence")
        if not d:
            continue
        fam = rec["arm"].split("@")[0]
        if fam.startswith("baseline"):
            continue
        per_fam[fam][rec["prompt"]].append(1.0 if d.get("identical") else 0.0)
        prompt_class[rec["prompt"]] = rec.get("class", "?")
        if not d.get("identical") and d.get("common_prefix_frac") is not None:
            forks[fam].append(d["common_prefix_frac"])
    return per_fam, prompt_class, forks


def divergence_dose_response(a: dict, b: dict) -> None:
    """Does losslessness move with the target's quantization, and in which direction?

    llama.cpp #25618 states it as a binary: greedy speculative output diverges from vanilla on a
    quantized target, a bf16 target preserves parity. A ladder turns that into a dose-response,
    which is the whole first reason phase_q exists. It is also the weaker half of this ladder:
    two rungs about one bit apart, and no bf16 anchor at all, because bf16 for this model is
    52 GB and fits on neither card here. The bf16 leg lives in phase_qsmall.
    """
    fa, cls_a, forks_a = divergence_by_family(a)
    fb, cls_b, forks_b = divergence_by_family(b)
    # The bootstrap below strata on cls_a. That is only right if both rungs class their prompts
    # the same way; a silent disagreement would stratify one rung's draws by the other's labels.
    if cls_a != cls_b:
        differ = sorted(k for k in set(cls_a) | set(cls_b) if cls_a.get(k) != cls_b.get(k))
        print(f"\n  prompt classes disagree between the rungs on {differ[:5]}; not comparing "
              f"divergence, because the stratified draw would use one rung's labels for both.")
        return
    fams = sorted(set(fa) & set(fb))
    print("\n" + "-" * 100)
    print("DIVERGENCE DOSE-RESPONSE. Each arm against ITS OWN rung's baseline text, so the only")
    print("thing that moves between the two columns is the target's quantization. #25618 states")
    print("this as a binary (quantized diverges, bf16 does not); two rungs one bit apart make it")
    print("a two-point dose-response with no bf16 anchor -- suggestive at best, and phase_qsmall")
    print("is where the bf16 leg actually is.")
    if not fams:
        missing = sorted(set(fa) ^ set(fb))
        print(f"  No arm family is present in both rungs (unmatched: {missing}).")
        return
    widths = []
    print(f"\n  {'arm family':14s} {a['label'][:10]:>16s} {b['label'][:10]:>16s}"
          f"   {'delta pp':>9s}  {'95 % interval':>20s}")
    for fam in fams:
        na = sum(len(x) for x in fa[fam].values())
        nb = sum(len(x) for x in fb[fam].values())
        ia = sum(sum(x) for x in fa[fam].values())
        ib = sum(sum(x) for x in fb[fam].values())
        ra, rb = (100.0 * ia / na if na else 0.0), (100.0 * ib / nb if nb else 0.0)
        cell_a = f"{int(ia)}/{na} ({ra:.1f}%)"
        cell_b = f"{int(ib)}/{nb} ({rb:.1f}%)"
        try:
            # (baseline, arm) is (first rung, second rung); the statistic is second minus first.
            # relative=False because a prompt that is never identical has a rate of zero and a
            # relative statistic would divide by it.
            iv = ST.paired_cluster_bootstrap(
                baseline={k: v for k, v in fa[fam].items()},
                arm={k: v for k, v in fb[fam].items()},
                prompt_class=cls_a, relative=False)
            band = f"[{100 * iv.lo:+.1f}, {100 * iv.hi:+.1f}]"
            pt = f"{100 * iv.point:+.1f}"
            iv_width = 100 * (iv.hi - iv.lo)
        except Exception as e:
            band, pt, iv_width = f"({e.__class__.__name__})", "    -", None
        print(f"  {fam:14s} {cell_a:>16s} {cell_b:>16s}   {pt:>9s}  {band:>20s}")
        if iv_width is not None:
            widths.append(iv_width)
    if widths:
        # Say what the design can see, from the data rather than from a power calculation.
        # `identical` is one bit per (prompt, pass); 25 prompts of it buys much less resolution
        # than the same 25 prompts buy for a continuous rate, and the intervals show it.
        print(f"\n  Widest interval above spans {max(widths):.1f} percentage points. A binary")
        print("  outcome over 25 prompts resolves very little: any dose-response smaller than")
        print("  that is invisible here regardless of whether it is real. Treat a delta whose")
        print("  interval covers zero as UNMEASURED, not as absent.")
    print("\n  Median shared prefix when the output DID fork (fraction of the baseline text):")
    for fam in fams:
        ma = statistics.median(forks_a[fam]) if forks_a.get(fam) else None
        mb = statistics.median(forks_b[fam]) if forks_b.get(fam) else None
        sa = f"{ma:.3f}" if ma is not None else "-"
        sb = f"{mb:.3f}" if mb is not None else "-"
        print(f"  {fam:14s} {a['label'][:10]:>10s} {sa:>7s}      {b['label'][:10]:>10s} {sb:>7s}")
    print("  A rate that moves without the fork depth moving says divergence became more or less")
    print("  FREQUENT; both moving together says it also became earlier or later in the text.")


def identification_check(a: dict, b: dict, shared: list[int]) -> bool:
    """Is the drafter doing the same thing on both rungs? Returns True if it looks unchanged.

    This runs BEFORE the cost comparison because it is that comparison's precondition, not a
    footnote to it. `k` is the whole speculative cycle, so a difference in the fitted slope has
    at least two possible sources: what a verified position costs, and how many positions the
    drafter actually produces and gets accepted. Only the first is what this ladder is about.

    It is not settled by construction. The MTP head lives INSIDE the target gguf, so quantizing
    the target quantizes the drafter too -- a rung could plausibly draft differently, accept
    differently, and move `c` for a reason that has nothing to do with the verification cost.
    Two quantities separate them, both already on cost_model.py's rows:

        acceptance   accepted / drafted, the drafter's hit rate
        width_lo     drafted / forwards + 1, the width actually realised, which sits well below
                     the nominal n_max+1 because the server reuses a surviving draft tail
                     instead of re-drafting

    If both hold still across the rungs and `c` moves, the movement is in the cost of verifying
    a position. If either moves, `c`'s difference is a mixture and this file says so rather than
    reporting the slope as though it were clean.
    """
    print("\n" + "-" * 100)
    print("IDENTIFICATION. The MTP head is inside the target gguf, so quantizing the target also")
    print("quantizes the drafter. Before a difference in `c` can be read as a difference in what")
    print("a verified position COSTS, the drafter has to be shown to behave the same on both")
    print("rungs -- otherwise the slope is a mixture of cost and drafting behaviour.")

    def by_family(v, field):
        out = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in v["rows"]:
            if r["width"] in shared:
                out[r["arm"].split("@")[0]][r["prompt"]].append(r[field])
        return out

    cls = {r["prompt"]: r["class"] for r in a["rows"]}
    clean = True
    for field, label, tol in (("acceptance", "acceptance (accepted/drafted)", 0.02),
                              ("width_lo", "realised width (drafted/forward + 1)", 0.05)):
        fa, fb = by_family(a, field), by_family(b, field)
        fams = sorted(set(fa) & set(fb))
        print(f"\n  {label}")
        print(f"  {'arm family':14s} {a['label'][:10]:>12s} {b['label'][:10]:>12s}"
              f"   {'delta':>9s}  {'95 % interval':>20s}")
        for fam in fams:
            ma = statistics.fmean([x for v_ in fa[fam].values() for x in v_])
            mb = statistics.fmean([x for v_ in fb[fam].values() for x in v_])
            try:
                iv = ST.paired_cluster_bootstrap(baseline=dict(fa[fam]), arm=dict(fb[fam]),
                                                 prompt_class=cls, relative=False)
                band = f"[{iv.lo:+.4f}, {iv.hi:+.4f}]"
                moved = not iv.spans_zero and abs(iv.point) > tol
            except Exception as e:
                band, moved = f"({e.__class__.__name__})", False
            flag = "  <-- MOVED" if moved else ""
            clean = clean and not moved
            print(f"  {fam:14s} {ma:>12.4f} {mb:>12.4f}   {mb - ma:>+9.4f}  {band:>20s}{flag}")
        # {tol} is a convention of this file, not a threshold anyone measured. It exists so a
        # shift that is statistically resolvable but far too small to account for the slope
        # difference does not raise a flag. The delta and the interval are both printed, so a
        # reader who wants a different line can draw it.
        print(f"  (flagged when the interval excludes zero AND |delta| > {tol:g}; that cut is a"
              f" convention of this file, not a measured threshold)")

    if clean:
        print("\n  Nothing moved materially. The drafter produces and lands the same draft on both")
        print("  rungs, so a difference in `c` below is not a difference in drafting behaviour.")
    else:
        print("\n  SOMETHING MOVED. `c` below is then a mixture: part verification cost, part the")
        print("  drafter behaving differently on a differently quantized head. Do not report the")
        print("  slope difference as a per-position cost without separating these first.")
    return clean


def guards(a: dict, b: dict) -> list[str]:
    """Reasons to refuse the comparison. Returned rather than raised so all of them are seen."""
    bad = []
    for v in (a, b):
        if v["incidents"]:
            bad.append(f"{v['label']}: {len(v['incidents'])} incident(s) logged; "
                       f"this repo does not compare results it would have to disown")
        if len(v["baseline_arms"]) != 1:
            bad.append(f"{v['label']}: {len(v['baseline_arms'])} baseline arms {v['baseline_arms']}; "
                       f"`k` is divided by a baseline and which one must not be ambiguous")
        if len(v["on_path"]) < 2:
            bad.append(f"{v['label']}: only {len(v['on_path'])} width(s) inside the MMVQ limit; "
                       f"a slope needs two")
    if a["model"] == b["model"]:
        bad.append("both rungs name the same model file; there is no contrast to measure")
    if a["sha256"] and a["sha256"] == b["sha256"]:
        bad.append(f"both rungs have model sha256 {a['sha256']}...; same weights, no contrast")
    shared_p = set(a["prompts"]) & set(b["prompts"])
    for v in (a, b):
        if len(shared_p) < len(v["prompts"]):
            bad.append(f"{v['label']} has {len(v['prompts'])} prompts against "
                       f"{len(shared_p)} shared; the pairing would silently drop the rest")
    for v in (a, b):
        if not v["n_passes_expected"] or not v["n_prompts_expected"]:
            bad.append(f"{v['label']}: design block has no passes/n_prompts, so completeness "
                       f"cannot be judged against intent")
            continue
        # Every (arm, pass) present, and each holding every prompt. Comparing a finished rung
        # against a half-finished one is not a comparison of two targets; it is a comparison of
        # a three-pass mean against a one-pass mean, and the first run of this tool did exactly
        # that without complaining.
        want = {(arm, p) for arm in (v["all_arms"]) for p in range(1, v["n_passes_expected"] + 1)}
        missing = want - set(v["shape"])
        if missing:
            bad.append(f"{v['label']}: {len(missing)}/{len(want)} arm-passes missing "
                       f"(e.g. {sorted(missing)[:2]}); the rung is not finished")
        ragged = {k: n for k, n in v["shape"].items() if n != v["n_prompts_expected"]}
        if ragged:
            bad.append(f"{v['label']}: arm-passes with the wrong prompt count "
                       f"{sorted(ragged.items())[:2]}")
        if len(v["passes"]) < 2:
            bad.append(f"{v['label']}: {len(v['passes'])} pass; with one pass there is no "
                       f"within-rung estimate of session drift, and drift is the confound this "
                       f"comparison cannot otherwise bound")
    if sorted(a["passes"]) != sorted(b["passes"]):
        bad.append(f"pass counts differ ({a['passes']} vs {b['passes']}); the two `c` estimates "
                   f"would be means over different amounts of averaging")
    if a["binaries"] != b["binaries"]:
        bad.append(f"the two rungs ran different binaries ({a['binaries']} vs {b['binaries']}); "
                   f"only same-binary deltas compare")
    if a["prompt_tags"] != b["prompt_tags"]:
        bad.append("the declared prompt sets differ, so the pairing is not over the same prompts")
    for f in ("max_tokens", "common_args"):
        if a[f] != b[f]:
            bad.append(f"design.{f} differs ({a[f]!r} vs {b[f]!r}); the cycles are not comparable")
    if a["mmvq_max"] != b["mmvq_max"]:
        bad.append(f"MMVQ limit differs ({a['mmvq_max']} vs {b['mmvq_max']}), so the two fits "
                   f"are over different dispatch regimes")
    return bad


def report(a: dict, b: dict) -> int:
    W = 100
    print("=" * W)
    print("Phase Q -- marginal cost of a verified position, across the quantization ladder")
    print("=" * W)

    for v in (a, b):
        print(f"\n{v['label']}")
        print(f"  model        {v['model']}")
        print(f"  sha256       {v['sha256']}...    captured {v['captured_at']}    host {v['host']}")
        print(f"  records      {v['n_records']}   passes {v['passes']}   prompts {len(v['prompts'])}"
              f"   baseline arm {v['baseline_arms']}")
        print(f"  widths       on-path {v['on_path']}"
              + (f"  off-path {v['off_path']}" if v["off_path"] else "")
              + f"   (MMVQ limit {v['mmvq_max']}, "
              + ("read from this run's own record" if v["mmvq_from_record"] else "assumed") + ")")
        print(f"  baseline     {v['baseline_tok_s']:.2f} tok/s"
              f"  ->  one serial decode step {1000.0 / v['baseline_tok_s']:.3f} ms")

    bad = guards(a, b)
    if bad:
        print("\n" + "-" * W)
        print("REFUSED. The comparison is not made because:")
        for x in bad:
            print(f"  - {x}")
        return 1

    shared = sorted(set(a["on_path"]) & set(b["on_path"]))
    print("\n" + "-" * W)
    print(f"Both fits restricted to the shared widths {shared}. k(w) is curved -- cost_model.py's")
    print("own fit reports a 6.7x lack of fit against prompt scatter on this data -- so `c` is a")
    print("CHORD over the widths fitted, and two chords over different arcs are not estimates of")
    print("the same quantity.")

    identified = identification_check(a, b, shared)

    fits = {}
    for v in (a, b):
        ci = CM.fit_ci(v["rows"], shared)
        if ci is None:
            print(f"\n{v['label']}: fit failed on the shared widths.")
            return 1
        step_ms = 1000.0 / v["baseline_tok_s"]
        fits[v["label"]] = (ci, step_ms)
        c, k0 = ci["c"], ci["k0"]
        print(f"\n  {v['label']:12s} k0 = {k0.point:.4f} [{k0.lo:.4f}, {k0.hi:.4f}]"
              f"   c = {c.point:.4f} [{c.lo:.4f}, {c.hi:.4f}]"
              f"   ({ci['n_prompts']} prompts)")
        print(f"  {'':12s} c in wall time = {c.point * step_ms:.3f} ms per verified position"
              f"  [{c.lo * step_ms:.3f}, {c.hi * step_ms:.3f}]")

    # ---------------------------------------------------------------- the paired difference
    print("\n" + "-" * W)
    print("PAIRED DIFFERENCE. One draw of prompts, both rungs refitted on it, the difference")
    print("taken inside the replicate. Two marginal intervals cannot answer this: the rungs are")
    print("estimated on the same prompts and move together, and treating a shared component as")
    print("independent noise is the error that produced and then retracted Correction 13.")

    ga, gb = a["rows"], b["rows"]
    d = CM.delta_c_ci(ga, shared, gb, shared)
    if d is None:
        print("  The paired fit did not converge.")
        return 1
    print(f"\n  c({a['label']}) - c({b['label']}) = {d.point:+.4f}"
          f"  [{d.lo:+.4f}, {d.hi:+.4f}]"
          f"   nominal 95 %, {d.n_clusters} shared prompts")
    rel = 100.0 * d.point / fits[a["label"]][0]["c"].point
    print(f"  as a fraction of c({a['label']}): {rel:+.1f} %")

    # Shape check on the DIFFERENCE, not on each fit. The residual is mostly curvature in k(w),
    # it is shared between two arms measured on the same card, and adding it twice inflates a
    # difference it largely cancels from. Same reasoning as cost_model.py's H6b section.
    ka = {w: statistics.fmean([r["k"] for r in ga if r["width"] == w]) for w in shared}
    kb = {w: statistics.fmean([r["k"] for r in gb if r["width"] == w]) for w in shared}
    xd = [w - 1 for w in shared]
    yd = [ka[w] - kb[w] for w in shared]
    sed = CM._slope_se(xd, yd)
    shape_ok, bound = True, None
    if sed is None:
        print("\n  Two shared widths only: the difference is a line through two points and has no")
        print("  residual. The interval above is the only uncertainty available.")
    else:
        a0, b0, _ = CM._linfit(xd, yd)
        res = [y - (a0 + b0 * x) for x, y in zip(xd, yd)]
        worst = max(abs(r) for r in res)
        dof = len(xd) - 2
        tcrit = CM._t95(dof)
        t = abs(b0) / sed if sed else float("inf")
        print(f"\n  Shape check on the DIFFERENCE across widths {shared}: residuals "
              + " ".join(f"{r:+.5f}" for r in res) + f", se(slope) {sed:.5f} on {dof} dof.")
        shape_ok = t >= tcrit
        bound = None if shape_ok else tcrit * sed
        scale = statistics.fmean([abs(y) for y in yd]) or 1.0
        if shape_ok:
            how = ("to numerical precision" if worst < 1e-9 * scale else f"to within {worst:.5f}")
            print(f"  The two k(w) curves differ by a straight line {how}, so whatever curvature")
            print(f"  they carry is shared and cancels; the interval above decides.")
        else:
            print(f"  The difference is NOT itself linear across these widths, so the curvature")
            print(f"  does not cancel. Its slope is {t:.2f} se against a 95 % point of {tcrit:.2f},")
            print(f"  which bounds the comparison at +/-{bound:.4f}.")

    # ---------------------------------------------------------------- the millisecond question
    print("\n" + "-" * W)
    print("THE SAME DIFFERENCE IN WALL TIME. This is a different question, not a unit change of")
    print("the answer above: `c` is denominated in each target's own decode step, and these two")
    print("steps differ.")
    sa, sb = fits[a["label"]][1], fits[b["label"]][1]
    ca, cb = fits[a["label"]][0]["c"].point, fits[b["label"]][0]["c"].point
    print(f"  {a['label']:12s} step {sa:.3f} ms   c {ca:.4f}  ->  {ca * sa:.3f} ms")
    print(f"  {b['label']:12s} step {sb:.3f} ms   c {cb:.4f}  ->  {cb * sb:.3f} ms")
    dim_dir = "lower" if d.point < 0 else "higher"
    ms_delta = ca * sa - cb * sb
    ms_dir = "lower" if ms_delta < 0 else "higher"
    print(f"  dimensionless: {a['label']} is {dim_dir} by {abs(d.point):.4f} ({abs(rel):.1f} %)")
    print(f"  wall time    : {a['label']} is {ms_dir} by {abs(ms_delta):.3f} ms "
          f"({abs(100.0 * ms_delta / (cb * sb)):.1f} %)")
    if (d.point < 0) != (ms_delta < 0):
        print("  THE TWO DISAGREE IN SIGN. The decode steps differ by "
              f"{abs(100.0 * (sa - sb) / sb):.1f} %, which is larger than the difference in the")
        print("  dimensionless slope, so the target that pays less relative to itself still pays")
        print("  more per position in wall time. H2' is stated as a relative cost, so the")
        print("  dimensionless line is the one that bears on it; the wall-time line is what a")
        print("  deployment feels. Reporting either alone would be a half-truth.")
    else:
        print("  Both denominations agree in sign, so this comparison does not turn on the choice.")
    # No interval is put on the millisecond figure. It would need the baseline's own sampling
    # uncertainty propagated through a ratio, the baselines come from different sessions, and the
    # sign question above is answered by the point estimates and the 16 % step gap alone.
    print("  (No interval on the millisecond figures: it would need each baseline's uncertainty")
    print("   propagated through a ratio across sessions, which this design cannot pair.)")

    # ---------------------------------------------------------------- the internal yardstick
    print("\n" + "-" * W)
    print("INTERNAL YARDSTICK. `c` refitted from each pass alone, every pass using all prompts,")
    print("so this spread is NOT prompt sampling -- the bootstrap above already covers that. Each")
    print("arm-pass gets a fresh server (design.fresh_server_per_arm_per_pass), so it does bound")
    print("server startup, allocation and cache state.")
    print("  It is a LOWER BOUND on what separates the rungs, not an estimate of it. The three")
    print("  passes of a rung run minutes apart inside one invocation, on one thermal trajectory;")
    print("  the rungs here ran hours apart. Whatever drifts on the scale of hours is invisible")
    print("  to this yardstick, so clearing it is necessary and not sufficient.")
    spreads = {}
    for v in (a, b):
        pc = per_pass_c(v, shared)
        if not pc:
            continue
        vals = list(pc.values())
        print(f"\n  {v['label']:12s} " + "  ".join(f"pass{p}: {c:.4f}" for p, c in sorted(pc.items())))
        if len(vals) < 2:
            # NOT a spread of zero. One pass gives no information about between-session movement,
            # and letting it enter the max() below as 0.0 let the other rung's spread stand in for
            # both -- which is how the first run of this tool concluded "several times the
            # within-rung drift" from a rung that had no drift estimate at all.
            print(f"  {'':12s} one pass: no within-rung drift estimate available")
            continue
        rng = max(vals) - min(vals)
        spreads[v["label"]] = rng
        print(f"  {'':12s} spread {rng:.4f}   sd {statistics.stdev(vals):.4f}"
              f"   ({100.0 * rng / statistics.fmean(vals):.1f} % of the mean)")
    if len(spreads) < 2:
        print("\n  Fewer than two rungs supplied a drift estimate, so the difference above cannot")
        print("  be weighed against drift at all. That is a missing check, not a passed one.")
    else:
        worst = max(spreads.values())
        ratio = abs(d.point) / worst if worst else float("inf")
        print(f"\n  widest within-rung pass spread {worst:.4f}"
              f"   against a cross-rung difference of {abs(d.point):.4f}"
              f"   ({ratio:.1f}x)")
        # The 3x line below is a convention of this file, not a result. phase_q.py notes that
        # H2' over one bit predicts roughly three times the 2.7 % drift this repo has already
        # seen in `c` -- that is a statement about the expected effect, not a decision rule.
        # The ratio is printed so a reader can apply their own line.
        if ratio < 1.0:
            print("  The difference is SMALLER than one rung's own pass-to-pass spread. Whatever")
            print("  the bootstrap interval says, this does not separate quantization from drift.")
        elif ratio < 3.0:
            print("  The difference is the same order as within-rung drift. Treat it as suggestive")
            print("  and not as an established effect; more passes, not more prompts, would help.")
        else:
            print("  The difference is several times the within-rung drift, so drift of the size")
            print("  this instrument shows on its own does not account for it.")

    divergence_dose_response(a, b)

    # ---------------------------------------------------------------- confounds
    print("\n" + "-" * W)
    print("CONFOUNDS. Not folded into any interval above, because nothing in this design")
    print("separates them from the effect.")
    print(f"  Sessions      {a['label']} captured {a['captured_at']}")
    print(f"                {b['label']} captured {b['captured_at']}")
    print("                Prompt pairing removes prompt difficulty. It does not remove anything")
    print("                that differs between these two moments.")
    for v in (a, b):
        temps = [s.get("entry_temp_c") for s in v["settle"].values() if s.get("entry_temp_c")]
        loads = [h.get("loadavg_1m") for h in v["host_load"].values() if h.get("loadavg_1m") is not None]
        cont = sum(1 for h in v["host_load"].values() if h.get("contended"))
        clocks = [g.get("at_end", {}).get("clocks.current.graphics")
                  for g in v["gpu"].values() if g.get("at_end")]
        clocks = [c for c in clocks if c]
        print(f"  {v['label']:12s} entry temp {min(temps):.0f}-{max(temps):.0f} C" if temps else
              f"  {v['label']:12s} entry temp n/a", end="")
        print(f"   loadavg {min(loads):.2f}-{max(loads):.2f}" if loads else "   loadavg n/a", end="")
        print(f"   contended arm-passes {cont}", end="")
        print(f"   end SM clock {min(clocks):.0f}-{max(clocks):.0f} MHz" if clocks else "")
    print("  Scheme        both UD-* rungs are unsloth dynamic quants, so bit width is confounded")
    print("                with quantization scheme; a Q8_0 rung would break that pairing but")
    print("                needs a 48 GB card. Plot against measured bits per weight, not labels.")

    # ---------------------------------------------------------------- verdict
    print("\n" + "-" * W)
    if not identified:
        print("PRECONDITION FAILED: the drafter did not hold still across the rungs. Every")
        print("  statement below about `c` is about a mixture of verification cost and drafting")
        print("  behaviour, whatever its interval says.")
    established = shape_ok and not d.spans_zero
    drift_ok = len(spreads) >= 2 and abs(d.point) / max(spreads.values()) >= 3.0
    if established and drift_ok and not identified:
        print("VERDICT: the slopes differ by more than drift, but the drafter did NOT hold still")
        print("  across the rungs, so the difference is not attributable to verification cost")
        print("  alone. See the IDENTIFICATION section.")
    elif established and drift_ok:
        print("VERDICT: the marginal costs differ, and by more than this instrument's own")
        print("  session-to-session drift. Everything about the speculative cycle is held except")
        print("  the target's quantization, so part of what a verified position costs moves with")
        print("  it. Direction and rough size still have to be checked against H2' by hand: that")
        print("  account is a different estimator and no derivation relates the two.")
    elif established and len(spreads) < 2:
        print("VERDICT: the interval excludes zero, but with no drift estimate from both rungs")
        print("  there is nothing to weigh it against. Not established.")
    elif established:
        print("VERDICT: the interval excludes zero, but the difference is not large against")
        print("  within-rung pass drift. That is a claim about prompts, not about sessions.")
        print("  Not established.")
    elif shape_ok:
        print("VERDICT: no difference in marginal cost is established; the interval contains zero.")
    else:
        print("VERDICT: not resolved. The two k(w) curves are not parallel over these widths, so")
        print("  one number for the difference in slope is not a summary of them.")
    print("=" * W)
    return 0


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        print("error: give two result files", file=sys.stderr)
        return 2
    if len(sys.argv) > 3:
        print("error: two rungs at a time. `c` is a chord and a three-way comparison would need "
              "a shared width range across all three, which this does not check.", file=sys.stderr)
        return 2
    pa, pb = sys.argv[1], sys.argv[2]
    a, b = rung_view(load(pa), pa), rung_view(load(pb), pb)
    return report(a, b)


if __name__ == "__main__":
    sys.exit(main())
