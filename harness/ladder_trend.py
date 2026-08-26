"""A quantization ladder as a trend, not as a set of pairwise differences.

`cross_rung.py` compares two rungs and refuses three, because `c` is a chord over the widths it
was fitted on and a three-way comparison needs a shared width range that it does not check. That
is the right refusal for a pairwise tool. A four-rung ladder asks a different question -- does
`c` move WITH quantization, and how -- and the answer is a slope, not six differences.

The x axis is measured, not a label. `UD-Q5_K_XL` and `Q6_K` are names; what varies is bits per
weight, and that is file size over parameter count. Both quantities now live in the result:
`env.model_size_bytes` records the size of the file that actually ran, precisely because these
ladders delete their weights once a rung verifies.

Parameter count is derived from the BF16 rung when the ladder has one, as size/2 -- bf16 is two
bytes per weight by definition. It is an over-estimate, because a GGUF also carries metadata and
any tensor the quantizer left at higher precision, and it is the same over-estimate for every
rung, so it shifts the bits-per-weight axis without changing any slope's sign or significance.
Without a BF16 rung the absolute axis is unavailable and the trend is reported against file size
alone, which is the same line under a different scaling.

Three trends, in the order the argument needs them:

  acceptance   must be FLAT. The MTP head is inside the target file, so quantizing the target
               quantizes the drafter; if acceptance moves with bits per weight then a moving `c`
               is a mixture of verification cost and drafting behaviour, and the second trend
               below cannot be read as a cost.
  c            the marginal cost of a verified position. Reported dimensionless and in
               milliseconds, because `c` is denominated in each target's own decode step and
               those differ across rungs -- on Phase Q the two denominations disagreed in SIGN.
  identical    byte-level agreement with the non-speculative baseline. llama.cpp #25618 scopes
               its finding to quantized targets and says bf16 preserves parity, so this trend is
               the one H9 turns on, and a ladder that includes bf16 is the only place it can be
               asked.

The bootstrap is paired across rungs: one draw of prompts, every rung refitted on that draw, the
slope computed inside the replicate. Two marginal intervals cannot answer whether a slope differs
from zero when the rungs share their prompts, and treating a shared component as independent
noise is the error that produced and then retracted Correction 13.

    python3 harness/ladder_trend.py results/phase_qsmall_*.json
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


def load_rung(path: str) -> dict:
    with open(path) as fh:
        d = json.load(fh)
    rows = CM.collect(d)
    spec = [r for r in rows if r["spec_type"] and r["spec_type"] != "none"]
    mmvq_max, _ = CM.recorded_mmvq_max(d)
    widths = sorted({r["width"] for r in spec})
    env = d.get("env") or {}
    design = d.get("design") or {}
    arms = d.get("arms") or {}
    # The rung's name comes from the ARMS, not from the filename. Every arm here is
    # `<method>@<rung>`, which the matrix builds from the rung it was told to run, so it is the
    # run's own record of what it measured. Splitting the filename on "_" gave "XL" for
    # phase_q_UD-Q4_K_XL and "M" for phase_qsmall_Q4_K_M -- two rungs collapsing to one label,
    # and the guard that catches duplicates keys on hashes, not on labels.
    _named = {r["arm"].split("@", 1)[1] for r in spec if "@" in r["arm"]}
    return {
        "path": path,
        "label": (sorted(_named)[0] if len(_named) == 1 else Path(path).stem)[:14],
        "rows": spec,
        "records": d.get("records") or [],
        "on_path": [w for w in widths if w <= mmvq_max],
        "mmvq_max": mmvq_max,
        "size_bytes": env.get("model_size_bytes"),
        "model": env.get("model"),
        "sha256": (env.get("model_sha256") or "")[:12],
        "captured_at": env.get("captured_at"),
        "passes_expected": design.get("passes"),
        "n_prompts_expected": design.get("n_prompts"),
        "prompt_tags": sorted(design.get("prompt_tags") or []),
        "binaries": {t: {n: (v or {}).get("sha256_16") for n, v in (f.get("binaries") or {}).items()}
                     for t, f in (design.get("kernel_facts") or {}).items()},
        "max_tokens": design.get("max_tokens"),
        "common_args": design.get("common_args"),
        "all_arms": sorted(arms),
        "shape": collections.Counter((r["arm"], r["pass"]) for r in (d.get("records") or [])),
        "passes": sorted({r["pass"] for r in spec}),
        "incidents": d.get("incidents") or [],
        "baseline_tok_s": statistics.fmean([r["baseline_tok_s"] for r in spec]) if spec else None,
    }


def per_pass_c(v: dict, shared: list[int]) -> dict[int, float]:
    """`c` refitted from each pass alone, every pass using all prompts.

    The paired bootstrap covers prompt sampling and nothing else -- and it covers it well, since
    pairing cancels the variation the rungs share. On this ladder the slope's half-width comes out
    an order of magnitude below any single rung's, which is that cancellation working. What none
    of it sees is that each rung is one session and the rungs ran hours apart. The pass-to-pass
    spread within a rung is the only replication of a fresh server this design has, so it is a
    LOWER bound on what separates two rungs: clearing it is necessary, not sufficient.
    """
    xs = [w - 1 for w in shared]
    out: dict[int, float] = {}
    for p_idx in sorted({r["pass"] for r in v["rows"]}):
        g = [r for r in v["rows"] if r["pass"] == p_idx]
        by_prompt, prompt_class = CM._fit_prompts(g, shared)
        if not prompt_class:
            continue
        fit = CM._fit_on(by_prompt, prompt_class, sorted(prompt_class), shared, xs)
        if fit:
            out[p_idx] = fit[1]
    return out


def guards(rungs: list[dict]) -> list[str]:
    """Everything that has to hold before a slope across these rungs means anything."""
    bad = []
    for v in rungs:
        if v["incidents"]:
            bad.append(f"{v['label']}: {len(v['incidents'])} incident(s)")
        if not v["passes_expected"] or not v["n_prompts_expected"]:
            bad.append(f"{v['label']}: no design block; completeness cannot be judged")
            continue
        want = {(a, p) for a in v["all_arms"] for p in range(1, v["passes_expected"] + 1)}
        missing = want - set(v["shape"])
        if missing:
            bad.append(f"{v['label']}: {len(missing)}/{len(want)} arm-passes missing")
        ragged = {k: n for k, n in v["shape"].items() if n != v["n_prompts_expected"]}
        if ragged:
            bad.append(f"{v['label']}: ragged arm-passes {sorted(ragged.items())[:2]}")
        if v["size_bytes"] is None:
            bad.append(f"{v['label']}: env.model_size_bytes absent, so this rung has no x value")
    ref = rungs[0]
    for v in rungs[1:]:
        if v["prompt_tags"] != ref["prompt_tags"]:
            bad.append(f"{v['label']}: different prompt set from {ref['label']}")
        if v["binaries"] != ref["binaries"]:
            bad.append(f"{v['label']}: different binaries from {ref['label']}")
        for f in ("max_tokens", "common_args", "passes_expected"):
            if v[f] != ref[f]:
                bad.append(f"{v['label']}: design.{f} differs from {ref['label']}")
    sizes = [v["size_bytes"] for v in rungs if v["size_bytes"]]
    if len(set(sizes)) != len(sizes):
        bad.append("two rungs report the same file size; they are not distinct points")
    shas = [v["sha256"] for v in rungs if v["sha256"]]
    if len(set(shas)) != len(shas):
        bad.append("two rungs report the same model hash; they are the same weights")
    return bad


def bits_per_weight(rungs: list[dict]) -> tuple[dict[str, float] | None, str]:
    """{label -> bits per weight}, and how the parameter count was obtained."""
    bf16 = [v for v in rungs if "bf16" in v["label"].lower() and v["size_bytes"]]
    if not bf16:
        return None, ("no BF16 rung, so parameter count is unavailable and the trend is against "
                      "file size alone -- the same line under a different scaling")
    n_params = bf16[0]["size_bytes"] / 2.0
    how = (f"parameter count {n_params/1e9:.3f}e9, derived as the BF16 rung's "
           f"{bf16[0]['size_bytes']:,} bytes / 2 (bf16 is two bytes per weight). An over-estimate "
           f"by whatever metadata and higher-precision tensors the file carries, identically for "
           f"every rung, so it shifts the axis and not any slope's sign")
    return {v["label"]: v["size_bytes"] * 8.0 / n_params for v in rungs if v["size_bytes"]}, how


def paired_slope(rungs: list[dict], shared: list[int], stat: str,
                 *, n_boot: int = 4000, alpha: float = 0.05, seed: int = 20260826):
    """Slope of `stat` against file size, with one prompt draw refitting every rung.

    `stat` is 'c' or 'k0'. The x axis is size in gigabytes so the slope is a readable number;
    bits per weight is a linear rescaling of it and is reported alongside where available.
    """
    import random
    prep = []
    for v in rungs:
        by_prompt, prompt_class = CM._fit_prompts(v["rows"], shared)
        if not prompt_class:
            return None
        prep.append((v, by_prompt, prompt_class))
    shared_tags = sorted(set.intersection(*[set(p[2]) for p in prep]))
    if len(shared_tags) < 2:
        return None
    classes: dict[str, list[str]] = collections.defaultdict(list)
    for tag in shared_tags:
        classes[prep[0][2][tag]].append(tag)
    xs_fit = [w - 1 for w in shared]
    idx = 0 if stat == "k0" else 1

    def slope_for(draw):
        pts = []
        for v, by_prompt, prompt_class in prep:
            fit = CM._fit_on(by_prompt, prompt_class, draw, shared, xs_fit)
            if fit is None:
                return None
            pts.append((v["size_bytes"] / 1e9, fit[idx]))
        if len({p[0] for p in pts}) < 2:
            return None
        a, b, r2 = CM._linfit([p[0] for p in pts], [p[1] for p in pts])
        return b, a, r2, pts

    point = slope_for(shared_tags)
    if point is None:
        return None
    rng = random.Random(seed)
    slopes = []
    for _ in range(n_boot):
        draw = []
        for tags in classes.values():
            draw.extend(rng.choices(tags, k=len(tags)))
        got = slope_for(draw)
        if got:
            slopes.append(got[0])
    if len(slopes) < n_boot // 2:
        return None
    slopes.sort()

    def pct(q):
        return slopes[max(0, min(len(slopes) - 1, int(round(q * (len(slopes) - 1)))))]

    return {"slope": point[0], "lo": pct(alpha / 2), "hi": pct(1 - alpha / 2),
            "intercept": point[1], "r2": point[2], "points": point[3],
            "n_prompts": len(shared_tags)}


def paired_mean_slope(rungs: list[dict], per_rung: list[dict], prompt_class: dict,
                      *, n_boot: int = 4000, alpha: float = 0.05, seed: int = 20260826):
    """Slope of a per-prompt mean against file size, one prompt draw shared by every rung.

    Used for acceptance and for share with no divergence observed through the token cap, neither of which is fitted: both are already
    a number per (prompt, pass), so the replicate redraws prompts and re-averages rather than
    refitting a line through widths.
    """
    import random
    shared_tags = sorted(set.intersection(*[set(p) for p in per_rung]))
    if len(shared_tags) < 2 or len({v["size_bytes"] for v in rungs}) < 2:
        return None
    classes: dict[str, list[str]] = collections.defaultdict(list)
    for tag in shared_tags:
        classes[prompt_class[tag]].append(tag)

    def slope_for(draw):
        pts = []
        for v, d in zip(rungs, per_rung):
            per_class: dict[str, list[float]] = collections.defaultdict(list)
            for t in draw:
                if d.get(t):
                    per_class[prompt_class[t]].append(statistics.fmean(d[t]))
            if not per_class:
                return None
            pts.append((v["size_bytes"] / 1e9,
                        statistics.fmean([statistics.fmean(xs) for xs in per_class.values()])))
        a, b, r2 = CM._linfit([p[0] for p in pts], [p[1] for p in pts])
        return b, a, r2, pts

    point = slope_for(shared_tags)
    if point is None:
        return None
    rng = random.Random(seed)
    slopes = []
    for _ in range(n_boot):
        draw = []
        for tags in classes.values():
            draw.extend(rng.choices(tags, k=len(tags)))
        got = slope_for(draw)
        if got:
            slopes.append(got[0])
    if len(slopes) < n_boot // 2:
        return None
    slopes.sort()

    def pct(q):
        return slopes[max(0, min(len(slopes) - 1, int(round(q * (len(slopes) - 1)))))]

    return {"slope": point[0], "lo": pct(alpha / 2), "hi": pct(1 - alpha / 2),
            "r2": point[2], "points": point[3], "n_prompts": len(shared_tags)}


def _by_prompt(v: dict, family: str, field: str) -> dict[str, list[float]]:
    return {p: [r[field] for r in v["rows"] if r["prompt"] == p
                and r["arm"].split("@")[0] == family and r["width"] in v["on_path"]]
            for p in {r["prompt"] for r in v["rows"]}}


def _identical_by_prompt(v: dict, family: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = collections.defaultdict(list)
    for rec in v["records"]:
        d = rec.get("divergence")
        if not d or rec["arm"].split("@")[0] != family:
            continue
        out[rec["prompt"]].append(1.0 if d.get("identical") else 0.0)
    return dict(out)


def report(rungs: list[dict]) -> int:
    W = 100
    print("=" * W)
    print("Quantization ladder as a trend: does `c` move WITH bits per weight?")
    print("=" * W)
    for v in rungs:
        gb = (v["size_bytes"] or 0) / 1e9
        print(f"\n{v['label']:10s} {v['size_bytes'] or 0:>15,d} bytes  {gb:6.2f} GB"
              f"   sha {v['sha256']}...  captured {v['captured_at']}")
        print(f"{'':10s} widths on-path {v['on_path']}   passes {v['passes']}"
              f"   baseline {v['baseline_tok_s']:.2f} tok/s"
              f"  -> step {1000.0/v['baseline_tok_s']:.3f} ms")

    bad = guards(rungs)
    if bad:
        print("\n" + "-" * W + "\nREFUSED:")
        for x in bad:
            print(f"  - {x}")
        return 1

    shared = sorted(set.intersection(*[set(v["on_path"]) for v in rungs]))
    if len(shared) < 2:
        print(f"\nThe rungs share {len(shared)} width(s); a slope needs two.")
        return 1
    bpw, how = bits_per_weight(rungs)
    print("\n" + "-" * W)
    print(f"All fits restricted to the shared widths {shared}. `c` is a chord over the widths it")
    print("was fitted on, so every rung must be fitted over the same arc for a slope across them")
    print("to be a slope in one quantity.")
    print(f"\nx axis: file size in GB. {how}.")
    if bpw:
        print("  " + "   ".join(f"{k} {v:.3f} bpw" for k, v in bpw.items()))

    families = sorted(set.intersection(
        *[{r["arm"].split("@")[0] for r in v["rows"]} for v in rungs]))
    prompt_class = {r["prompt"]: r["class"] for r in rungs[0]["rows"]}

    # ---------------------------------------------------------------- 1. acceptance must be flat
    print("\n" + "-" * W)
    print("1. ACCEPTANCE vs SIZE -- must be FLAT for the rest to mean anything.")
    print("   The MTP head is inside the target file, so quantizing the target quantizes the")
    print("   drafter. A moving acceptance makes a moving `c` a mixture of cost and behaviour.")
    identified = True
    for fam in families:
        per = [_by_prompt(v, fam, "acceptance") for v in rungs]
        s = paired_mean_slope(rungs, per, prompt_class)
        if s is None:
            print(f"  {fam:10s} (slope unavailable)")
            continue
        flat = s["lo"] <= 0 <= s["hi"]
        identified = identified and flat
        vals = "  ".join(f"{v['label']}:{p[1]:.4f}" for v, p in zip(rungs, s["points"]))
        print(f"  {fam:10s} slope {s['slope']:+.5f} /GB [{s['lo']:+.5f}, {s['hi']:+.5f}]"
              f"  {'FLAT' if flat else 'MOVES'}   {vals}")
    print("\n  " + ("Acceptance is flat across the ladder; `c` below is not drafting behaviour."
                    if identified else
                    "ACCEPTANCE MOVES. `c` below is a mixture -- do not read it as a cost."))

    # ---------------------------------------------------------------- 2. c
    print("\n" + "-" * W)
    print("2. `c` vs SIZE -- the marginal cost of a verified position.")
    s_c = paired_slope(rungs, shared, "c")
    s = s_c
    if s is None:
        print("  The paired fit did not converge.")
        return 1
    print(f"  slope {s['slope']:+.5f} per GB  [{s['lo']:+.5f}, {s['hi']:+.5f}]"
          f"   r2 {s['r2']:.4f}   {s['n_prompts']} shared prompts, paired")
    print(f"  {'rung':10s} {'GB':>7s} {'c':>9s} {'step ms':>9s} {'c in ms':>9s}")
    for v, (x, c) in zip(rungs, s["points"]):
        step = 1000.0 / v["baseline_tok_s"]
        print(f"  {v['label']:10s} {x:7.2f} {c:9.4f} {step:9.3f} {c*step:9.3f}")
    if bpw:
        span_gb = max(p[0] for p in s["points"]) - min(p[0] for p in s["points"])
        span_bpw = max(bpw.values()) - min(bpw.values())
        if span_bpw:
            print(f"\n  per bit of weight: {s['slope']*span_gb/span_bpw:+.5f} "
                  f"[{s['lo']*span_gb/span_bpw:+.5f}, {s['hi']*span_gb/span_bpw:+.5f}]")
    # The millisecond trend is a DIFFERENT question: `c` is denominated in each rung's own decode
    # step and those differ, so the two can disagree in sign. Phase Q's two rungs did.
    ms = [(v["size_bytes"]/1e9, c * 1000.0/v["baseline_tok_s"]) for v, (x, c) in zip(rungs, s["points"])]
    a_ms, b_ms, r2_ms = CM._linfit([p[0] for p in ms], [p[1] for p in ms])
    print(f"\n  in WALL TIME: slope {b_ms:+.4f} ms per GB, r2 {r2_ms:.4f}. No interval: it needs "
          f"each\n  baseline's own uncertainty propagated through a ratio across sessions.")
    if (s["slope"] < 0) != (b_ms < 0):
        print("  THE TWO DISAGREE IN SIGN. `c` is relative to each rung's own decode step and the")
        print("  steps differ across the ladder. H2' is stated as a relative cost, so the")
        print("  dimensionless slope is the one that bears on it.")

    # ---------------------------------------------------------------- 2b. the drift yardstick
    print("\n" + "-" * W)
    print("   DRIFT YARDSTICK. The interval above is prompt sampling only. Each rung is one")
    print("   session and the rungs ran hours apart, so `c` refitted per pass -- every pass using")
    print("   all prompts -- bounds what a fresh server contributes. It is a LOWER bound on what")
    print("   separates two rungs; clearing it is necessary and not sufficient.")
    spreads = {}
    for v in rungs:
        per = per_pass_c(v, shared)
        if len(per) < 2:
            print(f"     {v['label']:10s} one pass: no drift estimate available")
            continue
        vals = list(per.values())
        spreads[v["label"]] = max(vals) - min(vals)
        print(f"     {v['label']:10s} "
              + "  ".join(f"p{p}:{c:.4f}" for p, c in sorted(per.items()))
              + f"   spread {spreads[v['label']]:.4f}")
    if len(spreads) >= 2:
        worst = max(spreads.values())
        pts = [c for _, c in s_c["points"]]
        span = max(pts) - min(pts)
        half = (s_c["hi"] - s_c["lo"]) / 2
        print(f"\n     widest within-rung spread {worst:.4f}   span across the ladder {span:.4f}"
              + (f"   ({span/worst:.1f}x)" if worst else ""))
        print(f"     the slope's half-width is {half:.7f}, which is "
              f"{worst/half:.0f}x SMALLER than that drift, so the interval printed above is not")
        print("     the binding uncertainty on a cross-rung claim -- the span-over-drift ratio is.")
    else:
        print("\n     Fewer than two rungs supplied a drift estimate, so the span cannot be")
        print("     weighed against drift at all. That is a missing check, not a passed one.")

    # ---------------------------------------------------------------- 3. divergence, H9
    print("\n" + "-" * W)
    print("3. NO DIVERGENCE OBSERVED THROUGH THE TOKEN CAP, vs SIZE -- llama.cpp #25618 and H9.")
    print("   #25618 says greedy speculative output diverges on quantized targets and stays")
    print("   identical on bf16. On a ladder that is a positive slope, and the bf16 rung is the")
    print("   only place the claim's own control exists.")
    for fam in families:
        per = [_identical_by_prompt(v, fam) for v in rungs]
        if not all(per):
            continue
        s2 = paired_mean_slope(rungs, per, prompt_class)
        if s2 is None:
            continue
        rates = "  ".join(f"{v['label']}:{100*p[1]:.1f}%" for v, p in zip(rungs, s2["points"]))
        clear = not (s2["lo"] <= 0 <= s2["hi"])
        # stats.Interval.near_zero's rule, applied here too. The percentile bootstrap undercovers
        # at 25 prompts and the error is one-sided, so an interval clearing zero by under about
        # 1.3 half-widths is not a verdict to lean on. Note the calibration behind that number was
        # measured on three CONTINUOUS processes; `identical` is binary and a cluster mean over
        # three passes can only be {0, 1/3, 2/3, 1}, so the true coverage here is not established
        # even by that.
        half = (s2["hi"] - s2["lo"]) / 2.0
        margin = (min(abs(s2["lo"]), abs(s2["hi"])) / half) if (half > 0 and clear) else 0.0
        tag = "CLEAR OF ZERO" if clear else "covers zero"
        if clear and margin < 1.3:
            tag += f", but only by {margin:.2f} half-widths -- do not lean on it"
        print(f"  {fam:10s} slope {100*s2['slope']:+.2f} pp/GB "
              f"[{100*s2['lo']:+.2f}, {100*s2['hi']:+.2f}]  {tag}")
        print(f"  {'':10s} {rates}")
    print("\n  A binary outcome over 25 prompts resolves very little; Phase Q's pairwise intervals")
    print("  spanned 32 percentage points. Treat an interval covering zero as UNMEASURED.")
    print("=" * W)
    return 0


def main() -> int:
    paths = sys.argv[1:]
    if len(paths) < 3:
        print(__doc__)
        print("error: a trend needs at least three rungs; use cross_rung.py for two",
              file=sys.stderr)
        return 2
    rungs = [load_rung(p) for p in paths]
    rungs.sort(key=lambda v: v["size_bytes"] or 0)
    return report(rungs)


if __name__ == "__main__":
    sys.exit(main())
