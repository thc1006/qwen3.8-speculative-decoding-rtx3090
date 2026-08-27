"""Statistics for paired arm comparisons.

Two things the predecessor repo's analysis did not do, and that the community tables this repo
is scoped against also do not do:

1. The unit of independent replication is the PROMPT, not the request. Running the same prompt
   five times gives five correlated measurements, not five independent ones. Treating them as
   independent shrinks the interval by roughly sqrt(5) and manufactures significance. Every
   interval here comes from a CLUSTER bootstrap that resamples prompts, carrying all of a
   prompt's passes together.

2. The reported effect is CLASS-STRATIFIED. Speculative decoding on this model family moves
   code and prose in opposite directions; a raw mean over prompts reports the class mixture as
   if it were an effect. The primary endpoint is the mean of per-class means, and per-class
   effects are always reported alongside it.

An interval that spans zero is reported as "no detected effect". It is never reported as a
direction.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    n_clusters: int
    # Classes that contributed no variance because they hold a single prompt. Resampling one
    # item with replacement always returns that item, so such a class makes the interval
    # narrower than the design can justify. An interval with any of these is a lower bound on
    # the true width, and callers must say so rather than print it as if it were estimated.
    singleton_classes: tuple = ()

    @property
    def spans_zero(self) -> bool:
        return self.lo <= 0.0 <= self.hi

    @property
    def width_understated(self) -> bool:
        return bool(self.singleton_classes)

    @property
    def margin_half_widths(self) -> float:
        """How far the nearer bound sits from zero, counted in half-widths.

        The percentile bootstrap is not second-order accurate and undercovers at the sample size
        this study runs. Measured here against three data-generating processes at 25 prompts,
        800 replications each: 90.9 % for a normal, 90.6 % for a uniform and 88.0 % for a
        heavy-tailed mixture, against a nominal 95 %; at 50 prompts it recovers to 92.4 %. A t
        interval on the same draws reaches 94.1 %. The error is one-sided, the intervals come out
        too narrow, so the verdicts that can move are the ones whose interval nearly touches zero
        already.

        Those three numbers had no reproducible source until 2026-08-27. `harness/coverage_sim.py`
        is that source now, and it also covers the process this study actually scores its
        divergence verdicts on, which is BINARY and whose cluster mean over three passes takes one
        of four values. At 2000 replications, where a coverage figure's own Monte Carlo standard
        error is 0.6 to 0.7 points, it puts the binary case at 90.2 % for 25 prompts and 93.3 % for
        50 -- inside the band the continuous processes occupy, so this threshold applies to a
        binary outcome with about the same force. The same run reads 91.1 %, 92.0 % and 87.5 % for
        normal, uniform and heavy-tailed at 25: normal and heavy-tailed land on the figures above
        within one standard error, and uniform is 2.3 away. An earlier 300-replication pass had the
        discrepancy on `normal` instead; the standard error is 1.4 to 2.0 points at that size, so
        which of the three disagreed was itself noise. See PREREGISTRATION.md
        Correction 30. Restoring the missing coverage is worth roughly a 1.15 to 1.25 times wider
        interval, so a margin under about 1.3 is a verdict that should not be leaned on.

        Zero when the interval already spans zero.
        """
        half = (self.hi - self.lo) / 2.0
        if half <= 0 or self.spans_zero:
            return 0.0
        return min(abs(self.lo), abs(self.hi)) / half

    @property
    def near_zero(self) -> bool:
        """True when undercoverage at this sample size could reach zero."""
        return not self.spans_zero and self.margin_half_widths < 1.3

    def __str__(self) -> str:
        return f"{self.point:+.2f} [{self.lo:+.2f}, {self.hi:+.2f}]"


def stratified_mean(per_class: dict[str, list[float]]) -> float:
    """Mean of per-class means. Classes with no data are skipped, not imputed."""
    means = [statistics.fmean(v) for v in per_class.values() if v]
    if not means:
        raise ValueError("no data in any class")
    return statistics.fmean(means)


def _stratified_from_prompts(
    prompt_values: dict[str, float], prompt_class: dict[str, str]
) -> float:
    per_class: dict[str, list[float]] = {}
    for tag, val in prompt_values.items():
        per_class.setdefault(prompt_class[tag], []).append(val)
    return stratified_mean(per_class)


def paired_cluster_bootstrap(
    baseline: dict[str, list[float]],
    arm: dict[str, list[float]],
    prompt_class: dict[str, str],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 20260824,
    relative: bool,
) -> Interval:
    """Paired cluster bootstrap over prompts.

    `relative` has no default on purpose. It switches the unit between a percentage and raw
    tok/s, both of which print as a plausible number, and a caller that omits it gets the one it
    probably did not mean without anything saying so. Every caller in this repo passes it.

    ARGUMENT ORDER IS (baseline, arm) AND IT MATTERS. The statistic is arm-minus-baseline;
    swapping the two silently inverts the sign of every effect this study reports. Callers
    should pass these by keyword.

    `baseline` and `arm` map prompt tag -> list of per-pass decode rates. Both arms must cover
    the same prompts. Resampling is done over prompts WITHIN each class, which preserves the
    stratified design and keeps every bootstrap replicate balanced.

    `relative=True` returns percentage change rather than absolute delta.

    Resampling happens within each class, so a class holding a single prompt contributes zero
    variance: drawing one item with replacement always returns that item. If every class holds
    one prompt the interval collapses onto the point estimate and reads as perfect precision
    when it actually means the design could not estimate precision at all. Such classes are
    listed in `Interval.singleton_classes` and flagged by `Interval.width_understated`.
    """
    tags = sorted(set(baseline) & set(arm))
    if not tags:
        raise ValueError("no overlapping prompts between arms")
    missing = (set(baseline) | set(arm)) - set(tags)
    if missing:
        raise ValueError(f"arms do not cover the same prompts; missing from one side: {sorted(missing)}")

    # Per-prompt paired statistic, computed once.
    def prompt_stat(tag: str) -> float:
        b = statistics.fmean(baseline[tag])
        a = statistics.fmean(arm[tag])
        if relative:
            if b == 0:
                raise ValueError(f"baseline is zero for prompt {tag}")
            return (a - b) / b * 100.0
        return a - b

    stats_by_tag = {t: prompt_stat(t) for t in tags}
    point = _stratified_from_prompts(stats_by_tag, prompt_class)

    by_class: dict[str, list[str]] = {}
    for t in tags:
        by_class.setdefault(prompt_class[t], []).append(t)

    rng = random.Random(seed)
    reps: list[float] = []
    for _ in range(n_boot):
        per_class_means: list[float] = []
        for cls_tags in by_class.values():
            k = len(cls_tags)
            drawn = [stats_by_tag[cls_tags[rng.randrange(k)]] for _ in range(k)]
            per_class_means.append(statistics.fmean(drawn))
        reps.append(statistics.fmean(per_class_means))

    reps.sort()
    lo = reps[max(0, int(math.floor((alpha / 2) * n_boot)))]
    hi = reps[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    singletons = tuple(sorted(c for c, ts in by_class.items() if len(ts) == 1))
    return Interval(point=point, lo=lo, hi=hi, n_clusters=len(tags),
                    singleton_classes=singletons)


def per_class_intervals(
    baseline: dict[str, list[float]],
    arm: dict[str, list[float]],
    prompt_class: dict[str, str],
    **kw,
) -> dict[str, Interval]:
    """Same bootstrap, computed separately within each class."""
    out: dict[str, Interval] = {}
    classes = sorted({prompt_class[t] for t in set(baseline) & set(arm)})
    for cls in classes:
        sub = {t: c for t, c in prompt_class.items() if c == cls}
        b = {t: v for t, v in baseline.items() if t in sub}
        a = {t: v for t, v in arm.items() if t in sub}
        if b and a:
            out[cls] = paired_cluster_bootstrap(b, a, sub, **kw)
    return out


def within_prompt_cv(values: dict[str, list[float]]) -> dict[str, float]:
    """Coefficient of variation across passes, per prompt. A run-quality check: a prompt whose
    passes disagree wildly signals thermal drift, background load, or a zombie server."""
    out: dict[str, float] = {}
    for tag, vals in values.items():
        if len(vals) >= 2:
            m = statistics.fmean(vals)
            if m:
                out[tag] = statistics.stdev(vals) / m * 100.0
    return out
