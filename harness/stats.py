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

    @property
    def spans_zero(self) -> bool:
        return self.lo <= 0.0 <= self.hi

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
    relative: bool = False,
) -> Interval:
    """Paired cluster bootstrap over prompts.

    ARGUMENT ORDER IS (baseline, arm) AND IT MATTERS. The statistic is arm-minus-baseline;
    swapping the two silently inverts the sign of every effect this study reports. Callers
    should pass these by keyword.

    `baseline` and `arm` map prompt tag -> list of per-pass decode rates. Both arms must cover
    the same prompts. Resampling is done over prompts WITHIN each class, which preserves the
    stratified design and keeps every bootstrap replicate balanced.

    `relative=True` returns percentage change rather than absolute delta.
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
    return Interval(point=point, lo=lo, hi=hi, n_clusters=len(tags))


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
