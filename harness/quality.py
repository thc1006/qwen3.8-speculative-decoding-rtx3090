"""Output-quality guards.

Why these exist:

* vLLM issue #52475 reports MTP speculative decoding producing REPETITION COLLAPSE on a hybrid
  Gated DeltaNet Qwen3.8 target. Degenerate repeated output is fast. A benchmark that only
  records tok/s will report a collapsed arm as the winner. Every arm is therefore screened for
  degeneracy, and a degenerate request disqualifies its throughput number rather than being
  averaged in.

* Speculative decoding is routinely described as "lossless by construction". At the
  distribution level that is true. At the level of the bytes a user actually sees it is not
  guaranteed: batched verification reduces logits in a different floating-point order than
  sequential decode, which can flip a greedy argmax and fork the text. Prior work on a 3090 Ti
  states plainly that spec decode is not bit-for-bit lossless at temperature 0 on free-form
  prose. This module measures the divergence instead of asserting either position.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass


# --------------------------------------------------------------------------- degeneracy

@dataclass(frozen=True)
class DegeneracyReport:
    max_ngram_repeat: int          # highest repeat count of any 8-gram
    distinct_ratio: float          # distinct tokens / total tokens (word level)
    longest_run_chars: int         # longest run of a repeated short substring
    is_degenerate: bool
    reason: str = ""


def assess_degeneracy(
    text: str,
    *,
    n: int = 8,
    max_repeat: int = 6,
    min_distinct_ratio: float = 0.18,
    min_words: int = 40,
) -> DegeneracyReport:
    """Flag collapsed / looping output.

    Thresholds are deliberately permissive: legitimate code and structured config repeat a lot.
    The aim is to catch true collapse (the same phrase emitted dozens of times), not to police
    style. Anything flagged is reported and excluded from throughput aggregation, never silently
    dropped.
    """
    words = re.findall(r"\S+", text)
    if len(words) < min_words:
        return DegeneracyReport(0, 1.0, 0, False, "too short to assess")

    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    max_rep = max(grams.values()) if grams else 0
    distinct_ratio = len(set(words)) / len(words)

    longest_run = 0
    for size in (1, 2, 3, 4, 6, 8):
        i = 0
        while i < len(words) - size:
            chunk = words[i:i + size]
            reps = 1
            j = i + size
            while words[j:j + size] == chunk:
                reps += 1
                j += size
            if reps > 1:
                longest_run = max(longest_run, reps * sum(len(w) + 1 for w in chunk))
            i = max(i + 1, j - size)

    reasons = []
    if max_rep > max_repeat:
        reasons.append(f"{n}-gram repeated {max_rep}x (limit {max_repeat})")
    if distinct_ratio < min_distinct_ratio:
        reasons.append(f"distinct-word ratio {distinct_ratio:.3f} < {min_distinct_ratio}")
    return DegeneracyReport(
        max_ngram_repeat=max_rep,
        distinct_ratio=distinct_ratio,
        longest_run_chars=longest_run,
        is_degenerate=bool(reasons),
        reason="; ".join(reasons),
    )


# --------------------------------------------------------------------------- losslessness

@dataclass(frozen=True)
class DivergenceReport:
    identical: bool
    first_diff_char: int | None     # index of first differing character
    common_prefix_frac: float       # shared prefix / length of the shorter text
    len_ref: int
    len_arm: int
    sha_ref: str
    sha_arm: str


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def compare_outputs(reference: str, arm: str) -> DivergenceReport:
    """Character-level divergence between a baseline and a speculative arm at greedy sampling.

    Reported as a fraction of shared prefix rather than a pass/fail, because the interesting
    result is *where* the texts fork, not merely that they do.
    """
    if reference == arm:
        return DivergenceReport(True, None, 1.0, len(reference), len(arm),
                                _sha(reference), _sha(arm))
    limit = min(len(reference), len(arm))
    i = 0
    while i < limit and reference[i] == arm[i]:
        i += 1
    return DivergenceReport(
        identical=False,
        first_diff_char=i,
        common_prefix_frac=(i / limit) if limit else 0.0,
        len_ref=len(reference),
        len_arm=len(arm),
        sha_ref=_sha(reference),
        sha_arm=_sha(arm),
    )


@dataclass(frozen=True)
class RelativeDegeneracy:
    """Degeneracy of an arm judged against the BASELINE's own output for the same prompt.

    Absolute thresholds cannot separate "this model writes repetitive nginx config" from "this
    arm collapsed", because both look repetitive. The baseline answering the same prompt is the
    only fair reference: if the baseline's distinct-word ratio is 0.41 and the arm's is 0.04,
    that arm collapsed regardless of where any fixed threshold sits.
    """
    distinct_ratio_ref: float
    distinct_ratio_arm: float
    ratio_drop_frac: float          # (ref - arm) / ref
    ngram_repeat_ref: int
    ngram_repeat_arm: int
    collapsed: bool
    reason: str = ""


def assess_against_baseline(
    reference: str,
    arm: str,
    *,
    max_ratio_drop: float = 0.45,
    ngram_blowup: float = 4.0,
    min_words: int = 40,
) -> RelativeDegeneracy:
    """Flag an arm as collapsed only relative to what the baseline produced for the same prompt.

    `max_ratio_drop`: the arm loses this fraction of the baseline's lexical diversity.
    `ngram_blowup`: the arm's worst n-gram repeat is this many times the baseline's.
    """
    r = assess_degeneracy(reference, min_words=min_words)
    a = assess_degeneracy(arm, min_words=min_words)

    drop = 0.0
    if r.distinct_ratio > 0:
        drop = (r.distinct_ratio - a.distinct_ratio) / r.distinct_ratio

    reasons = []
    if drop > max_ratio_drop:
        reasons.append(
            f"lexical diversity fell {drop*100:.0f}% vs baseline "
            f"({r.distinct_ratio:.3f} -> {a.distinct_ratio:.3f})")
    if r.max_ngram_repeat > 0 and a.max_ngram_repeat > r.max_ngram_repeat * ngram_blowup:
        reasons.append(
            f"n-gram repetition {a.max_ngram_repeat}x vs baseline {r.max_ngram_repeat}x")
    elif r.max_ngram_repeat == 0 and a.max_ngram_repeat > 8:
        reasons.append(f"n-gram repeated {a.max_ngram_repeat}x where baseline had none")

    return RelativeDegeneracy(
        distinct_ratio_ref=r.distinct_ratio,
        distinct_ratio_arm=a.distinct_ratio,
        ratio_drop_frac=drop,
        ngram_repeat_ref=r.max_ngram_repeat,
        ngram_repeat_arm=a.max_ngram_repeat,
        collapsed=bool(reasons),
        reason="; ".join(reasons),
    )
