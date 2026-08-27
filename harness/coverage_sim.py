"""How often does this study's interval actually contain the effect it is estimating?

`stats.Interval.near_zero` rests on three measured coverage figures -- 90.9 % for a normal draw,
90.6 % uniform, 88.0 % heavy-tailed, at 25 prompts against a nominal 95 % -- and those figures
live only in docstrings. The simulation that produced them was not in the repository, so they
could be quoted but not rechecked, and they could not be extended to a process they never
covered.

They never covered the one that matters most to Corrections 22 and 26. `identical` is BINARY: a
prompt's cluster mean over three passes can only be 0, 1/3, 2/3 or 1. Whether a percentile
bootstrap covers at 95 % on a four-valued cluster mean is not answered by a calibration run on
continuous draws, and H9, H10 and H11 are all scored on exactly that statistic.

This rechecks the recorded figures and adds the binary case. At 2000 replications, where a
coverage estimate's own Monte Carlo standard error is 0.6 to 0.7 points, normal and heavy-tailed
land on the docstring values within one standard error and uniform is 2.3 away. A 300-replication
pass had put the discrepancy on `normal` at 2.0 standard errors instead; the standard error is
1.4 to 2.0 points at that size, so which of the three disagreed was itself noise. `reproduces` is
still too strong a word for a run with one process 2.3 standard errors out. The design simulated is this
study's: five classes of five prompts, three passes each, resampling prompts within class, and
the same estimator the reports use -- `stats.paired_cluster_bootstrap`.

Cost note: coverage needs an interval per replication and each interval is a full bootstrap, so
the work is replications x n_boot. The recorded figures used 800 replications; this defaults to
2000, which takes about two and a half minutes and is what the committed artifact was generated
at, and every printed figure names the counts behind it rather than inheriting authority from the
docstring it is checking.
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
from collections import defaultdict

import stats

CLASSES = ("code", "prose", "reason", "chat", "zh")


def _design(n_prompts: int) -> tuple[list[str], dict[str, str]]:
    tags = [f"p{i:02d}" for i in range(n_prompts)]
    cls = {t: CLASSES[i % len(CLASSES)] for i, t in enumerate(tags)}
    return tags, cls


def _draw(process: str, rng: random.Random, mean: float) -> float:
    if process == "normal":
        return rng.gauss(mean, 1.0)
    if process == "uniform":
        return mean + rng.uniform(-1.732, 1.732)
    if process == "heavy":
        # 90 % tight, 10 % ten times wider: a mixture, which is what a real per-prompt rate
        # distribution looks like once one prompt hits a cliff the others do not.
        return rng.gauss(mean, 10.0 if rng.random() < 0.10 else 1.0)
    if process == "binary":
        return 1.0 if rng.random() < mean else 0.0
    raise ValueError(process)


def coverage(process: str, *, n_prompts: int, passes: int, replications: int, n_boot: int,
             base_mean: float, arm_mean: float, seed: int = 20260824) -> dict:
    """Fraction of replications whose interval contains the true effect.

    The true effect is `arm_mean - base_mean` for an absolute interval. Prompt-level offsets are
    drawn once per replication and applied to BOTH arms, which is what makes the design paired:
    without that the bootstrap would be estimating a quantity the study never estimates.
    """
    tags, cls = _design(n_prompts)
    rng = random.Random(seed)
    truth = arm_mean - base_mean
    covered = 0
    usable = 0
    margins: list[float] = []
    widths: list[float] = []
    for _ in range(replications):
        base: dict[str, list[float]] = {}
        arm: dict[str, list[float]] = {}
        for t in tags:
            # A per-prompt offset shared by both arms. On the binary process it shifts the
            # probability rather than the value, because a binary outcome has nowhere to put an
            # additive offset.
            if process == "binary":
                # Deliberately narrow enough that neither arm's probability reaches 0 or 1. A
                # clamp would pull the low arm up, make the realised difference smaller than
                # `truth`, and show up as undercoverage that the estimator did not commit.
                span = min(base_mean, arm_mean, 1.0 - base_mean, 1.0 - arm_mean) * 0.75
                shift = rng.uniform(-span, span)
                bm, am = base_mean + shift, arm_mean + shift
            else:
                shift = rng.gauss(0.0, 2.0)
                bm, am = base_mean + shift, arm_mean + shift
            base[t] = [_draw(process, rng, bm) for _ in range(passes)]
            arm[t] = [_draw(process, rng, am) for _ in range(passes)]
        try:
            iv = stats.paired_cluster_bootstrap(base, arm, cls, n_boot=n_boot, relative=False)
        except ValueError:
            continue
        usable += 1
        widths.append(iv.hi - iv.lo)
        if iv.lo <= truth <= iv.hi:
            covered += 1
        if not iv.spans_zero:
            margins.append(iv.margin_half_widths)
    return {
        "process": process, "n_prompts": n_prompts, "passes": passes,
        "replications": usable, "n_boot": n_boot,
        "truth": truth,
        "coverage": covered / usable if usable else float("nan"),
        "mean_width": statistics.fmean(widths) if widths else float("nan"),
        "margins": margins,
    }


def _fmt(row: dict) -> str:
    # A coverage figure is itself an estimate from `replications` Bernoulli trials, and printing
    # it to one decimal without its own uncertainty is what let a 300-replication pass read
    # 93.7 % against a docstring's 90.9 % and look like a contradiction when it was 2.0 Monte
    # Carlo standard errors. Morris, White and Crowther (2019) give the formula; at p = 0.95 you
    # need about 211 replications for an MCSE of 1.5 points and about 1900 for 0.5. At the 2000
    # this file now defaults to it is 0.6 to 0.7, and normal comes back at 91.1 % -- on the
    # recorded figure, with uniform the one that is out. Printed beside every row so nobody has
    # to derive which differences are real.
    p, k = row["coverage"], row["replications"]
    mcse = math.sqrt(p * (1.0 - p) / k) if k and 0.0 <= p <= 1.0 else float("nan")
    return (f"  {row['process']:8s} n={row['n_prompts']:<3d} passes={row['passes']}  "
            f"coverage {p * 100:5.1f} % +- {mcse * 100:.1f} (MCSE)  "
            f"(mean width {row['mean_width']:.3f}, {k} replications x "
            f"{row['n_boot']} resamples)")


def main() -> None:
    ap = argparse.ArgumentParser()
    # 2000, not 400: `analysis/bootstrap_coverage.txt` is generated at 2000 and a default that
    # does not reproduce the committed artifact is the same defect as a stale report. It costs
    # about two and a half minutes.
    ap.add_argument("--replications", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--processes", default="normal,uniform,heavy,binary")
    ap.add_argument("--n-prompts", default="25,50")
    args = ap.parse_args()

    print("=" * 92)
    print("PERCENTILE BOOTSTRAP COVERAGE, against a nominal 95 %")
    print("=" * 92)
    print("Continuous rows check the figures stats.Interval.margin_half_widths cites. The binary")
    print("row is the one those figures never covered, and it is the process every divergence")
    print("verdict in this study is scored on.")
    print()

    binary_rows = []
    for process in args.processes.split(","):
        for n in (int(x) for x in args.n_prompts.split(",")):
            if process == "binary":
                # 52 % against 16 % is Phase Q-small's own BF16-to-Q4 contrast, the interval
                # Correction 22 leaned on.
                row = coverage(process, n_prompts=n, passes=args.passes,
                               replications=args.replications, n_boot=args.n_boot,
                               base_mean=0.16, arm_mean=0.52)
                binary_rows.append(row)
            else:
                row = coverage(process, n_prompts=n, passes=args.passes,
                               replications=args.replications, n_boot=args.n_boot,
                               base_mean=0.0, arm_mean=1.0)
            print(_fmt(row))
        print()

    for row in binary_rows:
        m = row["margins"]
        if not m:
            continue
        m = sorted(m)
        under = sum(1 for x in m if x < 1.3) / len(m)
        print(f"binary, n={row['n_prompts']}: of {len(m)} intervals that cleared zero, "
              f"{under * 100:.1f} % did so by under 1.3 half-widths.")
        print(f"                 margin quartiles: {m[len(m) // 4]:.2f} / "
              f"{m[len(m) // 2]:.2f} / {m[3 * len(m) // 4]:.2f}")
    print()
    print("A coverage BELOW 95 % means the interval is too narrow and a verdict near zero can be")
    print("wrong more often than the interval admits. A coverage at or above it means the 1.3")
    print("threshold is conservative for this process, and says so for the first time with a")
    print("number rather than by analogy to the continuous case.")


if __name__ == "__main__":
    main()
