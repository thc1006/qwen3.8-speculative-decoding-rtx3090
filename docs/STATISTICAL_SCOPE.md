# Statistical scope

What the intervals in this repository are, what they are not, and what has been measured about
them rather than assumed. This was three long paragraphs inside the README's results section; it
is here because the README was doing eight jobs at once and the statistical caveats kept drifting
out of step with the Limitations section further down.

The inferential unit is the prompt, `n = 25`; the 5 passes are repeated measurements of the same
prompt, not independent samples, and 875 is not a sample size. Intervals are a paired cluster
bootstrap over prompts, on the class-stratified effect, and are **nominal** 95 %. They under-cover.

Four synthetic ABSOLUTE-difference processes come back at **87.5 % to 92.0 % at `n = 25`**. Those
were the only figures here until 2026-08-29, and they were quoted as though they applied to this
headline. They do not: `coverage_sim.py` called the bootstrap with `relative=False` while
`analyze.py` computes the Phase A effect with `relative=True`, which is a per-prompt percentage
change -- a nonlinear statistic with a random denominator, whose coverage does not follow from the
absolute case.

So it was measured rather than argued about. On a process fitted to this data's own shape -- the
per-prompt baseline CV of 0.001 that `results/phase_a.json` actually shows, a per-prompt effect of
+59.77 % with sd 21.85, and 0.15 % pass-level noise -- the **relative-ratio estimand covers at
90.2 % +- 0.7**, inside the band the absolute processes occupy. A ratio misbehaves when its
denominator can approach zero; on 25 prompts whose baselines span 41.4 to 41.7 tok/s, it cannot.
The 1.3 half-width rule below remains a sensitivity flag rather than a coverage correction.
Those come from [`analysis/bootstrap_coverage.txt`](analysis/bootstrap_coverage.txt):
four synthetic processes at 2000 replications x 2000 resamples each, every row carrying its own
Monte Carlo standard error of 0.6 to 0.7 points, which is the precision the standard formula
([Morris, White and Crowther 2019](https://onlinelibrary.wiley.com/doi/10.1002/sim.8086)) asks
about 1900 replications for.

An older set, 88.0-90.9 %, is quoted in `stats.py` from an 800-replication simulation whose code
was never in the repository, and at this replication count the reproduction lands on it: normal
**91.1 %** against 90.9 % recorded, 0.3 standard errors, and heavy-tailed **87.5 %** against
88.0 %, 0.7. Uniform is the one that does not, **92.0 %** against 90.6 %, 2.3 away. An earlier
300-replication run had put the discrepancy on `normal` instead, at 2.0 standard errors; that was
Monte Carlo noise, and settling which of the three actually disagrees is what the larger run
bought. The binary process every divergence verdict in this study is scored on comes back at
**90.2 %**, inside the band the continuous ones occupy. All of them are synthetic
data-generating processes rather than this data's own unknown distribution, so they diagnose the
estimator rather than quantify this interval; the primary Phase A effects sit far from zero under
any of the sets.

Undercoverage is why an interval that only just clears zero is not read as a result.
`stats.Interval.near_zero` counts how far the nearer bound sits from zero in half-widths and calls
anything under 1.3 too close to lean on; `analyze.py` names any verdict that sits inside that
margin.

## What no interval here carries

None of them carry uncertainty from changing host, card, build, model or prompt population. The
prompts were purposively constructed rather than sampled from deployment traffic, so the bootstrap
measures sensitivity to resampling *this* suite under its class structure and is not a
population-representative traffic interval.

Intervals across secondary arms, classes and follow-up phases are nominal and unadjusted for
multiplicity. They are not simultaneous 95 % family-wise statements. The per-class rows in
particular are exploratory: five purposively selected prompts per class, thinking mode confounded
with the reason class, language confounded with task content, and prompt order fixed in class
blocks.

## Why divergence is counted over prompts, not requests

For the byte-level divergence outcome the passes are deterministic -- every repeated pass
reproduces the same bytes -- so counting requests would treat one observation as three or five.
Divergence and right-censoring are reported over arm-prompt cells for that reason. Reporting them
over records is what made a censoring drop look larger than it was: 260 of 750 at the 400-token cap
against 9 of 375 at 1600 put the cross-tree control into one denominator and left it out of the
other.

## Limitation moved here from the README

- **Statistical scope.** The inferential unit is 25 prompts, not 875 records. The percentile
  cluster bootstrap undercovers at that size on every process tested
  (87.5-92.0 % against a nominal 95 %, four synthetic processes, 2000 replications each, Monte
  Carlo standard error 0.6 to 0.7 points), so the printed intervals should be read as
  under-covering rather than as exact 95 % statements, and none of them carry uncertainty from changing host,
  card, build, model or prompt population. The prompts were purposively constructed rather than
  sampled from deployment traffic, so the bootstrap measures sensitivity to resampling this suite
  under its class structure and is not a population-representative traffic interval. Intervals
  across secondary arms, classes and follow-up phases are nominal and unadjusted for multiplicity;
  they are not simultaneous 95 % family-wise statements.
