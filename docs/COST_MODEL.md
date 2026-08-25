# The verification-step cost model

Extracted from the README so the front page stays readable. Part of
[`thc1006/qwen3.8-speculative-decoding-rtx3090`](https://github.com/thc1006/qwen3.8-speculative-decoding-rtx3090).

## A cost model, not a table

llama.cpp reports enough per request to recover the cost of one speculative verification step in
units of a plain decode step, as `speedup = mean_len / k` with `k(w) = k0 + c*(w - 1)` and
`w = n_max + 1`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../analysis/plot_cost_model_dark.png">
  <img alt="Three stacked panels against verification width, from Phase A only. Top: tokens accepted per target pass rises from 2.3 to 3.3 but falls far below a dotted line showing growth in proportion to width. Middle: the cost of one target pass rises linearly, with c equal to 0.2829 for draft-mtp and 0.2784 for draft-dflash on these five arms; the completed ladder supersedes both at 0.2904 and 0.2481. Bottom: speedup, the ratio of the two, falls from 1.60 to 1.23 across the widths measured here." src="../analysis/plot_cost_model.png">
</picture>

| phase | method | widths | k0 | **c** | r^2 |
|---|---|---|---:|---:|---:|
| A | `draft-mtp` | 3, 4, 6 | 0.8937 | **0.2829** | 0.9998 |
| A | `draft-dflash` | 5, 8 | 0.7825 | **0.2784** | (2 points, so r^2 is arithmetic) |
| n-max | `draft-mtp` | 2, 3, 4, 5, 6, 7, 8 | 0.8888 | **0.2904** | 0.9958 |
| n-max | `draft-dflash` | 3, 5, 7 | 0.9443 | **0.2481** | 0.9947 |

Phase A fitted three MTP widths and two DFlash2 widths; two points make an r^2 of 1 arithmetic
rather than evidence, which is why `phase_nmax` runs the full ladder. On seven MTP widths the line
holds at r^2 = 0.9958 with five residual degrees of freedom.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../analysis/plot_dispatch_boundary_dark.png">
  <img alt="The cost of one verification step against verification width, from the n-max ladder. Widths 2 through 8 for draft-mtp and 3, 5 and 7 for draft-dflash rise on a straight line, with c equal to 0.2904 at r-squared 0.9958 and 0.2481 at 0.9947. A vertical line marks MMVQ_MAX_BATCH_SIZE at 8. Width 9 sits past it with an open marker, 26 percent below the line for draft-mtp and 7 percent below for draft-dflash." src="../analysis/plot_dispatch_boundary.png">
</picture>

The two fits stop at width 8 deliberately. `MMVQ_MAX_BATCH_SIZE` is 8, so a wider verification
batch never reaches that kernel at all: at width 9 `k` sits **26 % below** what the MMVQ line
predicts for MTP and 6.7 % below for DFlash2, and throughput jumps back from +9.1 % at width 8 to
+39.1 % at width 9. Fitting one line across the boundary dragged the MTP coefficient from 0.2904
to 0.2215 and the fit from 0.9958 to 0.8304. The same boundary shows up a third way: two unrelated
drafters that share only the verification width agree on the fork position for 25 of 25 prompts at
widths 3, 5 and 7. At width 9 they agree on only 8 of 25, and that is not the control failing: they
never verified at the same width there. `n_max` is what was asked for, and the width verified is
one plus what the drafter actually proposed. DFlash2 fills 87 % of its budget at `n_max` 8 and MTP
99 %, so they run at 7.94 and 8.93 columns, one inside `MMVQ_MAX_BATCH_SIZE` and one past it. At
widths 3, 5 and 7 the two match to 0.00 columns. `harness/width_groups.py` now computes effective
width per arm and refuses to score a pair that differ by more than a quarter column.

**`c` does not agree between the two methods.** Paired on the 25 prompts both are fitted on,
`c(draft-dflash) - c(draft-mtp)` is **-0.0424 [-0.0434, -0.0413]**, which clears zero: 0.2481
against 0.2904, a gap of 15 %. Phase A's two-point fit put them at 0.2784 and 0.2829 and that
near-agreement carried an inference, that the marginal cost sits in the machinery both methods
share. The completed ladder does not support it. `k` is the whole speculative cycle - target
verification, the drafter's own forward passes, sampling, launch and synchronisation, output
extraction and any per-step state management - and the two methods share all of it except the
drafter, so a difference this size says part of the marginal cost moves with the drafter. `c` is
therefore reported as a total marginal cost per verified position, not as a target-verification
cost, and not as a quantity common to both methods. Separating the components needs per-context
event timing, a replay that skips drafter compute, or a profiler decomposition; none of those is
in this repo.

One threat to `c` can be checked without any of them. Once an arm diverges from its baseline it is
decoding a different token sequence, so what follows is not a comparison of two widths on one
trajectory. Between a fifth and a quarter of these requests come out byte-identical, and those
share the whole trajectory. Fitting on those alone gives **0.2898 against 0.2904 for `draft-mtp`
and 0.2476 against 0.2481 for `draft-dflash`**, a gap of 0.2 % in both. Divergence does not move
the coefficient here. `cost_model.py` prints this comparison on every run rather than leaving it
as a one-off.

Accepted length shows diminishing increments over the measured widths while `k` grows roughly
linearly inside the MMVQ path, so the ratio has an interior maximum in principle. **The best
tested setting is n-max 2 for both methods on this card at this target quantisation.** For MTP the
peak sits at width 3, with a tested and slower point on each side: width 2 gives **+44.96 %
[+43.45, +46.54]** and width 4 gives **+52.36 % [+48.49, +56.55]** against width 3's **+58.84 %
[+55.90, +61.89]**.

Those intervals do not settle it. Each is an arm against its baseline, not one arm against
another, and they share the same baseline and the same 25 prompts, so they are strongly
correlated and overlap is not the relevant test in either direction. On the numbers, width 2 and
width 3 happen not to overlap while width 3 and width 4 do, between 55.90 and 56.55. A paired
contrast of the two arms has not been computed. DFlash2 is weaker still: its ladder starts at
n-max 2, so its best point has no tested point below it and is not bracketed at all, and its
1.7-point lead over n-max 4 sits inside two intervals that overlap almost completely. Both are
also settings chosen on the same data they were measured on; confirming either without that
selection needs fresh prompts or fresh passes.

<details>
<summary>Why an RTX 5090 report recommends the opposite setting, and what would have to differ</summary>

The PR thread disagrees. `lance0` reports on an RTX 5090 with a `UD-Q6_K_XL` target that n-max 7
is right for DFlash2, since the drafter's `block_size` is 8 and lower values discard tokens the
block already paid for. Here n-max 4 beats n-max 7 by a wide margin, 1.520x against 1.228x, and
the completed ladder goes shallower still: n-max 2 at 1.537x.

The model says both can be true, and says what would have to differ. For width 8 to beat width 5
on this measured acceptance curve, `c` would have to be below **0.0543**. Phase A's two DFlash2
points give 0.2784, 5.1 times too large; the completed ladder gives 0.2481 over widths 3, 5 and 7,
4.6 times too large. Either way the shallower setting wins by a wide margin. Phase R2 shows what `c` responds to: over the tested GA102 clock ranges the baseline responds to
core clock with an elasticity of 0.27 while the speculative arms sit at 0.76-0.81. That is
consistent with `c` being dominated by compute, but clock elasticity is not a bottleneck
measurement - it also moves with the voltage-frequency curve, power headroom, occupancy and launch
amortisation - so calling the verify path compute-bound would need per-kernel counters this study
does not have. Read as a sensitivity threshold rather than a hardware prediction: **holding this
card's acceptance curve fixed**, width 8 overtakes width 5 once `c` drops below 0.0543, and
measuring `c` on another card needs one baseline and three widths there. Whether a 5090's `c` is
below it is not established here, and a different card can also move the acceptance curve, the
kernel family and the dispatch boundary.

One assumption is doing work there and is not verified: the calculation uses this card's
`mean_len` curve, taken at `UD-Q4_K_XL`. A higher-precision target may accept more, which would
raise `mean_len` at depth and make the required `c` less extreme. Phase Q walks the target
quantisation ladder to separate the two.

</details>

<details>
<summary>A missing "- 1" that produced plausible wrong numbers, and what it changed</summary>

`mean_len = (predicted_n - 1) / (predicted_n - accepted - 1)`. The `- 1` was missing at first and
the numbers it produced looked fine. The first generated token comes out of the prompt-processing
pass, not out of a decode forward, and leaving it in the count inflated the forward count by one.
Checking the derivation against the server's own `mean len` log line on all 625 speculative
requests is what found it.

The correction moves `c` by 0.8 % and changes nothing claimed here, but it is why
[`upstream/`](../upstream/) carries a one-line patch to expose the verification-step count the
server already holds: a derivation that reproduces plausible numbers and is quietly wrong by a
percent is exactly what an exposed counter prevents. Around 30 % of requests need one further
step removed, which is what truncation at the token cap looks like, and the API cannot say which,
so the figures above are low by under 1 % and are reported as such.

Across five prompt classes whose acceptance rates differ by nearly tenfold, the class means of
`k` span **0.26 % to 0.94 %** of their own mean, depending on the arm. An earlier version of this
section quoted 0.35-0.54 %, which was the dispersion of `k` over individual requests rather than
over class means. The record-level figure is the smaller and better-looking of the two, and it is
not the one this claim is about.

</details>

<details>
<summary>What the cost model rules out: the overhead is not paid on rejection</summary>

A state-rollback account charges the overhead to *rejection*: 48 of this model's 64 layers are
Gated DeltaNet and cannot roll back by truncating a KV suffix. Writing that as
`k = k_verify + r*w*(1 - acceptance)` makes `r` estimable from the slope of `k` against
acceptance, where `w` is the draft length per target forward pass.

Across an acceptance range of **0.096-0.980**, the bound holds in every completed matrix. Taking
the largest upper 95 % confidence limit on `r` among the arms the model applies to, and converting
it to the share of the cycle it would account for:

| matrix | arms fitted | largest upper limit on `r` | share of cycle cost |
|---|---|---|---|
| A | 5 | +0.00435 | 0.29 % |
| C | 3 | +0.00075 | 0.08 % |
| KV | 5 | +0.00293 | 0.22 % |
| NMAX | 12 | +0.04818 | 1.29 % |
| R2 | 14 | +0.01518 | 1.37 % |

The share is the comparable column, because `r` is per rejected *draft token* and arms at
different widths carry the same total cost at very different `r`. Nothing reaches 1.4 %.

Three corrections were needed before that sentence could be written, and each of them had been
silently wrong:

* **The verdict came from a point estimate's sign.** `r^2` was computed, printed, and never
  consulted, so an arm whose fit explained 13 % of the variance in `k` was announced in the same
  words as one explaining 99 %. Every `r` now carries a prompt-cluster bootstrap interval and the
  verdict comes from whether it excludes zero.
* **The bound was `max` over the arms' point estimates.** The maximum of several noisy estimates
  is biased upward and is not a bound on anything. It is now the largest upper confidence limit.
  That gate also decided whether this section's conclusion was printed at all, so on the Phase M
  data one arm clearing zero by 0.10 half-widths -- well inside the undercoverage measured at
  n = 25 -- suppressed the finding entirely.
* **`w` is not `n_max`.** The server reuses a surviving draft tail instead of re-drafting
  (`tools/server/server-context.cpp:2893`), so a cycle can cost a forward pass and generate
  nothing. On the MTP and DFlash arms the realised `w` sits within 0.7 % of `n_max` and the
  distinction is immaterial; on `dflash2-n8` it is 6.94 against 8, and on the 0.8B `draft-simple`
  arms it is **4.20 against 8** and **varies with acceptance at r = +0.94**. There the regressor
  is inside the response. The induced bias is positive in the slope and therefore negative in
  `r` -- which is the direction of this section's own conclusion, so those arms are reported and
  excluded from the bound rather than counted toward it.

Three arms return an `r` that is significantly *negative*: `k` rises with acceptance at constant
draft width. That is the opposite of a rollback account and is what a cost paid per position
verified looks like near saturation, where `mean_len` climbs toward `w + 1` while the cycle cost
does not.

What the bound covers is the component of cost proportional to `w*(1 - acceptance)`, and that
component is approximately none. It does not bound a fixed checkpoint paid every verification
step, a fixed restore paid once per rejection, or a cost depending on where in the draft the first
rejection lands: the first two are absorbed into `k_verify` and are invisible to a slope against
acceptance. Separating them needs per-step drafted and accepted lengths, which the server does not
yet report. The hypothesis was this repo's own, pre-registered, and is reported as unsupported in
the form it was written.

</details>

