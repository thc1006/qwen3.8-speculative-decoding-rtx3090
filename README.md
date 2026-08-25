# Qwen3.8-27B speculative decoding on a single RTX 3090

A controlled study of `draft-mtp` and `draft-dflash` on llama.cpp, with the hypotheses,
the analysis plan and the prior-art sweep committed **before** any measurement in
[`PREREGISTRATION.md`](PREREGISTRATION.md), which is append-only. Two of its own hypotheses have
since been recorded as unsupported.

Successor to [`thc1006/qwen3.6-speculative-decoding-rtx3090`](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090),
where the same question on a 3B-active MoE came out net negative on llama.cpp. That write-up
explained the loss by expert saturation: a draft of K tokens well below the ~94-token
saturation threshold forces the verify pass to load the union of K positions' expert slices. [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
is dense-hybrid - no experts, no routing, no union - so that mechanism cannot decide the answer
here, and the question was open again.

> **Status, 2026-08-25.** Complete: Phase A (875 measurements, 0 incidents), Phase R (1125),
> Phase R2 (1575, 0 incidents), Phase KV (175), the n-max ladder (1050), Phase C (750), and the
> four-build forced-warp intervention (600) with its disassembly and kernel benchmark. The depth
> ladder is running and has two of five rungs. Phase M and Phase Q are queued behind it. Later
> phases are designed and not yet measured; each says so where it appears.

**It is not open any more.**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="analysis/plot_headline_dark.png">
  <img alt="Dot-and-whisker plot of five speculative arms against a non-speculative baseline of 41.55 tok/s. mtp-n2 at verification width 3 is +59.8 % with a 95 % interval of +57.0 to +62.8; mtp-n3 +52.3 %; dflash2-n4 +51.9 %; mtp-n5 +32.1 %; dflash2-n7 +22.6 %. Every interval lies clear of zero." src="analysis/plot_headline.png">
</picture>

## Findings

| | |
|---|---|
| **Is it worth enabling?** | Yes. MTP at `--spec-draft-n-max 2` is **+59.8 %** [+57.0, +62.8] over no speculation. |
| **Which n-max?** | **2** for `draft-mtp`, **4** for `draft-dflash` - the best of those measured, and derived from a cost model rather than picked from a table. |
| **Does DFlash2 beat the built-in MTP head?** | No. **+51.9 %** at its own best depth. It drafts longer blocks and its fixed cost is lower, but acceptance falls faster with depth. |
| **Energy, or just time?** | Both, in direction. Board telemetry puts decode energy for a 400-token answer at roughly **-37 %** (3980 -> 2503 J). One limit of that figure is now measured rather than assumed. `power.draw` on Ampere is a rolling average of about a second, and the depth ladder samples the instantaneous field beside it: the two integrals agree to 0.00-0.34 % on the baselines and differ by 0.58-1.97 % on the speculative arms, always in the same direction. The averaged field understates exactly the arms being compared, so the saving is nearer **-36 %**, and `analyze.py` prints the gap per arm wherever energy appears. The request-boundary and prefill-subtraction limits still stand and still want an energy-counter remeasurement. No prior-art study publishes an energy figure for this model. |
| **Lossless at temperature 0?** | **No.** 76-80 % of greedy requests diverge from the non-speculative baseline. Deterministic, and it reproduces exactly across passes. |
| **Why does deeper drafting stop paying?** | Each extra verified position costs **c ~ 0.28** of a plain decode step. With both clocks pinned, the baseline and the speculative arms sit in opposite corners: bandwidth elasticity **0.80 against 0.14**, compute elasticity **0.27 against 0.76**. |
| **Does the MoE result carry over?** | No. The sign flips: net loss there, large net win here. |
| **Which prompts benefit?** | Within this suite, code and reasoning most, the Chinese tasks least - and `dflash2-n7` is **+22.6 % overall while being a net loss on three of five classes**. Thinking mode is collinear with the reason class and the Chinese prompts are different tasks rather than translations, so these are differences between the selected prompts, not language or class effects. |

**Contents** - [What this is not claiming](#what-this-is-not-claiming) |
[Results](#results-phase-a) | [Cost model](#a-cost-model-not-a-table) |
[Losslessness](#losslessness) | [Resource response](#resource-response) |
[Design](#design) | [Later phases](#later-phases) | [Reproduce](#reproduce) |
[Limitations](#limitations)

## What this is not claiming

The prior-art sweep found that several things a first draft of this README would have called
"first" are already published. **The throughput table is not the contribution.** What is left,
and what this repo went after, is the protocol and the axes nobody ran: a paired interleaved
design that puts an interval on every number, a thermal gate at arm entry, per-request energy,
byte-level output divergence, and a mechanism test that separates two competing explanations for
why deeper drafting stops paying.

<details>
<summary>The five results that already exist, and whose priority is theirs</summary>

- **Single RTX 3090 + DFlash2 + a Q4 target** is already reported twice in the comment thread of
  llama.cpp PR #27342, by `treo` (`UD-Q4_K_XL`, 32K ctx) and `ouening` (`UD-Q4_K_M`, 128K ctx,
  Windows).
- **llama.cpp `draft-mtp` on a 3090** is covered in depth by
  [`sudoingX/qwen38-mtp`](https://github.com/sudoingX/qwen38-mtp), across six 3090 rows plus
  quant, KV-type, power and DSpark sweeps.
- **Drafter-quantization comparison** is partly answered in that same PR thread, on a 32 GB card.
- **vLLM on a single 3090** for this model is covered by
  [`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090).
- **Losslessness on consumer hardware** was studied in
  [arXiv 2607.17283](https://arxiv.org/html/2607.17283), though on Apple silicon, with Qwen2.5,
  and with classic two-model speculation rather than MTP or DFlash2.

</details>

## Results: Phase A

7 arms x 25 prompts x 5 passes = **875 measurements, 0 incidents, 0 excluded, 0 quality-flagged.**
Intervals are a paired cluster bootstrap over prompts, on the class-stratified effect.

| arm | verify width | tok/s | vs own-tree baseline | tok/J | decode J per request |
|---|---:|---:|---|---:|---:|
| baseline @ master | - | 41.55 | - | 0.1005 | 3980 |
| baseline @ PR #27342 | - | **41.55** | - | 0.1005 | 3979 |
| **mtp-n2** | 3 | 66.39 | **+59.8 % [+57.0, +62.8]** | 0.1627 | **2503 (-37 %)** |
| mtp-n3 | 4 | 63.29 | +52.3 % [+48.5, +56.5] | 0.1549 | 2684 |
| dflash2-n4 | 5 | 63.13 | +51.9 % [+45.6, +58.2] | 0.1554 | 2835 |
| mtp-n5 | 6 | 54.89 | +32.1 % [+26.4, +37.8] | 0.1343 | 3228 |
| dflash2-n7 | 8 | 50.95 | +22.6 % [+14.7, +30.4] | 0.1251 | 3786 |

The two trees agree to 41.55 tok/s and produce **byte-identical output on 125/125 prompt-passes**,
so nothing in the DFlash2 numbers is attributable to the unmerged branch. Run-to-run CV within a
prompt is <= 0.3 %.

<details>
<summary>How the energy figures are measured, and why prefill is subtracted per arm</summary>

Both energy columns are decode-only. Prefill is measured separately, in its own eight-repetition
calibration per prompt, and subtracted. Counting it, the same request goes 4050 -> 2583 J, a 36.2 %
saving. Prefill is measured per arm rather than assumed constant, and it is not: 70.9 J for the
baseline against 83.2 J for `dflash2-n7`, because a speculative arm processes the prompt through
its drafter as well. `joule`, `tok/J` and `watt` appear zero times in PR #27342's 60-comment
thread.

</details>

### The headline number hides a sign change

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="analysis/plot_per_class_dark.png">
  <img alt="Heatmap of throughput change against baseline for five speculative arms across five prompt classes. Code and reasoning are strongly positive for every arm, from +55 % to +117 %. Chat, prose and Chinese fall towards zero as verification width grows and turn negative for dflash2-n7 at width 8: minus 4 % on chat, minus 11 % on prose, minus 29 % on Chinese." src="analysis/plot_per_class.png">
</picture>

<details>
<summary>The per-class numbers</summary>

| arm | code | reason | prose | chat | zh |
|---|---:|---:|---:|---:|---:|
| mtp-n2 | +92.0 % | +73.6 % | +48.6 % | +46.9 % | +37.7 % |
| mtp-n3 | +98.9 % | +71.2 % | +37.0 % | +31.3 % | +23.2 % |
| dflash2-n4 | +116.8 % | +89.4 % | +23.0 % | +30.2 % | +0.3 % |
| mtp-n5 | +90.7 % | +54.9 % | +7.1 % | +7.1 % | +0.6 % |
| **dflash2-n7** | +91.8 % | +65.4 % | **-11.1 %** | **-4.3 %** | **-28.7 %** |

</details>

`dflash2-n7` is +22.6 % overall. It is also a net loss on three of five classes. That is the same
failure this repo documents in the predecessor's own headline, where a mixture statistic got
reported as an effect - except here it runs the other way: the average flatters an arm that hurts
most of the workload. Which is why the primary endpoint is class-stratified.

### A cost model, not a table

llama.cpp reports enough per request to recover the cost of one speculative verification step in
units of a plain decode step, as `speedup = mean_len / k` with `k(w) = k0 + c*(w - 1)` and
`w = n_max + 1`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="analysis/plot_cost_model_dark.png">
  <img alt="Three stacked panels against verification width. Top: tokens accepted per target pass rises from 2.3 to 3.3 but falls far below a dotted line showing growth in proportion to width. Middle: the cost of one target pass rises linearly, with c equal to 0.2829 for draft-mtp and 0.2784 for draft-dflash. Bottom: speedup, the ratio of the two, falls from 1.60 to 1.23 across the widths measured." src="analysis/plot_cost_model.png">
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
  <source media="(prefers-color-scheme: dark)" srcset="analysis/plot_dispatch_boundary_dark.png">
  <img alt="The cost of one verification step against verification width, from the n-max ladder. Widths 2 through 8 for draft-mtp and 3, 5 and 7 for draft-dflash rise on a straight line, with c equal to 0.2904 at r-squared 0.9958 and 0.2481 at 0.9947. A vertical line marks MMVQ_MAX_BATCH_SIZE at 8. Width 9 sits past it with an open marker, 26 percent below the line for draft-mtp and 7 percent below for draft-dflash." src="analysis/plot_dispatch_boundary.png">
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

**`c` agrees to within 15 %** between the target's own built-in nextn head and a structurally
unrelated 1.1 GB block-diffusion drafter. `k` is the whole speculative cycle, though: target
verification, the drafter's own forward passes, sampling, launch and synchronisation, output
extraction and any per-step state management. Two methods that share everything except the drafter
narrow the marginal cost to that shared machinery without identifying which part of it, so `c` is
reported here as a total marginal cost per verified position and not as a target-verification
cost. Separating the components needs per-context event timing, a replay that skips drafter
compute, or a profiler decomposition; none of those is in this repo.

One threat to `c` can be checked without any of them. Once an arm diverges from its baseline it is
decoding a different token sequence, so what follows is not a comparison of two widths on one
trajectory. Between a fifth and a quarter of these requests come out byte-identical, and those
share the whole trajectory. Fitting on those alone gives **0.2898 against 0.2904 for `draft-mtp`
and 0.2476 against 0.2481 for `draft-dflash`**, a gap of 0.2 % in both. Divergence does not move
the coefficient here. `cost_model.py` prints this comparison on every run rather than leaving it
as a one-off.

`mean_len` saturates with depth while `k` grows linearly, so the ratio has an interior maximum in
principle. `phase_nmax` now brackets it: width 2 gives **+44.96 % [+43.45, +46.54]** and width 3
gives **+58.84 % [+55.90, +61.89]**, so the peak sits at width 3 with a tested and slower point on
each side and non-overlapping intervals. **The best setting is n-max 2 for MTP and n-max 4 for
DFlash2 on this card at this target quantisation**, and it is now a bracketed maximum rather than
the smallest width that happened to be tried. It remains the best setting selected on the same
data it was measured on; confirming it without that selection needs fresh prompts or fresh passes.

<details>
<summary>Why an RTX 5090 report recommends the opposite setting, and what would have to differ</summary>

The PR thread disagrees. `lance0` reports on an RTX 5090 with a `UD-Q6_K_XL` target that n-max 7
is right for DFlash2, since the drafter's `block_size` is 8 and lower values discard tokens the
block already paid for. Here n-max 4 beats n-max 7 by a wide margin, 1.520x against 1.228x.

The model says both can be true, and says what would have to differ. For width 8 to beat width 5
on this measured acceptance curve, `c` would have to be below **0.0543**; it is 0.2784 here, 5.1
times too large. Phase R2 shows what `c` responds to: over the tested GA102 clock ranges the baseline responds to
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
[`upstream/`](upstream/) carries a one-line patch to expose the verification-step count the
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
`k = k_verify + r*n_max*(1 - acceptance)` makes `r` estimable from the slope of `k` against
acceptance. Across an acceptance range of **0.096-0.918**, every arm returns **|r| <= 0.0028**
decode-steps per rejected token, r^2 between 0.001 and 0.060.

No relationship appears that is consistent with that specific proxy. What it bounds is the
component of cost proportional to `n_max*(1 - acceptance)`, and that component is approximately
none. It does not bound a fixed checkpoint paid every verification step, a fixed restore paid once
per rejection, or a cost depending on where in the draft the first rejection lands: the first two
are absorbed into `k_verify` and are invisible to a slope against acceptance. Separating them needs
per-step drafted and accepted lengths, which the server does not yet report. The hypothesis was
this repo's own, pre-registered, and is reported as unsupported in the form it was written.

</details>

### Losslessness

Speculative arms are byte-identical to their baseline on only 25-30 of 125 prompt-passes:
**76-80 % of requests diverge**, forking at a median 23 % into the text. Every arm is nonetheless
**100/100 reproducible across passes**, so the divergence is deterministic rather than noise.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="analysis/plot_width_partition_dark.png">
  <img alt="Five-by-five matrix giving the share of 25 prompts on which two speculative arms fork from the baseline at the same character. Arms at verification widths 3 and 4 agree with each other on 100 percent; arms at widths 5, 6 and 8 agree with each other on 100 percent; across the two groups agreement falls to 44 percent. The 100 percent block spans both drafters." src="analysis/plot_width_partition.png">
</picture>

Fork positions partition the arms into exactly two groups by verification width, `{3,4}` against
`{5,6,8}`, identically in all five passes. **The grouping crosses drafters**: width 5 and width 8
are DFlash2 while width 6 is the built-in MTP head, and all three agree with each other on every
prompt. So drafter identity does not predict the grouping and verification width does. That
boundary co-occurs with the CUDA `calc_nwarps` table switching `ncols_dst` from four warps to two.

Two things qualify that, and both were found by this repo looking for them.

**The intervention does not support the warp count as the cause.** Three builds of the same
revision, differing only in that table, were pre-registered with their outcomes written down
first. The forced-up build passes every validity gate - the greedy baseline is byte-identical
across builds on 25 of 25 prompts, the widths it did not touch are identical on 50 of 50, the
widths it did touch differ on 60 of 75, and disassembly confirms the edit changed exactly the
kernels it should and no others. But of the 18 prompts that can discriminate, the registered
prediction that widths 5, 6 and 8 adopt the `{3,4}` fork positions held on **3**. The forced-down
direction was void on its first attempt for a reason worth stating: its table row included width
1, and a drafter decodes one token at a time, so it perturbed every arm through its drafter.

That set has since been replaced by four builds from a single cmake configure, which is what the
first attempt lacked: its control had been built under a different configure, and a reconfigure
regenerates `flags.make` and recompiles every `ggml-base` source, which is enough to move a
width-1 greedy baseline that shares byte-identical kernel machine code. The fourth build is a
second stock one, and it is the control the first set never had. **control and control2 agree on
0 of 6202 SASS kernels differing and on 150 of 150 outputs byte for byte**, so the build is
deterministic and a difference from a forced build is the table.

The result is a null, and a clean one. Disassembly shows the edit reached the machine code and
only there: forced-up differs from control in 92 of 6202 kernels, all `mul_mat_vec_q`, at template
widths 5 to 8; forced-down2 in 46, at 3 and 4. `ggml_cuda_should_use_mmvq` falls through to
`ne11 <= MMVQ_MAX_BATCH_SIZE` on Ampere for every quantized type, so those kernels are the ones
that run. `test-backend-ops perf` puts the effect on the kernel at **+13.6 % at width 5 and
+26.7 % at width 8**, against a rebuild noise floor of 0.17 %. And the output does not move: **0
of 75 records differ for forced-up, 0 of 50 for forced-down2**.

So the warp count changes this kernel a great deal and changes no output byte. A fork position is
a property of the text, and a mechanism that cannot change the text cannot change where two texts
diverge. **The warp count is out as the cause**, on those grounds rather than on a measured
absence, and the co-occurrence above is reported as co-occurrence. What else changes at that width
is open.

**Every agreement is censored, not some of them.** An earlier version of this paragraph measured
the window in characters and reported 15 of 25 prompts censored, with the partition checked against
the 10 that were not. The design fixes the window in tokens, and characters per token span 1.36 to
6.17 across these prompts, so that split was an artefact of the wrong unit. Measured in tokens,
`harness/truncation_audit.py` gives 490 of 750 Phase A records diverged, 260 right-censored, and
**0 that reached EOS**. No record anywhere in this study stopped on its own, so no identity here is
exact: every one means "did not diverge within 400 tokens". There is no clean subset, because the
censoring is uniform, and the robustness check the earlier text claimed cannot be run on this data
at all. Forks resolve between token 6 and token 334, the latest at 83 % of the window. What settles
it is a larger budget, which is TODO.md item D2.

This corroborates [llama.cpp #25618](https://github.com/ggml-org/llama.cpp/issues/25618) rather
than discovering anything: that thread already establishes the phenomenon, its
quantization-dependence, its drafter-independence, and a root cause on the Vulkan side. What is
still open is the **CUDA** boundary, and a width-localised boundary is what this repo can add -
now with the intervention result attached, which points away from the mechanism the thread
proposes.
[llama.cpp #26750](https://github.com/ggml-org/llama.cpp/issues/26750) asks the same question on
Blackwell; this card is sm_86 and cannot answer it. See
[`docs/UPSTREAM_CONTRIBUTIONS.md`](docs/UPSTREAM_CONTRIBUTIONS.md).

### Resource response

<details>
<summary>The speculative arms boosted lower than their own baselines, so the speedups are understated</summary>

Every speculative arm ran at a **lower SM clock than the baseline it is compared against** - 1.98 %
lower for `dflash2-n7`, 4.17 % lower for `mtp-n5` - and 3 to 5 degrees hotter. Nothing was pinned;
this is the card boosting less because a speculative arm draws more power for the same wall time.

The direction matters. A treatment arm running *faster* than its control would inflate the effect;
one running slower deflates it. Correcting with this study's own SM-clock elasticity for the
interval those clocks sit in, 0.78 for the speculative arms:

| arm | clock vs its baseline | measured | at matched clock |
|---|---:|---:|---:|
| `mtp-n2` | -3.87 % | +59.77 % | ~ +64.7 % |
| `mtp-n3` | -4.01 % | +52.32 % | ~ +57.2 % |
| `dflash2-n4` | -3.06 % | +51.94 % | ~ +55.7 % |
| `mtp-n5` | -4.17 % | +32.10 % | ~ +36.5 % |
| `dflash2-n7` | -1.98 % | +22.63 % | ~ +24.6 % |

The measured column stays the headline, for two reasons. It is the conservative one, and it is
what the card actually delivers to a user who has not pinned anything. The matched-clock column is
an estimate from an elasticity measured in a different phase, not a measurement, and it is here so
that the boost difference is not mistaken for something working in the study's favour.

</details>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="analysis/plot_bound_by_dark.png">
  <img alt="Two panels. Top: bandwidth elasticity against compute elasticity at the top of the clock range. The baseline sits at 0.80 bandwidth and 0.27 compute; both speculative arms sit near 0.15 bandwidth and 0.78 compute, in the opposite corner. Bottom: compute elasticity by interval. From 600 to 1200 MHz all three are between 0.80 and 0.93; from 1200 to 1710 MHz the baseline falls to 0.27 while the speculative arms stay at 0.76 and 0.80." src="analysis/plot_bound_by.png">
</picture>

Phase R varied memory bandwidth and power budget independently, 1125 measurements, and confirmed
the assumption the design rests on: lowering the power limit to 250 W and 175 W leaves the memory
clock at 9501 MHz unchanged, so the two levers are separable on this card. Its own review then
found that a power cap is a poor compute lever, because the clock it produces is an outcome rather
than a setting. **Phase R2** re-ran the compute axis with the SM clock pinned at 600, 1200 and
1710 MHz, 1575 measurements, 0 incidents, and it is the one quoted here.

At the top of the clock range the two workloads sit in opposite corners:

| | bandwidth elasticity | compute elasticity | bound by |
|---|---:|---:|---|
| baseline | **0.79-0.81** | **0.27** | memory bandwidth |
| mtp-n3 | 0.13-0.15 | 0.76 | compute |
| mtp-n7 | 0.17-0.18 | 0.81 | compute |

The two elasticities very nearly swap. That is what speculation does to the workload: one target
pass scores several positions at once, so the decode stops waiting on memory and starts waiting on
arithmetic.

The regime matters and the intervals show where it changes. From 600 to 1200 MHz everything is
compute-starved and everything scales with clock: baseline 0.804, mtp-n3 0.913, mtp-n7 0.931.
From 1200 to 1710 MHz the baseline hits its bandwidth ceiling and stops responding, 0.266, while
the speculative arms keep scaling at 0.759 and 0.805. The ratio between them therefore is not one
number: it is 1.14x in the low regime and 2.85x in the high one, which is why this repo reports
elasticities per interval and never pools them across a regime change.

Pinning tightened the intervals to the third decimal, and it binds at five of the seven
conditions. It does not bind at the top two. A pin holds only while the power limit does not, and
a speculative arm draws more at the same clock, so at `sm1700` the methods land on 1710, 1698 and
1708 MHz against one request, and at `sm1700-bwhi` on 1710, 1689 and 1703. The elasticities in the
table above cross `sm1700`, and `harness/elasticity.py` marks every interval that crosses an
unmatched condition rather than leaving it to be noticed in the clock columns. The arithmetic is
unaffected, since each elasticity divides by that arm's own log clock ratio, but the comparison
between arms there spans slightly different ranges. Phase R, for contrast, was mismatched at 30.0 %
and 35.8 %.

## Design

| | |
|---|---|
| target | `unsloth/Qwen3.8-27B-GGUF`, `Qwen3.8-27B-UD-Q4_K_XL.gguf` (17.56 GB) |
| architecture | `qwen35`, 64 layers, `full_attention_interval: 4` -> **48 Gated DeltaNet + 16 full attention**, vocab 248320, native VL |
| MTP | embedded in the quant: `qwen35.nextn_predict_layers = 1`, `blk.64.nextn.*` present (verified by reading the GGUF) |
| GPU | 1 x RTX 3090 24 GB, driver 610.43.02, 420 W default, **reset to stock for the primary matrix** - the card was found overclocked and the first Phase A run was discarded ([`docs/GPU_AS_FOUND.md`](docs/GPU_AS_FOUND.md)) |
| host | Debian 13, kernel 6.12, i9-13900, 31 GB RAM |
| engine | llama.cpp from source, CUDA 13.3, `CMAKE_CUDA_ARCHITECTURES=86`, two trees with identical flags |
| trees | `master` @ `c060ca9` (build 200), **PR #27342** (DFlash2, unmerged) @ `d1a522f` |
| prompts | 25, balanced **5 per class** over code / prose / reason / chat / zh; every prompt written to exceed the 400-token cap |
| sampling | greedy, full sampler chain pinned explicitly, `cache_prompt: false`, `--parallel 1` |

<details>
<summary>The ten controls, and the specific failure each one prevents</summary>

Every one of these was added because something measurable went wrong without it.

| control | the failure it prevents |
|---|---|
| **arms interleaved within each pass, order rotated** | running arms sequentially confounds session drift with arm identity |
| **thermal gate at arm entry** | this card sits on its cap and loses **9.3 % of SM clock** (1950 -> 1769 MHz) over one pass; larger than several effects under study, and it lands inside every paired comparison |
| **dual-tree baseline** | DFlash2 needs an unmerged branch; comparing it to a master-tree baseline would conflate the method with the branch |
| **`cache_prompt: false`, verified via `t_cache_n`** | prompts share a system message within a class; prefix caching also *interacts* with speculation ([vLLM #38182](https://github.com/vllm-project/vllm/issues/38182)) - a confound of this exact shape forced a retraction in the sibling repo |
| **drafter assertion, log evidence + `t_draft_n > 0`** | the predecessor repo shipped a table row whose draft model never attached; a flag can be accepted and ignored |
| **port-ownership guard** | a killed-but-unreaped server keeps answering `/health`; a contributor to another study published three rows measured against a zombie |
| **class-stratified primary endpoint** | the effect has **opposite signs** across classes; a raw mean reports the prompt mixture as if it were a result |
| **cluster bootstrap over prompts** | passes of one prompt are repeated measures, not independent samples |
| **degeneracy screen relative to baseline** | collapsed output is fast; [vLLM #52475](https://github.com/vllm-project/vllm/issues/52475) reports MTP repetition collapse on this model family |
| **stock-clock enforcement** | this card arrived overclocked while the README said "stock"; the harness now reads the offsets and **refuses to run** unless they are zero or an overclock is declared |

A full audit of the predecessor repo's methodology is in
[`docs/METHODOLOGY_AUDIT.md`](docs/METHODOLOGY_AUDIT.md), including a re-analysis of that repo's
own committed data showing its headline effect was understated about two-fold by prompt-class
mixture.

</details>

<details>
<summary>Two things went wrong and are recorded, not smoothed over</summary>

1. **The card arrived overclocked** (memory +400 MHz, core +100 MHz, 450 W against a 420 W
   default) while this README described it as stock. The first Phase A run was **discarded**, the
   card reset, and the harness now refuses to start on a non-stock card.
   [`docs/GPU_AS_FOUND.md`](docs/GPU_AS_FOUND.md).
2. **The completed run's process crashed** with a glibc `double free or corruption` after writing
   its last record. The cause was a harness bug (`preexec_fn` alongside a sampling thread, now
   `start_new_session=True`). All 875 measurements survived; the final pass's derived comparisons
   were recomputed from the recorded text. `PREREGISTRATION.md`, Correction 2.

</details>

## Later phases

Each hypothesis was written down before its data existed, in the addenda to
[`PREREGISTRATION.md`](PREREGISTRATION.md). Six of the eight below have since been measured; the
status column says which.

| phase | question | status |
|---|---|---|
| **R2** | does the compute elasticity hold with the SM clock pinned rather than power-capped? | **complete**, 1575 measurements, 0 incidents |
| **KV** | does the width partition survive an f16 cache, or was it an artefact of q8_0? | complete |
| **n-max** | the full width ladder, 2 to 9, for the CUDA boundary question | **complete**, 1050 measurements. The registered prediction held: widths partition as `{2,3,4}` and `{5,6,7,8}`, with 9 on its own past the MMVQ dispatch limit |
| **C** | does drafter quantization change the answer, and does the predecessor's v3.0 need an erratum? | **complete**, 750 measurements, 0 incidents. It barely changes the answer and the highest precision is the slowest: q8 **+53.4 %**, q4k **+52.0 %**, bf16 **+48.5 %**, so a bf16 drafter costs about five points to run. The class effect is larger than the quantization effect: code +117 %, reason +90 %, zh +0.8 %. The three n-gram methods fail three different ways, and the counters separate them: `ngram-mod` has `t_draft_n = 0` on all 75 records and output byte-identical to baseline on all 75, so its flag was accepted and did nothing; `ngram-map-k` drafts on 6 of 75; `ngram-cache` drafts 9699 tokens and accepts **none**, which is where its -8.3 % comes from |
| **L** | does the long-context decode collapse of [#27623](https://github.com/ggml-org/llama.cpp/issues/27623) reproduce on sm_86, and does speculation survive it? | **running**, two of five rungs. Through 64 K there is no collapse and speculation neither amplifies nor masks the decline: retention against each method's own 8 K rung is 76.4 % for the baseline, 78.3 % for mtp-n2 and 78.0 % for dflash2-n4, and the speedups hold at +54 % and +47 %. The report puts the cliff past 80 K, which is the fourth rung, so the verdict is withheld until it runs6K |
| **M** | does `draft-mtp` at small n-max escape the MoE penalty that `draft-simple` at n-max 8 suffers? | designed, anchored on reproducing the predecessor's -44.6 % |
| **Q** | does the target quantization ladder move the marginal cost per verified position? | driver written, needs a 48 GB card above `UD-Q5_K_XL` |
| **V** | does the same comparison hold on vLLM rather than llama.cpp? | designed, [`docs/PHASE_V_DESIGN.md`](docs/PHASE_V_DESIGN.md) |

## Reproduce

```bash
# toolchain (Debian 13; NVIDIA CUDA repo already configured)
sudo apt-get install -y cuda-toolkit-13-3 ninja-build ccache

# two trees, identical flags: DFlash2 is an unmerged PR, so there is no prebuilt for it, and
# mixing a prebuilt master with a self-built PR binary would reintroduce a build confound
git clone https://github.com/ggml-org/llama.cpp llamacpp-master
cp -r llamacpp-master llamacpp-dflash2
cd llamacpp-dflash2 && git fetch origin pull/27342/head:pr-27342 && git checkout pr-27342 && cd ..
for t in llamacpp-master llamacpp-dflash2; do
  CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B $t/build -S $t -GNinja \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CCACHE=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build $t/build -j --target llama-server
done

# models
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf --local-dir models/target
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF --local-dir models/dflash2

# the harness's own tests: one case per defect this study shipped and later found
python3 harness/test_harness.py

# run + analyse (standard library only)
python3 harness/bench.py --matrix phase_a --passes 5 --out results/phase_a.json
python3 harness/analyze.py results/phase_a.json

# figures (the only step that needs a third-party package). Use the venv's interpreter:
# the harness itself runs on the system python, matplotlib is installed only here, and
# analysis/plot.py imports it at module level.
.venv/bin/pip install matplotlib && .venv/bin/python analysis/plot.py
```

`harness/bench.py --prompts-per-class 1` runs a reduced dry run; reduced runs label themselves in
the output file so they can never be mistaken for a full result.

## Limitations

- **Single card, single host.** Absolute tok/s are **not** comparable to the predecessor repo's
  numbers: that work used two other physical 3090s with 350 W caps.
- **`-c 8192` for the primary matrix.** Long-context behaviour - including whether speculation
  survives the ~25x decode collapse past ~80 K reported in
  [llama.cpp #27623](https://github.com/ggml-org/llama.cpp/issues/27623) - is a separate phase.
- **No EAGLE3.** The build supports `draft-eagle3` but no EAGLE3 drafter has been published for
  Qwen3.8-27B, so the method cannot be evaluated.
- **No multimodal input** in the primary matrix; llama.cpp has historically refused speculation
  together with `--mmproj` ([#19712](https://github.com/ggml-org/llama.cpp/issues/19712)).
- **Single-stream only** in Phase A. Speculative decoding is a single-stream optimisation and its
  advantage is reported by others to vanish by `--parallel 4`; measuring that is a later phase.

## License

MIT for code. Results (JSON / CSV) under CC-0.

## Author

Hsiu-Chi Tsai | GitHub [`thc1006`](https://github.com/thc1006)
