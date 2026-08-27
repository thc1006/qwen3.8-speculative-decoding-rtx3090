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

Phase C has since measured the predecessor's own drafting configuration on this dense target: a
0.8B draft-then-verify drafter at n-max 8 runs at **-29.8 % [-33.1, -26.4]**. The predecessor's
comparable arm, re-analysed on the class-stratified estimand this repo uses, was -21.5 %
([`docs/METHODOLOGY_AUDIT.md`](docs/METHODOLOGY_AUDIT.md)). The dense model loses **more**, with
no experts at all, so **expert routing is not necessary for that configuration to lose.**

That is a narrower statement than it may look, and it is deliberately the only one made here.
"Not necessary" is not "not a cause" and not "does not affect the size of the loss". Attributing
the cross-study difference to drafting method rather than architecture would take Phase M, and
Phase M's preregistered replication anchor failed
([`analysis/phase_m_anchor.txt`](analysis/phase_m_anchor.txt): *"nothing else in Phase M should be
read as a statement about the predecessor until this is understood"*). This README therefore
draws no architecture conclusion.

> **Evidence status, 2026-08-27.**
> **Complete and interpreted:** Phase A (875 request records, 0 incidents), Phase R (1125),
> Phase R2 (1575, 0 incidents), Phase KV (175), the n-max ladder (1050), Phase C (750), the
> depth ladder at five of five rungs (8 K, 32 K, 64 K, 80 K, 96 K, 180 records each), Phase
> Q-small at four of four rungs (375 each), and the four-build forced-warp intervention (600)
> with its disassembly and kernel benchmark.
> **Complete, interpretation limited:** Phase M, 1575 records, 0 incidents, both targets in one
> session -- its throughput and acceptance results stand, and its architecture, `k0`, `c` and
> fixed-cost interpretations are withheld; see the Findings row and Limitations. Phase Q at two
> rungs of four, which is every rung a 24 GB card can hold, reported as a cross-session
> association.
> Phase B complete, `--spec-draft-n-max` crossed with `--spec-draft-p-min`, 21 arm-passes, 525
> records, 0 exclusions, 2 recorded host-contention incidents.
> **Run, with most of its arms unable to start:** Phase V, 75 records. `baseline-vllm` serves and
> gives the cross-engine decode-rate anchor; both MTP arms fail to load on this card, six recorded
> failures, because the MTP module allocates its own bf16 `embed_tokens` and `lm_head`, 2.37 GiB
> each, on top of a 17.33 GiB target. Filed as
> [vllm#53887](https://github.com/vllm-project/vllm/issues/53887).

**For this single-stream, 8K-context, 400-token regime, decode throughput is not open
any more.** Semantic quality, mechanism, multi-user serving, generalisation across GPUs
and the exact energy magnitude are.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="analysis/plot_headline_dark.png">
  <img alt="Dot-and-whisker plot of five speculative arms against a non-speculative baseline of 41.55 tok/s. mtp-n2 at verification width 3 is +59.8 % with a nominal 95 % interval of +57.0 to +62.8; mtp-n3 +52.3 %; dflash2-n4 +51.9 %; mtp-n5 +32.1 %; dflash2-n7 +22.6 %. Every interval lies clear of zero." src="analysis/plot_headline.png">
</picture>

## Findings

| | |
|---|---|
| **Is it worth enabling?** | Yes. MTP at `--spec-draft-n-max 2` is **+59.8 %** [+57.0, +62.8] over no speculation. |
| **Which n-max?** | **2** for both, on the completed ladder. For `draft-mtp` it is the highest of the eight depths tested and has a slower tested point on each side. For `draft-dflash` it is the highest of the four tested, 1.7 points above `n-max=4`, but it is also the shallowest DFlash2 depth in the ladder, so it is not bracketed, and no direct paired interval between 2 and 4 has been computed. |
| **Does DFlash2 beat the built-in MTP head?** | Not on the best point estimate: **+58.8 %** for MTP `n-max=2` against **+53.7 %** for DFlash2 `n-max=2` on the same ladder. This is not a paired test between the two methods, and they run on different llama.cpp trees against their own matched baselines. At the 80 K depth rung the ordering reverses, on overlapping intervals. |
| **Energy, or just time?** | Both, in direction, and the direction is smaller than it first read. Board telemetry puts decode energy for a 400-token answer at **-37 %** (3980 -> 2503 J); two characterised effects bias the relative comparison in the same direction, and applying both as a sensitivity adjustment brings it nearer **-35 %**. That is a sensitivity-adjusted estimate, not a counter-based correction. All current telemetry sensitivity checks preserve the direction. The magnitude stays provisional, and **not** because a hardware energy counter has yet to be read: published characterisation of this sensor puts its steady-state error at a proportional ±5 % in either direction, larger than the corrections applied here, and NVIDIA's own cumulative-energy counter is reported to disagree with the integral of `power.draw` by about a factor of two. Only an external power meter would settle it. What this card does **not** suffer is the 25 %-of-runtime sampling gap that dominates A100 and H100 measurements; the RTX 3090 samples its whole runtime. Details below. |
| **Bit-exact with serial greedy decoding?** | **No.** Re-run with the generation cap raised to 1600 tokens, **92-100 % of requests diverge** depending on the arm, and 267 of 525 records stop at EOS rather than on the cap. Right-censoring falls from 260 of 750 records at a 400-token cap to **9 of 375**; those 9 mean no divergence was observed inside the window, which is not identity through to the end of an answer. The divergence is deterministic and reproduces exactly across passes, 125 of 125 repeated cells. |
| **Why does deeper drafting stop paying?** | Each extra verified position costs **c ~ 0.25-0.29** of a plain decode step across the whole speculative cycle, while accepted length shows diminishing increments. Over the measured clock intervals the baseline is the more memory-clock-sensitive workload and the speculative arms the more SM-clock-sensitive ones. That is a response measurement, not a roofline or a per-kernel bottleneck attribution. |
| **Does the predecessor's negative result transfer?** | **Data complete, causal and cost interpretation withheld.** Phase M ran both targets in one session, 1575 records, 0 incidents. In the per-protocol series the built-in MTP head has positive point estimates on both targets (MoE **+29.2 %**, dense **+59.5 %** at n-max 2) and the 0.8B `draft-simple` arms negative ones on both (MoE -59.6 % to -70.8 %, dense -28.8 % to -34.9 %). Three things stop that from becoming a mechanism. The preregistered replication anchor **does not hold** (-65.6 % against a registered -32 % to -12 %), and the phase's own gate says nothing in it may then be read as a statement about the predecessor. 33 records are excluded from the per-protocol series for stopping before the token cap; they are one prompt, `zh_self_intro`, across all three passes of all eleven MoE arms including `baseline-moe`, so the MoE half is estimated on 24 of 25 prompts rather than on a treatment-correlated subset -- balanced, but still an exclusion decided by an outcome. The analyser now tabulates the intention-to-treat series beside the per-protocol one: across the nineteen reported arm effects the two differ by at most **0.42 points**, and every dense-target effect is unchanged, so the exclusion rule does almost no numerical work here. And the `mean_len` derivation every cost quantity rests on **fails its own integrity check on this phase**: mean gap -0.3494, worst 2.9054 over 1425 requests, against a documented bound of under 1 %. `k`, `c`, `k0`, the marginal-cost equality and the per-cycle fixed-cost ratio are therefore withdrawn from this README; `cost_model.py` now refuses to print them for a result that fails that check. |
| **Which prompts benefit?** | Within this suite, code and reasoning most, the Chinese tasks least - and `dflash2-n7` is **+22.6 % overall while having negative point estimates in three of the five declared classes**. Thinking mode is collinear with the reason class and the Chinese prompts are different tasks rather than translations, so these are differences between the selected prompts, not language or class effects. |

**Contents** - [What this is not claiming](#what-this-is-not-claiming) |
[Results](#results-phase-a) | [Cost model](#the-verification-step-cost-model) |
[Greedy divergence](#greedy-output-divergence) | [Clock response](#clock-response) |
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

7 arms x 25 prompts x 5 passes = **875 request records, 0 incidents, 0 excluded, 0 quality-flagged.**
The inferential unit is the prompt, `n = 25`; the 5 passes are repeated measurements of the same
prompt, not independent samples, and 875 is not a sample size. Intervals are a paired cluster
bootstrap over prompts, on the class-stratified effect, and are **nominal** 95 %: this repo's own
simulation puts the percentile interval's actual coverage between **86.3 % and 93.7 %** at
`n = 25`, so the widths printed here should be read as potentially under-covering rather than as
exact 95 % statements. Those are the reproducible figures, from
[`analysis/bootstrap_coverage.txt`](analysis/bootstrap_coverage.txt): four synthetic processes at
300 replications x 2000 resamples each. An older set, 88.0-90.9 %, is quoted in `stats.py` from an
800-replication simulation whose code was never in the repository; the reproduction agrees with it
to within about one Monte Carlo standard error on two of the three continuous processes and sits
2.0 standard errors from the third, so neither set should be read as a precise constant. A coverage
figure is itself an estimate from Bernoulli trials: at 300 replications its own Monte Carlo standard
error is 1.4 to 2.0 points depending on the process, and by the standard formula ([Morris, White and
Crowther 2019](https://onlinelibrary.wiley.com/doi/10.1002/sim.8086)) about 1900 replications would
be needed to pin it to half a point. `coverage_sim.py` now prints that standard error beside every
row rather than leaving the reader to derive which differences are real. All of
them come from synthetic data-generating processes, not from this data's own unknown distribution,
so they diagnose the estimator rather than quantify this interval; the primary Phase A effects sit
far from zero under either set.
`analyze.py` names any verdict that sits
inside that margin.

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
so no no-speculation offset between the branches was detected here, and every DFlash2 arm is
estimated against its same-tree baseline. That controls the branch's baseline main effect. It
cannot rule out an interaction between the branch and the DFlash2 method itself, because
`draft-dflash` does not exist on the master tree and so cannot be run there. Run-to-run CV within
a prompt is <= 0.3 %.

<details>
<summary>How the energy figures are measured, and why prefill is subtracted per arm</summary>

Both energy columns are decode-only. Prefill is measured separately, in its own eight-repetition
calibration per prompt, and subtracted. Counting it, the same request goes 4050 -> 2583 J, a 36.2 %
saving. Prefill is measured per arm rather than assumed constant, and it is not: 70.9 J for the
baseline against 83.2 J for `dflash2-n7`, because a speculative arm processes the prompt through
its drafter as well. `joule`, `tok/J` and `watt` appear zero times in PR #27342's 60-comment
thread.

Both, in direction, and the direction is smaller than it first read. Board telemetry puts decode energy for a 400-token answer at **-37 %** (3980 -> 2503 J). All three limits on that figure have now been measured rather than named, and two of the three bias the **relative** comparison in the same direction. `power.draw` on Ampere is a rolling average of about a second: sampled beside the instantaneous field, the two integrals agree to 0.00-0.34 % on the baselines and differ by 0.58-1.97 % on the speculative arms, always the same sign, so the averaged field understates exactly the arms being compared, worth about 1.1 points. The prefill subtraction removes a `max_tokens=1` calibration that runs on a server with the drafter already loaded, so it costs 10-17 % more energy on a speculative arm than on a baseline and takes out more than prefill: worth 0.3 to 0.8 points. The integral runs first sample to last, and the sampler's period is its nvidia-smi query plus the interval rather than the interval alone, so about 4 % of the request sits outside the window; that one limits the absolute joule totals rather than the ratio, which is why it is not one of the two above. Applying the two relative biases as a sensitivity adjustment brings the saving nearer **-35 %**. This is also not an energy comparison for an identical or quality-equivalent answer: most speculative generations follow a different token trajectory inside the window, and semantic quality is not scored. The energy columns are point estimates; no prompt-cluster interval is reported for them. `analyze.py` prints the per-arm gap and the window coverage wherever energy appears. Two facts about the instrument itself bound what any of this can mean, both from [*Part-time Power Measurements: nvidia-smi's Lack of Attention*](https://arxiv.org/abs/2312.02741), which characterises NVIDIA's built-in sensor against external meters. The first is favourable and specific to this card: the RTX 3090 has an instant rise time, a 100 ms update period and a **100 ms averaging window**, so the sensor samples its whole runtime. The A100 and H100 average over 25 ms of each 100 ms period and therefore sample only 25 % of it -- "during the other 75 % of the time, the GPU can be using drastically different power" -- and that is where that paper's largest errors come from. **None of that applies here.** The second fact is not favourable: the sensor's steady-state error is **proportional, roughly ±5 %**, rather than the flat ±5 W NVIDIA specifies, and it runs in **both directions** depending on the individual board's component tolerances. That is larger than the 1.1-point correction above and, unlike it, is not one-sided, so it does not cancel in a ratio by construction. It is not folded into any figure here, because a bidirectional instrument error of unknown sign on this particular board cannot be corrected for, only reported. Reading `nvmlDeviceGetTotalEnergyConsumption` would **not** settle the magnitude either: a report on NVIDIA's own developer forum, [confirmed there by a second user](https://forums.developer.nvidia.com/t/value-from-nvmldevicegettotalenergyconsumption-seems-to-be-off-by-a-factor/336318), puts that counter roughly a factor of two below the integral of `power.draw` over the same interval and says the gap widens the more often power is polled, with no vendor resolution. What would settle it is an external power meter, which is the reference the study above used. None of the sources in this repository's dated 2026-08-24 prior-art sweep reported an energy figure for this model and setup.

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
reported as an effect - except here it runs the other way: the average flatters an arm that is
negative on most of the classes this suite declares. It is not negative on most of any real
workload, because the five classes carry equal weight by construction and no deployment traffic
mix has been measured. Which is why the primary endpoint is class-stratified, and why a
deployment figure would have to reweight these classes by its own traffic.

### The verification-step cost model

`speedup = mean_len / k`, with `k(w) = k0 + c*(w-1)` fitted inside the MMVQ dispatch path. On the
completed ladder that is **c = 0.2904** for `draft-mtp` over widths 2-8 and **0.2481** for
`draft-dflash` over 3, 5 and 7. `k(w)` is curved, so a slope is a chord and two chords over
different width ranges are not the same quantity; compared on the widths both cover, 3, 5 and 7,
the difference is **-0.0473 [-0.0489, -0.0456]**. The two curves differ by a straight line to
within 2.4e-4, so the curvature they share cancels and the difference survives it. The chords are
not shared between the two methods over the widths both cover. A configured n-max of 8 puts the
requested width at 9, past `MMVQ_MAX_BATCH_SIZE`; what each method actually fills there differs
(draft-mtp 8.93 columns of 9, draft-dflash 7.94), so that point is analysed separately per method
rather than labelled as one boundary crossing.

Phase M fitted the same method on a 35B-A3B MoE and a 27B dense target in one session and is
**excluded from this section**: its `mean_len` derivation fails `cost_model.py`'s own integrity
check on that phase, and every quantity here is a function of `mean_len`. The dense n-max ladder
above stands on its own check. What the exclusion costs is the cross-architecture comparison of
`c`; nothing in this repository currently bounds a difference in marginal cost between the two
targets.

Derivation, the dispatch boundary, and what `k` does and does not identify:
[`docs/COST_MODEL.md`](docs/COST_MODEL.md).

### Greedy output divergence

At a 400-token cap **76-80 % of requests had diverged from serial greedy decoding**, and every
request hit that cap without reaching EOS, so the rest were right-censored rather than identical.
Re-running the same prompts and arms at **1600 tokens** resolves most of that: divergence reaches
**100 % for dflash2-n4, dflash2-n7 and mtp-n5**, 96 % for mtp-n2 and 92 % for mtp-n3; 267 of 525
records now stop at EOS; and censoring falls to **9 of 375 records on 2 of 25 prompts**. It is
deterministic and identical across passes at both caps, 150 of 150 and 125 of 125 repeated cells.

The first-divergence signatures form stable groups by effective verification width rather than by
drafter, **{3,4} and {5,6,8} at both caps**. Widths are grouped only on prompts where both actually
diverged: at 400 that was 19-20 of 25 prompts per pair, at 1600 it is 23 of 25 for widths 3 and 4
and all 25 for the rest, so the larger budget leaves the same partition resting on more evidence.
The four-build intervention still falsifies the tested `calc_nwarps` change as the cause of the
grouping.

The matrix, the censoring accounting and the width partition:
[`docs/GREEDY_DIVERGENCE.md`](docs/GREEDY_DIVERGENCE.md).

### Clock response

With the SM clock pinned rather than power-capped, the baseline and the speculative arms respond
to opposite clocks: memory-clock elasticity **0.79-0.81 against 0.13-0.18**, SM-clock elasticity
**0.27 against 0.76-0.81**. The ordering also changes with the interval, 1.14x apart below
1200 MHz and 2.85x above it, which is why elasticities are never pooled across that boundary.
Nothing here counts bytes moved or arithmetic issued, so it locates neither workload against a
hardware limit.

Per-interval intervals and the pinning method: [`docs/RESOURCE_RESPONSE.md`](docs/RESOURCE_RESPONSE.md).

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
   `start_new_session=True`). All 875 request records survived; the final pass's derived comparisons
   were recomputed from the recorded text. `PREREGISTRATION.md`, Correction 2.

</details>

## Later phases

Each hypothesis was written down before its data existed, in the addenda to
[`PREREGISTRATION.md`](PREREGISTRATION.md). Their status is given per row below, and it is mixed -- complete and interpreted, complete with
interpretation withheld, partially complete, running, and blocked. The
status column says which.

| phase | question | status |
|---|---|---|
| **B** | under confidence gating, does the overhead scale with tokens drafted or tokens rejected? | **complete**, 525 request records, `mtp-n3` and `mtp-n7` crossed with `p_min` 0.00 / 0.50 / 0.75 over three passes, 0 exclusions, 2 recorded host-contention incidents from processes of my own. **The cost tracks tokens drafted, not tokens rejected.** One-parameter models on the extensive form: cost per drafted token fits at r2 **0.9781** against **0.8242** for cost per rejected, and the residual-sum difference clears zero by 18.49 half-widths. Adding a per-step term keeps the ordering -- step + drafted at 4.229 ms/step and 6.112 ms/token, r2 **0.9912**, against step + rejected at r2 0.9687, clearing zero by **3.57 half-widths** -- and the winner does not change when the forward count is shifted from F-2 to F+1, which is the direction and size that derivation is known to be wrong by. H2 and H2' and the gate-depth arm design were registered in the initial commit; the model-comparison implementation was committed before the run finished, and the forward-count robustness sweep was added afterwards. What this does **not** do is identify a physical cause: it compares two counting regressors against excess time, with no intervention on quantization or arithmetic intensity. The joint fit is **not identified** -- the two regressors correlate at +0.9963 across arm means and the joint solution puts a negative coefficient on rejection -- and the absolute ms/step and ms/token wait on an exact verification-step count. Report: [`analysis/phase_b_mechanism.txt`](analysis/phase_b_mechanism.txt) |
| **R2** | does the compute elasticity hold with the SM clock pinned rather than power-capped? | **complete**, 1575 request records, 0 incidents |
| **KV** | does the width partition survive an f16 cache, or was it an artefact of q8_0? | complete |
| **n-max** | the full width ladder, 2 to 9, for the CUDA boundary question | **complete**, 1050 request records. Within the 400-token window, widths 2-8 produce two stable first-fork and censoring signatures, `{2,3,4}` and `{5,6,7,8}`, with 9 on its own past the MMVQ dispatch limit. The registered partition matched, but the four-build intervention then falsified warp count as its cause, so this is an observational signature and not a mechanism |
| **C** | does drafter quantization change the answer, and does the predecessor's v3.0 need an erratum? | **complete**, 750 request records, 0 incidents. It barely changes the answer. The point estimates against the same-tree baseline are ordered q8 **+53.4 %**, q4k **+52.0 %**, bf16 **+48.5 %**; no direct paired interval between two drafter precisions has been computed, so that ordering is descriptive and not a demonstrated separation. The class effect dwarfs the quantization effect: across the three precisions code runs +111 % to +118 %, reason +86 % to +92 % and zh -2.3 % to +0.8 %, a spread of more than a hundred points between classes against five between precisions. The three n-gram rows are activation diagnostics rather than three comparable efficacy measurements. `ngram-mod` emitted no drafts at all on 75 of 75 records, and that is the method working as designed, not a flag being ignored: its default `n_min = 48` discards the whole draft unless a match continues for 48 consecutive tokens, which a 400-token general writing, code and reasoning suite does not produce (Correction 25). Its -0.20 % measures entering the speculative path and drafting nothing, and its 75/75 match with the baseline is the absence of speculation rather than lossless speculation. `ngram-map-k` drafts on 6 of 75. Only `ngram-cache` is an active method here: it drafts 9699 tokens, accepts **none**, and that is where its -8.3 % comes from |
| **L** | does the long-context decode collapse of [#27623](https://github.com/ggml-org/llama.cpp/issues/27623) reproduce on sm_86, and does speculation survive it? | **complete**, five of five rungs, 180 records each, 0 incidents. **It does not reproduce.** Through a realised 98 300 tokens, past the 91 K worked example the report publishes, the baseline goes 39.7 -> 26.5 tok/s: a factor of **1.5 against the reported 25**, with the largest single-rung drop 1.16x and that on entering 64 K rather than past 80 K. The SM clock falls 1.60 % over the ladder, worth -0.43 points at elasticity 0.266, so the decline is depth and not throttling. Speculation survives it and `draft-dflash` survives it best: retention against each method's own 8 K rung is 66.9 % for the baseline, 68.8 % for mtp-n2 and **74.6 % for dflash2-n4**, whose acceptance rises slightly over the ladder, 2.607 to 2.650, while MTP's is flat. Its speedup leads at the two deepest rungs, +59.8 % [+50.2, +69.4] against +53.4 % [+48.8, +58.1] at 96 K, on intervals that overlap, so that ordering is a consistent point estimate and not a separation |
| **M** | does `draft-mtp` at small n-max escape the penalty that a 0.8B `draft-simple` at n-max 8 suffers, and does the architecture decide it? | **Data collection complete, 1575 records, 0 incidents; causal and cost interpretation withheld.** In the per-protocol series MTP has positive point estimates on both targets and `draft-simple` negative ones on both, each peaking at n-max 2 (MoE +29.2 % [+26.6, +31.8], dense +59.5 % [+56.6, +62.5]), and acceptance tracks the drafter rather than the target (78 % for the built-in head on both, 21-23 % for the 0.8B on both). What the phase does **not** currently identify: the preregistered **anchor does not hold** -- the 0.8B arm it replicates came out -65.6 % [-67.6, -63.7] against a registered -32 % to -12 % -- and the phase's own gate then forbids reading anything in it as a statement about the predecessor or about architecture. 33 records are excluded from the per-protocol series for stopping before the cap; they are one prompt, `zh_self_intro`, across all three passes of all eleven MoE arms including `baseline-moe`, so the exclusion is balanced within the MoE half rather than treatment-correlated. It is still decided by an outcome, so the analyser now tabulates the intention-to-treat series beside the per-protocol one, and the two differ by **at most 0.42 points** across nineteen arms, with every dense arm identical: the exclusion rule does essentially no work, which is a measurement rather than an argument. And the `mean_len` derivation underneath every cost quantity **fails its own integrity check here** (mean gap -0.3494, worst 2.9054 over 1425 requests, against a documented bound under 1 %), so `k0`, `c`, the marginal-cost equality and the 3.1-fold fixed-cost ratio this row used to report are withdrawn. Corrections 9, 10, 13-19c. Figure: [`analysis/plot_phase_m.png`](analysis/plot_phase_m.png) |
| **Q** | is the fitted whole-cycle cost chord associated with target-checkpoint quantization? | **Two rungs of four complete**, which is every rung this card can hold: UD-Q4_K_XL and UD-Q5_K_XL, 300 records and 0 incidents each, no arm-pass above sd 0.28 % against its own repeats. The fitted chord is 0.2842 at Q4 against 0.2554 at Q5; paired over the same 25 prompts on the shared widths {3,4,6} the difference is **+0.0288 [+0.0271, +0.0303]**, 10.1 % of Q4's, and 9.5x the widest within-rung pass spread. **In wall time the sign reverses**: the decode steps differ by 13.8 %, so the rung that pays 10 % less relative to itself pays 0.289 ms more per position. Acceptance moves at most +0.0079 and realised width at most 0.0027 across the rungs, every interval covering zero, so the drafter's proposal behaviour is stable within this design's resolution -- which is a statement about what it proposes and not about what its forward pass costs. **This is a cross-session association, not a causal estimate of target-verification cost.** The two rungs are separate sessions about eight hours apart, prompt pairing removes prompt difficulty but not hours-scale drift, and the MTP head is embedded in the target gguf and is quantized with it, so verifier and drafter-head compute move together. An interleaved rung design or a fixed external drafter is what would separate them. Byte-level divergence does not resolve: 24.0 % identical falls to 12.0 % at n-max 2 on intervals spanning 32 points -- unmeasured, not absent. Q6 and Q8_0 need 27.5 and 31.3 GB of VRAM and are **blocked on the card, not on disk**; an earlier version of this row said disk, because the driver was sizing a download from a VRAM table and demanding 33 GB for a 19.44 GiB file. Corrections 10, 11 |
| **Qs** | does the bf16 anchor #25618 rests on actually hold, and does #26750's CUDA acceptance figure reproduce on a second CUDA architecture? | **complete**, four rungs, 375 records each, 0 incidents. **The anchor holds as an effect and not as parity.** The share with **no divergence observed through output token 400** against each rung's own baseline is 16 / 8 / 4 % across Q4_K_M, Q6_K, Q8_0 and **52 % at BF16** -- every request stops at the cap and none reaches EOS, so a match inside the window is right-censored rather than identity to the end of an answer -- paired over the same prompts, the Q4_K_M-to-BF16 shift is **+36 to +44 pp** across the four tested MTP depths. Three of the four clear this repository's own 1.3-half-width sensitivity rule; the `mtp-n2` pair, **+36.0 pp [+16.0, +52.0]**, clears zero by only **0.89 half-widths**, which is inside the margin where the measured undercoverage can reach zero, so it is reported and not leaned on alone. But 52 % is not parity: 36 of 75 requests still diverge with **bf16 model weights** (the K/V cache is `q8_0` on every rung, so this is a weight-precision ladder and not an unquantized target), so #25618's "stays bit-identical on bf16" is too strong as written. Within the quantized rungs the rate *falls* with bit width, so bf16 is off that line rather than its endpoint. `mtp-n6@Q4_K_M` is the matched configuration for [#26750](https://github.com/ggml-org/llama.cpp/issues/26750) and measures **35.0 % [32.9, 37.3]** on sm_86, which is **57 points below** the ~92 % that report gives for Vulkan. Whether it agrees with that report's CUDA figure is **not established**: an overlap of intervals is a failure to exclude, not a reproduction, and what is unresolved is comparability rather than the figures: Correction 26 read both halves of that range from the issue and they are real and both CUDA -- 35.8 % on an RTX PRO 4000 (Blackwell) headline row and 40.7 % across four context and parallel sweep rows -- but that is a different CUDA architecture and a different prompt population, and the estimator behind it is **not known** to be the one computed here -- a class-stratified mean of per-request acceptance -- rather than a server-log aggregate over all drafted tokens, which is what `llama-server` itself reports. `c` falls with bit width (-0.019 per bit, clear of zero) but **saturates**, r2 0.666, and in wall time there is no trend at all (r2 0.019) because bf16's decode step is 2.44x Q4_K_M's. Acceptance is flat across the whole ladder, so none of that is the drafter moving. Scored in Correction 22 against hypotheses registered in Correction 21 |
| **V** | does the same comparison hold on vLLM rather than llama.cpp? | **Run, and what this card can produce is one arm and six recorded failures.** 75 records, `baseline-vllm` only, three passes at 47.52 / 47.53 / 47.52 tok/s -- a 0.02 % spread, and the first decode-only rate this study has from vLLM, taken from `vllm:request_decode_time_seconds` over `vllm:generation_tokens_total` rather than from wall time. Prefill is 1.29 % of inference on these requests, which is what a wall-clock rate would have folded into the comparison. Both MTP arms failed to start on all three passes: the same 2.37 GiB allocation at `qwen3_5_mtp.py:244` every time, which is `vocab_size 248320 x hidden_size 5120 x 2 bytes` for a bf16 `lm_head` the checkpoint does not contain, on top of a 17.33 GiB target. Filed as [vllm#53887](https://github.com/vllm-project/vllm/issues/53887). Design and the memory arithmetic: [`docs/PHASE_V_DESIGN.md`](docs/PHASE_V_DESIGN.md) |

## Reproduce

```bash
./scripts/reproduce_phase_a.sh
```

That is the whole procedure. The script reads every version-specific value from
[`repro/phase_a.lock.json`](repro/phase_a.lock.json) rather than carrying its own copy, so a rerun
and the record of what was measured cannot drift apart, and it stops on the failures that have
actually cost this study a run:

- the two llama.cpp trees are compared at their **full 40-character** commits,
  `c060ca974c773c7c3d17fd1b66dc9d312bc292c0` and `d1a522fc89c96d1a3057e35681f0c4859810623c`. A
  short prefix can resolve to a different object as a repository grows, which is the one thing
  pinning exists to prevent;
- both trees are configured and built with identical flags, because mixing a prebuilt master with
  a self-built PR binary reintroduces the build confound the two-tree design exists to remove;
- checksums are checked against
  [`models/SHA256SUMS.phase_a`](models/SHA256SUMS.phase_a), which lists **only the two files this
  phase loads**. The full manifest covers Phase M's MoE target, Phase Q's ladder rung and Phase
  Q-small's four rungs as well, so checking against it on a clean machine reports missing files
  for models a Phase A reproduction never downloads;
- the card is checked for compute capability 8.6 and about 20 GB free before anything is built;
- the harness's own tests must pass first;
- the result is written to `results/reproductions/phase_a_<host>_<utc>.json`. It never overwrites
  `results/phase_a.json`, which is the artifact you would be comparing against;
- the record count is checked against the 875 the lock file declares, and any incident is
  reported rather than left in the file.

Absolute tok/s are host-specific. Compare the paired effects, not the levels; see
[the fleet note](docs/GPU_AS_FOUND.md) for why figures from different hosts are never pooled here.

`python3 harness/bench.py --matrix phase_a --prompts-per-class 1 --out /tmp/dry.json` is a reduced
dry run. Reduced runs label themselves in the output file, so they can never be read back as a
full result.

Figures need matplotlib, which is the only third-party dependency and is not needed to reproduce
the numbers:

```bash
.venv/bin/pip install matplotlib && .venv/bin/python analysis/plot.py
```

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
- **Long-output workload.** Every primary prompt was written to reach the 400-token cap. Short
  answers and natural EOS termination are not represented, so this does not transfer to ordinary
  short chat traffic.
- **Curated prompts, not traffic.** The five classes carry equal weight by design. A deployment
  figure would have to reweight them by its own traffic mix, which has not been measured.
- **Statistical scope.** The inferential unit is 25 prompts, not 875 records. The percentile
  cluster bootstrap shows material undercoverage at that size on some tested processes
  (86.3-93.7 % against a nominal 95 %, four synthetic processes, 300 replications each), so the
  printed intervals may be too narrow, and none of them carry uncertainty from changing host,
  card, build, model or prompt population. The prompts were purposively constructed rather than
  sampled from deployment traffic, so the bootstrap measures sensitivity to resampling this suite
  under its class structure and is not a population-representative traffic interval. Intervals
  across secondary arms, classes and follow-up phases are nominal and unadjusted for multiplicity;
  they are not simultaneous 95 % family-wise statements.
- **Output agreement is right-censored, much less so since the cap was raised.** At a 400-token cap
  every primary request hit the cap and none reached EOS. The 1600-token re-run leaves 9 of 375
  records censored and 267 of 525 reaching EOS. Those 9 still mean no divergence was observed
  within 1600 tokens; they are not a statement about the whole answer. The re-run computed no
  divergence for the same-tree `baseline@pr27342` control, so that check exists only at 400.
- **No semantic-quality benchmark.** The harness measures byte-level divergence and obvious
  degeneration. Whether a diverged answer is better, worse or equivalent is not tested.
- **Dual-tree interaction.** Same-tree baselines control the branch's no-speculation main effect.
  They cannot identify a branch-by-DFlash2 interaction, because `draft-dflash` cannot be run on
  the master tree at all.
- **Energy magnitude is provisional.** Energy comes from sampled board-power telemetry and a
  separately measured prefill calibration. The audits support the direction; the exact percentage
  needs a hardware energy counter.
- **Residual order effects.** Phase A rotates arm order, but 5 passes do not close a 7-arm
  rotation, so each arm visits 5 of 7 positions, and prompt order is fixed in class blocks. The
  harness has `--latin-arms` and `--shuffle-prompts` (`bench.py:217`), but the committed primary
  matrix predates them and changing the design mid-study would have been worse.
- **Phase M's cost telemetry, and its exclusions.** The `mean_len` figure every cost quantity in
  Phase M rests on fails `cost_model.py`'s own integrity check on that phase, so no `k`, `c`,
  `k0` or fixed-cost decomposition is drawn from it. Its per-protocol series also drops 33
  records that stopped before the cap -- one prompt across every MoE arm, baseline included, so
  balanced rather than treatment-correlated, but still an exclusion decided by an outcome -- and
  the analyser now tabulates the intention-to-treat series beside the per-protocol one, and the
  two differ by at most 0.42 points across nineteen arms with every dense arm identical. The
  inclusion indicator is balanced across all eleven MoE arms; that is not a claim that missingness
  is independent of treatment, since early stopping is an outcome observed after treatment. What
  would settle the cost quantities is llama.cpp exposing `n_draft_verif_steps`, which
  `server_slot_stats` already holds and `to_json()` does not publish.
- **The quantization ladders are cross-session.** Phase Q's two rungs and Phase Q-small's four
  each ran in their own session. Pairing over prompts removes prompt difficulty; it does not
  remove hours-scale drift in entry temperature, clocks or host load, and the within-rung pass
  spread bounds only the minutes-scale part of that. The MTP head is also embedded in the target
  gguf and is quantized with it, so a rung changes verifier and drafter-head compute together.
  Stable acceptance shows the drafter's proposals did not change; it does not show its forward
  pass cost the same.
- **Follow-up hosts are not pooled.** The cross-host replications and the forced-warp
  intervention run on other machines, GPUs and toolchains. Their absolute throughput is
  host-specific and is never combined with Phase A's; only within-host contrasts are compared.
- **Selected settings are reported on the data that selected them.** The best `n-max` for each
  method is picked from the same ladder whose effect is then quoted. Fresh prompts, fresh passes
  or a second host are what would make that selection-independent.

## License

Original code in this repository is under the MIT License; see [`LICENSE`](LICENSE). The
measurement data this study produced -- `results/`, `analysis/`, `repro/` and the committed logs
-- is dedicated to the public domain under CC0; see [`LICENSE-DATA`](LICENSE-DATA). Upstream and
third-party material, including everything under `upstream/`, the copied source excerpts and the
patches, keeps its original license; see [`NOTICE`](NOTICE).

## Author

Hsiu-Chi Tsai | GitHub [`thc1006`](https://github.com/thc1006)
