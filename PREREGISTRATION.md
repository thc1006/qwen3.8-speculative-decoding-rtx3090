# Pre-registration

**Committed before any measurement was taken. This file is append-only; corrections go in a
dated ERRATA section at the bottom, never by editing the text above.**

Date: 2026-08-24
Author: Hsiu-Chi Tsai (`thc1006`)

## Why this document exists

The predecessor repo (`thc1006/qwen3.6-speculative-decoding-rtx3090`) had to issue a major scope
correction at v2.3: v2.2 claimed the negative finding was "hardware-class-independent", and a
clean A/B retest in a sibling repo falsified that. The correction was made honestly and in public,
but it was made *after* publication. Pre-registering the hypotheses and the analysis plan is the
cheapest available defence against repeating that.

## Hardware (third distinct RTX 3090 in this research line: disclose, do not pool)

| | v1 (s1) | v2/v3 (`3090` node) | **this repo** |
|---|---|---|---|
| host | 2x 3090, i7-11700, 62GB | 1x 3090 | 1x 3090, i9-13900, **31GB** |
| OS | Ubuntu 24.04 | Ubuntu 24.04 | **Debian 13** |
| driver | 580.126.09 | 580.126.09 | **610.43.02** |
| power cap | 350 W (stock) | 350 W (stock) | **420 W default; reset to stock - see below** |

The 420 W default is a different board SKU/vBIOS from the 350 W cards used in v1-v3. Absolute
tok/s are **not** comparable across these three hosts. Only within-host paired deltas are.

**Overclock, found and corrected mid-study (2026-08-24).** Ten minutes into the first full
Phase A run the card was discovered to be carrying `GPUMemoryTransferRateOffset=800`
(memory +400 MHz), `GPUGraphicsClockOffset=100`, and a 450 W limit against its 420 W default -
while this document described it as stock. **That run was discarded, not kept.** The offsets were
zeroed and the limit returned to 420 W, which restored the maximum memory clock to 9751 MHz, the
exact stock figure recorded in the predecessor repo's `BENCHMARK_ENV.md`. Full record in
`docs/GPU_AS_FOUND.md`.

This is not a cosmetic correction. Batch-1 decode is memory-bandwidth-bound while speculative
verification is comparatively compute-dense, so a memory overclock moves the baseline and the
speculative arms by *different* amounts - precisely the kind of differential that a paired
design is supposed to protect against. `harness/telemetry.overclock_state()` now records the
offsets in every result file and `bench.py` refuses to start on a non-stock card unless the
overclock is declared as an experimental factor.

## Primary hypotheses

**H1 (DFlash2 x Q4 target).** The predecessor repo's v3.0 concluded that a Q4 target collapses
DFlash because the drafter was trained against BF16/FP16 target hidden states. z-lab's own
published acceptance lengths for DFlash2 (BF16 5.28, Q8_0 5.13, Q4_K_M 5.39) point the other way.

- H1a: DFlash2 on a Q4_K_XL target on sm86 yields a *net speedup* over no-spec baseline.
- H1b: Drafter quantization (BF16 / Q8_0 / Q4_K_M) does **not** monotonically degrade acceptance
  against a Q4 target.
- **If H1a and H1b hold, the predecessor repo's v3.0 mechanism claim requires an erratum.**
  Pre-committing to issuing that erratum, whatever the result.

**H2 (state rollback).** Qwen3.8-27B is 48 linear-attention (Gated DeltaNet) + 16 full-attention
layers. On draft rejection, full-attention layers roll back by KV suffix truncation; GDN layers
must reconstruct recurrent state (cf. SpecLA, arXiv 2607.16673, which only evaluated GDN-1.3B on H100).

- H2a: The n-max optimum is pinned low (2-3) by rollback cost, not by acceptance decay alone.
- H2b: Therefore, holding acceptance fixed (via a `--spec-draft-p-min` gate) and raising n-max
  still degrades realized yield.
- Falsifier: if yield tracks acceptance monotonically once acceptance is controlled, H2 is wrong.

**H2' (competing explanation, from the PR #27342 author - must be discriminated, not assumed away).**
In the PR thread the author advances a different account of the same observation: the relative
cost of one extra decode step rises as the target gets more quantized (measured by them with
`llama-batched-bench` as BF16 6.7 %, Q8_0 14.5 %, Q4_K_M 23.4 %), because a 4-bit target is less
memory-bound, so the marginal compute of speculating further is proportionally more expensive.
They use this to explain why DFlash2 only narrowly beats MTP at 4-bit (1.447x vs 1.30x) while
leading clearly at BF16 (2.06x vs 1.58x).

This is a *quantization x arithmetic-intensity* explanation. H2 is a *layer-architecture*
explanation. They make different predictions and the experiment must separate them:

| | H2 (state rollback) | H2' (quantization/arithmetic intensity) |
|---|---|---|
| driver | 48 of 64 layers are GDN and cannot roll back by KV truncation | 4-bit target shifts the compute/bandwidth ratio |
| depends on target quant? | no | **yes, strongly** |
| depends on rejection rate? | **yes - cost is paid only on rejection** | no - cost is paid per drafted token regardless |
| prediction | at fixed acceptance, raising n-max still degrades yield | at fixed n-max, yield improves as the target gets *less* quantized |

Discriminating test, pre-committed: hold acceptance approximately fixed with the
`--spec-draft-p-min` gate and sweep n-max. H2 predicts continued degradation; H2' predicts the
degradation largely flattens because the drafted-token count, not the rejection count, is what
H2' charges for. **If the data favour H2', this repo reports H2' and says so.** H2 is not
this repo's finding to defend.

**Prior art note on drafter quantization (checked 2026-08-24, before measurement).** The same PR
thread already reports, on a 32 GB card, that Q4_K_M / Q8_0 / BF16 drafters produced
byte-identical output with identical acceptance counts, and that BF16 is actively worse there
because it crosses the VRAM ceiling. H1b is therefore **not novel as a question**; what is
untested is the 24 GB case, where a 17.5 GB target leaves far less headroom than a 32 GB card
does, and where that same commenter's VRAM argument predicts a larger penalty.

**H3 (dense-vs-MoE contradiction).** The public record disagrees:
njannasch.dev on a 5060 Ti 16GB reports Qwen3.6 **dense +MTP = 42% slower** while the MoE gains
1.47x; sudoingX on an RTX 3090 24GB reports Qwen3.8 **dense +MTP = +54% to +81%**.

- H3a: The sign flip is explained by llama.cpp build age (presence of the `ssm_scan` state-rollback
  commit), not by model generation or VRAM headroom.
- Competing explanations to be separated, not assumed: 3.6-vs-3.8, 16GB-vs-24GB, quant.

**H4 (the predecessor's central claim).** The 2026-04 finding was "no llama.cpp speculative-decode
configuration is a net win for Qwen3.6-35B-A3B on a consumer 3090", explained by MoE
expert-saturation. Qwen3.8-27B is dense-hybrid, so that mechanism does not apply.

- H4a: Under a matched protocol, dense-hybrid shows net *positive* yield where the A3B MoE showed
  net loss - isolating MoE routing, not consumer Ampere, as the cause.

## Analysis plan (fixed in advance)

- **Design:** paired, arms **interleaved within a session**, not blocked (all-A-then-all-B invites drift).
- **N:** >= 5 complete passes per arm. Prompt set fixed before measurement, and includes both the
  code/structured and prose classes, because every prior study on this model family finds the
  effect splits sharply by prompt class (code gains, prose often loses).
- **Statistic:** paired bootstrap CI on the per-prompt delta + effect size. Not mean+-std alone.
  A result whose CI crosses zero is reported as "no detected effect", never as a directional claim.
- **Primary endpoint:** decode tok/s as reported by llama-server (`predicted_per_second`),
  which includes draft and verify time in the denominator. This is the user-visible metric.
- **Secondary:** acceptance rate/length, tok/J, peak VRAM, TTFT.
- **Guards:** refuse to start when the port is bound and the owning PID is not ours (a contributor
  to sudoingX/qwen38-mtp published three fabricated-looking rows to a zombie server that kept
  answering `/health`); assert weights are GPU-resident; both arms at `--parallel 1`.

## Two design decisions fixed in advance

**Dual-tree baseline (build confound).** DFlash2 requires llama.cpp PR #27342, which is not
merged. Every other arm runs on master. Comparing a DFlash2 arm on the PR binary against a
baseline on the master binary would conflate the *method* with the *branch*: any kernel change
between the two trees lands entirely on the DFlash2 arm. This is not hypothetical -- a
contributor to `sudoingX/qwen38-mtp` disclosed exactly this confound in their own KV-cache
comparison ("part of the +47% is almost certainly the newer build's improved CUDA kernels").

Therefore **a no-spec baseline arm is run on BOTH trees**, `baseline@master` and
`baseline@pr27342`, with identical flags. The DFlash2 effect is estimated against
`baseline@pr27342`. The difference between the two baselines is reported as its own quantity;
if it is not within noise, every cross-tree comparison in this repo is reported with that
offset stated, not silently absorbed.

**Multiplicity.** The design produces roughly 8 arms x 5 classes of intervals. At the 95 % level
some will exclude zero by chance alone. Fixed in advance:

- **Primary endpoints** (confirmatory): the overall class-stratified paired effect of each
  speculative arm against its own-tree baseline. One interval per arm.
- **Secondary** (exploratory, explicitly labelled as such): all per-class effects, the n-max
  sweeps, and the drafter-quant ladder. These are for direction and mechanism, and no claim in
  this repo rests on a per-class interval alone.
- No arm is added to the matrix after seeing results in order to chase a significant one.

## Declared in advance as NOT covered

Multi-GPU / tensor-parallel. Non-CUDA backends. Training or fine-tuning drafters. Batch serving
beyond the explicit concurrency sweep. Quality benchmarking beyond the losslessness check
(no MMLU/GSM8K claims). Absolute cross-host comparison against v1-v3 of the predecessor repo.

## Known prior art (established by a search sweep on 2026-08-24, before measurement)

Claims of novelty in this repo are scoped against these. Anything already covered below will be
cited, not re-claimed.

- `sudoingX/qwen38-mtp`: llama.cpp `draft-mtp` on Qwen3.8-27B across 40+ GPUs incl. 6 RTX 3090 rows,
  plus DSpark on a 3090. Crowdsourced; heterogeneous builds/quants/OS; self-disclosed confounds.
- `syv-ai/qwen38-27b-rtx3090`: single 3090, vLLM + custom patches, MTP/DFlash2/lookup.
  Explicitly does not cover llama.cpp comparison, confidence intervals, or power efficiency.
- `tfriedel/qwen3.6-rtx3090-lab`: 4x 3090, Qwen3.6 dense + MoE + Qwen3.8-27B day-2. No formal statistics.
- `zptalk0221-cpu/llama-cpp-dflash2-qwen3.8-windows`: DFlash2 on RTX 4090 48GB, Q6_K target,
  ~70 vs ~60 tok/s. User-reported; no trial count; no acceptance data.
- `ianlpaterson` (3090 Ti, Qwen3.6-27B): DFlash **v1** vs MTP, N=10+, and the explicit finding that
  spec decode is not bit-for-bit lossless at temp=0 on free-form prose.
- `njannasch.dev` (5060 Ti): Qwen3.6 dense vs MoE; dense +MTP measured as 42% *slower*.
- KGP Talkie (RTX 5090): n-max sweep, KV sweep, and a `reasoning_effort` x acceptance ladder.
- `shreyansh26/qwen-spec-decode-benchmarking`: Qwen3.6-35B-A3B on B200, SGLang/vLLM.
- arXiv 2607.17283 "Lossless but Not Free": losslessness verified at three levels, but on
  Apple M3 + Qwen2.5 + classic two-model SD; the CUDA pairing was scripted, not executed.
- arXiv 2607.16673 "SpecLA": linear-attention state-rollback theory; H100 + GDN-1.3B only.
- **llama.cpp PR #27342 comment thread itself (60 comments, checked 2026-08-24)**: this is the
  closest prior art and it invalidates a novelty claim made earlier in this repo's planning.
  Two contributors have ALREADY posted single-RTX-3090 DFlash2 datapoints on a Q4 target:
  * `treo`: single RTX 3090, `Qwen3.8-27B-UD-Q4_K_XL` (the same file this repo uses), 32K ctx,
    q8_0 KV. MTP `n-max 3` overall 57.79 tok/s at accept_rate 0.6317 across 11 categories x 2
    samples; concludes DFlash2 "seems to be not significantly better than MTP".
  * `ouening`: RTX 3090 24G, Windows 11, `UD-Q4_K_M`, 128K ctx, q8_0 KV, baseline vs MTP
    n-max 2-5 vs DFlash2 n-max 2-5. Results posted as screenshots.
  **"First public RTX 3090 + DFlash2 + Q4 datapoint" is therefore FALSE and is not claimed.**
  What remains uncovered against these two: both are N=1 with 2 samples per category, neither
  publishes an interval or a paired design, neither runs a drafter-quantization ladder, and
  neither checks losslessness, output degeneracy, or energy. This repo's contribution is the
  controlled protocol and those missing axes -- not priority.
- llama.cpp issue #27623 (opened 2026-08-23, **zero comments**): decode collapses ~25x past
  ~80K context on this model while prompt processing stays fast; reported on RTX 4080 SUPER
  (sm89) across three quants. No reproduction on another architecture, and no one has tested
  how speculative decoding interacts with the cliff.
- llama.cpp: PR #22673 (MTP, merged), PR #23269 (MTP cleanup + recurrent-rollback fix),
  PR #27342 (DFlash2, **open**), issue #22947 (llama-bench spec-decode support - closed as not planned),
  issue #19712 (spec decode blocked with `--mmproj`), issue #27623 (~25x decode collapse past ~80K
  on this hybrid model, **open**), issue #27122 (MTP + `--split-mode tensor` CUDA lockup, open).
- vLLM: issue #52475 (MTP repetition collapse with turboquant KV on sm120, open),
  issue #52682 (Qwen3.8-27B-FP8 CUDA-graph capture hang on Ampere, open).

## INTERIM FINDINGS (appended during execution; the text above is never rewritten)

### 2026-08-24: H2 is in trouble, on this repo's own pre-registered falsifier

Status: **preliminary, 2 of 5 passes of Phase A.** Recorded now because it is a pre-registered
hypothesis moving against the person who registered it, and that should be on the record before
the remaining passes rather than after.

Phase A yields a cost model that was not anticipated when this document was written:

    speedup = mean_len / k,   where  mean_len = 1 + n_max * acceptance   (verified against
    llama.cpp's own per-request `mean len` field, exact to two decimals)

so `k` - the cost of one speculative verification step in units of a plain decode step - is
recoverable per request. Measured:

| arm | verification width | k | k spread across 5 prompt classes |
|---|---:|---:|---:|
| mtp-n2 | 3 | 1.4494 | 1.06 % |
| mtp-n3 | 4 | 1.7420 | 1.15 % |
| dflash2-n4 | 5 | 1.8871 | 1.79 % |
| mtp-n5 | 6 | 2.2929 | 1.38 % |
| dflash2-n7 | 8 | 2.7142 | 2.39 % |

(Figures corrected 2026-08-24 after a review of the estimator itself - see "Estimator
correction" at the end of this entry. The earlier numbers used llama.cpp's reported `mean len`
field and were biased by a prompt-dependent amount.)

**Test of H2.** A state-rollback account charges the overhead to *rejection*. Modelling that as
`k = k_verify + r * n_max * (1 - acceptance)` makes `r` estimable from the slope of k against
acceptance. Acceptance spans 0.096-0.918 in this data - nearly a ten-fold range - and every arm
returns `|r| <= 0.0024` with r^2 between 0.001 and 0.065 - no relationship at all. No
rejection-proportional cost is detectable on either drafter.

**This does not say rollback is free.** It bounds how much of the measured overhead rollback can
account for, and the bound is approximately nothing. H2 as stated - rollback as the dominant term
setting the n-max ceiling - is not supported.

**What replaces it.** Fitting `k = k0 + c * (w - 1)`:

- `draft-mtp`, widths 3/4/6: k0 = 0.8937, **c = 0.2803**, r^2 = 0.9998
- `draft-dflash`, widths 5/8: k0 = 0.7844, **c = 0.2757** (two points only: a straight line
  through two points is perfect by construction and this r^2 carries no information; Phase N adds
  widths 3/5/7/9 so that `c` is fitted rather than assumed)

Two unrelated drafters agreeing on `c` to 1.7 % while differing in `k0` by 14 % places the
marginal cost on the verification path, not on the drafter. Notably DFlash2's fixed cost `k0` is *lower* than
the target's own built-in nextn head.

**Consequence for Phase R.** The question is no longer "H2 or H2'" but "which resource sets
`c`". That is a sharper test on a single measured coefficient: H2' (arithmetic intensity) implies
`c` falls as the compute budget rises and is insensitive to memory clock; a memory-bound account
implies the reverse. Phase R is unchanged in design; its estimand is now `c` not a
throughput elasticity.

**Estimator correction (same day, before the phase completed).** The first version of this
analysis took `mean_len` from llama.cpp's own per-request `mean len` field. That field reports
`1 + n_max * accept_rate`, which assumes every step drafts the full `n_max`; steps that draft
fewer make it an overestimate. Measured against the physical definition

    forwards F = predicted_n - accepted        (each forward emits its own token plus its accepted drafts)
    mean_len   = predicted_n / F

the field is high by +0.17 % to +0.81 %, **and the error varies by prompt** - the same order as
the cross-class constancy of `k` that this entry uses as evidence. In other words the estimator
was contaminating the exact quantity the claim rests on.

An anomaly noted in the first version - that k sloped slightly *positive* against acceptance on
every arm - turned out to be that bias, not a physical effect: the field's error is largest on
high-acceptance prompts. Recomputing from the API counters removes it. The slopes now straddle
zero (-0.0167 to +0.0033) with r^2 between 0.001 and 0.065, the within-arm spread of k tightens
(e.g. mtp-n2 1.36 % -> 1.06 %), and the conclusion is unchanged but better supported.

Deriving from the API counters also removed an unverified assumption: the earlier version aligned
log lines to prompts by position. The log is now used only as an independent cross-check, and it
agrees with the API counters on **150/150 requests, to the token**.

### 2026-08-24: Phase A complete (875/875, 0 incidents), and two corrections

**Result.** Every speculative arm is faster than its own-tree baseline, all five intervals
excluding zero, with within-prompt run-to-run CV at or below 0.3 %:

| arm | width | tok/s | vs own-tree baseline (95 % CI) | k | tok/J | J/request |
|---|---:|---:|---|---:|---:|---:|
| baseline@master | - | 41.55 | - | - | 0.1005 | 3980 |
| baseline@pr27342 | - | 41.55 | - | - | 0.1005 | 3979 |
| mtp-n2 | 3 | 66.39 | **+59.77 [+56.95, +62.75] %** | 1.4497 | 0.1627 | **2503 (-37.1 %)** |
| mtp-n3 | 4 | 63.29 | +52.32 [+48.47, +56.48] % | 1.7425 | 0.1549 | 2684 |
| dflash2-n4 | 5 | 63.13 | +51.94 [+45.56, +58.17] % | 1.8874 | 0.1554 | 2835 |
| mtp-n5 | 6 | 54.89 | +32.10 [+26.38, +37.75] % | 2.2939 | 0.1343 | 3228 |
| dflash2-n7 | 8 | 50.95 | +22.63 [+14.68, +30.37] % | 2.7156 | 0.1251 | 3786 |

The dual-tree control is exact: the two baselines agree to 41.55 tok/s and produce **byte-identical
output on 125/125 prompt-passes**, so nothing in the DFlash2 comparison is attributable to the
unmerged branch.

**H4a is supported, and the class breakdown matters more than the headline.** `dflash2-n7` is
+22.6 % overall while being a *net loss* on three of five prompt classes (prose -11.1 %,
chat -4.3 %, zh -28.7 %). `mtp-n5` and `dflash2-n4` are within noise of zero on Chinese. A single
overall figure conceals a sign change - the same failure this repo documented in the predecessor's
headline (`docs/METHODOLOGY_AUDIT.md` A1), reproduced here in the opposite direction.

**Cost model, final.** `draft-mtp` (widths 3/4/6): k0 = 0.8934, **c = 0.2806**, r^2 = 0.9998.
`draft-dflash` (widths 5/8, two points): k0 = 0.7831, **c = 0.2761**. `c` agrees to 1.6 % between
unrelated drafters while `k0` differs by 14 %. Within-arm spread of `k` across five prompt classes
and five passes is 0.35-0.54 %. Independent cross-check: the API counters and llama.cpp's own log
lines agree on **625/625 requests, to the token**.

**H2 is not supported.** Over an acceptance range of 0.096-0.918, the rejection-proportional cost
`r` is at most +0.0028 decode-steps per rejected draft token, with r^2 between 0.001 and 0.060.
The overhead is charged per position verified, not per draft rejected. This bounds rollback's
contribution; it does not prove rollback is free.

**Losslessness.** Speculative arms are byte-identical to baseline on only 25-30 of 125
prompt-passes - 76-80 % diverge, forking at a median 23 % into the text - but every arm is
**100/100 reproducible across passes**. The divergence is deterministic, not noise. This is
consistent with, and corroborative of, llama.cpp #25618 rather than novel.

---

### CORRECTION 1: the baseline bandwidth elasticity is 0.75, not ~1.0

An earlier entry inferred a bandwidth elasticity of about 1.0 for the no-spec baseline from the
overclock removal: memory +400 MHz was removed and throughput fell 4.1 %. That step was **not a
bandwidth-only lever** - it also removed +100 MHz of core offset and dropped the power limit from
450 W to 420 W. Attributing the whole 4.1 % to bandwidth was wrong.

Phase R's pre-flight measures it properly, moving memory clock alone at a fixed 420 W:

| condition | memory clock under load | tok/s |
|---|---:|---:|
| bw-lo (-800 offset) | 9101 MHz | 40.52 |
| stock | 9501 MHz | 41.87 |
| bw-hi (+800 offset) | 9901 MHz | 43.15 |

That is +4.2 % / -4.2 % of memory clock for +3.1 % / -3.2 % of throughput: **elasticity ~ 0.75**.

A further subtlety, recorded because it limits even the controlled lever: at a fixed power cap,
raising the memory clock takes power from the core. SM clock under load reads 1922 MHz at stock
against 1886 and 1881 MHz in the two bandwidth arms, so the bandwidth arms are a net of "+4 %
memory, -2 % core" not a pure bandwidth step. Phase R's full run quantifies this; the
pre-flight is a single 200-token measurement per condition.

### CORRECTION 2: the Phase A process crashed after writing its last record

The run wrote all 875 records and then died with a glibc `double free or corruption (out)`,
before attaching pass 5's baseline comparisons and before releasing the run lock. The chain's
completeness gate refused to start Phase R as a result, which is the intended behaviour.

Root cause is in this harness, not in llama.cpp: `server.start()` used
`preexec_fn=os.setsid`, which runs Python code in the child after fork and is documented as
unsafe in a process with threads. This harness runs a GPU power-sampling thread throughout, and
the run performed roughly 7 900 fork/exec cycles. Replaced with `start_new_session=True`, which
performs the same `setsid` on the safe side of the fork; `os.killpg` still works, verified.

Pass 5's comparisons were **recomputed from the recorded text** using the harness's own
`_attach_baseline_comparisons`, with no measurement repeated or altered. The pre-repair file is
kept as `results/phase_a.pre_repair.json` and the repair is logged in the result's `repairs` field.

## ERRATA

_(none yet - the two items above are corrections to this document's own interim reasoning, not to
a published result. The v3.0 erratum for the predecessor repo remains pending on Phase C's
drafter-quantization ladder.)_

## ADDENDUM, registered 2026-08-24: hypotheses for Phase L

Registered after Phase A and during Phase R2, and before any Phase L measurement exists. It is
appended rather than folded into the hypothesis section above because that section was fixed
before the study began and does not get rewritten. What makes this a pre-registration and not a
rationalisation is the ordering: the depth ladder had not been run when this was written, and
the commit that adds it (`8159e60`) contains the harness and no results.

The occasion is llama.cpp issue #27623, opened 2026-08-23 with no replies. It reports decode
throughput on this exact model collapsing from 33 tok/s at 68K to 1.4 tok/s at 91K, roughly 25x,
while prompt processing stays at about 1300 tok/s. The reporter is on an RTX 4080 SUPER, sm_89,
and reproduced it across three quantisations. Two things follow that nobody has done: it has not
been tried on another architecture, and nobody has asked what speculation does to it.

**H5 (the cliff is not Ada-specific).** The collapse reproduces on sm_86 at a ratio of 10x or
more between the best and worst rung of the ladder.

- Falsified if the worst-to-best ratio is under 3x. That outcome is worth as much as a
  reproduction, since it localises the report to Ada and is a more useful comment on the issue
  than another confirmation would be.
- A gradual slope is not a cliff. The largest single-rung drop is reported alongside the overall
  ratio, and a finding is only called a cliff if one rung transition carries most of it.

**H5a (speculation helps more as context deepens).** The relative speedup over the matched
baseline increases with depth, monotonically over the rungs that clear H5's cliff.

The reasoning is the mechanism this study has already measured at shallow context, extended:
speculation converts bandwidth-bound decode into compute-bound verification, measured here as a
bandwidth elasticity of 0.14 to 0.21x against a compute elasticity of 1.71 to 1.74x. Attention
over a long KV cache is bandwidth work that grows with depth. A baseline pays one pass over the
cache per token emitted. A verification step covering k accepted tokens pays one pass for all k.
So the deeper the cache, the more a verification step saves relative to the baseline, and the
advantage should widen rather than hold.

- Falsified if the speedup is flat in depth (slope indistinguishable from zero across rungs) or
  shrinks. Either would say the KV pass is not what limits decode at depth here, and the
  mechanism claim would need narrowing to shallow context.
- This is the prediction most likely to fail, because it assumes the cliff has the same cause as
  ordinary depth slowdown. If the cliff is a distinct pathology, for instance an allocation or
  eviction problem rather than a bandwidth one, speculation has no reason to help and H5a can
  fail while H5 holds. The two are reported separately for that reason.

**H5b (acceptance is roughly depth-invariant).** Mean accepted tokens per verification step
varies by less than 15% across the ladder for a given method.

- If acceptance holds while throughput collapses, the loss is in the target and speculation is
  inheriting a problem rather than causing one.
- If acceptance falls with depth, the drafter is degrading on its own, which is a different
  finding and is what DFlash2's long-context claim would be failing at.

**What Phase L cannot settle.** The ladder stops at 96K because 128K with a q8_0 KV cache needs
23.8GB of a 24GB card. The reporter's cliff sits near 80K and 96K clears it, but the behaviour
beyond 96K on this card is not measured and will not be claimed. Deeper rungs need either q4_0
KV, which would confound depth with KV precision, or the 48GB card, which is covered in
`docs/A6000_PLAN.md`. The ladder is also 15 prompts rather than 25, three per class, which is
enough for the cluster bootstrap to resample within each class but gives wider intervals than
Phase A's.

## ADDENDUM, registered 2026-08-24: hypotheses for Phase M

Registered while Phase R2 runs and before the MoE target has been loaded once. The weights
finished downloading at 20:43 and no server has been started against them.

H3 and H4a above already frame the dense-versus-MoE question. What follows is narrower, and it
exists because reading the predecessor's own explanation back against its own arm list turned up
a gap that neither this document nor that study had noticed.

The predecessor measured draft-then-verify at K of 8, 16, 32 and 64, found a net loss at every
one, and explained it as MoE expert loading: a verification pass over K draft positions loads the
union of those positions' expert sets, and below the K at which that union saturates, the pass
buys fewer decode steps than it pays for in expert traffic. Acceptance was near 100 % throughout,
so drafting quality was not the problem.

That explanation is monotone in K, and its own logic points below the range that was measured.
MTP is self-speculation with K equal to n_max, so n_max of 1 or 2 sits far under the smallest
arm the predecessor ran. Its sibling vLLM study is the indirect signal: MTP at k=1 on the same
3090 came out 27.5 % faster, credited to vLLM's smaller K and lighter verify path. Nobody has
asked whether llama.cpp's MTP path escapes the penalty for the same structural reason, and the
GGUF carries the `blk.40.nextn.*` tensors needed to ask.

**H6 (small K escapes the MoE penalty).** On the MoE target, net yield rises monotonically as K
falls across the combined ladder, and at least one MTP arm at n_max of 1 or 2 is not a net loss,
meaning its interval's upper bound reaches zero or above.

- Falsified if every MTP arm loses by a margin comparable to the draft-then-verify arms. That
  would say the penalty is not a function of K, and the expert-union account the predecessor gave
  would need replacing rather than extending.
- Also falsified, differently, if yield is not monotone in K. A non-monotone curve would mean
  something other than expert-set size is moving, and the arms at n_max 4 to 7 are placed to
  overlap the draft-then-verify ladder specifically so the two paths can be compared at the same
  K rather than assumed to be different regimes.

**H6a (the anchor gates everything else).** `moe-draft08b-n8` reproduces the predecessor's loss:
its net yield against the MoE baseline is worse than -25 %.

- This runs first in the report for a reason. If the anchor does not reproduce, the difference
  between the two studies is in the harness, the build, or the card, and no Phase M result may be
  read as a statement about MoE until that is resolved. The predecessor's figures were 138.9 tok/s
  baseline against 77.0 tok/s at K=8, a 44.6 % loss, taken at 16384 context on a different 3090.
  This runs at 8192, so exact agreement is not expected and is not the test; the sign and the
  rough magnitude are.

**H6b (the cost model localises the difference).** Fitting k(w) = k0 + c(w-1) on the MoE gives a
marginal cost per verified position c that exceeds the dense model's, by roughly the factor
separating their net yields.

- This is the version of the expert-loading claim that produces a number instead of a narrative.
  The predecessor asserted the mechanism from acceptance being high while throughput fell; c
  measures the cost of a verified position directly.
- Falsified if c is comparable between the two models. In that case the MoE's loss lives in k0,
  the fixed per-step cost, and the story is about step overhead rather than about how much each
  extra speculated position costs.

**What Phase M cannot settle.** The two models differ in size as well as in routing, 21.28 GiB
against 16.35 GiB, so any yield difference carries a size component that this design cannot
separate. Both being hybrid attention at `full_attention_interval: 4`, both carrying MTP tensors,
and both sharing the 248320-token vocabulary removes the confounds that can be removed here; size
is not one of them. Nothing in this phase will be reported as isolating routing alone.

## Correction 3, 2026-08-24: the derived mean length was low by about a percent

Found by checking the harness against the server rather than by anything going wrong. The numbers
it produced were plausible, internally consistent, and stable across five passes.

`speedup = mean_len / k` is the model this study argues from, and `mean_len` has no API field, so
it is derived from the per-request counters. The derivation was

    predicted_n = accepted + F   =>   F = predicted_n - accepted,   mean_len = predicted_n / F

reasoning that each target forward pass emits one token of its own plus whatever drafts it
accepted. The first generated token does not come from a decode forward. It comes out of the
prompt-processing pass, so the decode phase emits `predicted_n - 1` tokens, not `predicted_n`:

    F = predicted_n - accepted - 1,   mean_len = (predicted_n - 1) / F

Compared against the `mean len` the server prints for every request, across all 625 speculative
requests of Phase A, the old form is systematically low: mean gap -0.0204, and the sign never
changes. The corrected form sits at -0.0050, which is inside the log's own `%5.2f` printing
precision.

**Why the existing integrity check did not catch it.** `cross_check_against_log` compared the API
counters against the log's counters and reported 0 mismatches out of 625, correctly, both before
and after this correction. The counters were never wrong. What was wrong was the arithmetic
applied to them, and nothing compared the derived quantity against the server's own value of the
same quantity. That comparison is now part of the check, and it was verified to fire on the old
formula and pass on the new one.

**What it moves.** `k` rises by 0.33 % at n-max 2 to 0.59 % at n-max 7. Because the bias grows
with depth it inflates the fitted `c` by about 0.8 %: `draft-mtp` k0 0.8934 -> 0.8937 and
c 0.2806 -> 0.2829; `draft-dflash` k0 0.7831 -> 0.7825 and c 0.2761 -> 0.2784. Both `c` still read
0.28 to two figures, the 1.6 % agreement between the two methods survives, the implied optima
stay at n-max 2 for MTP and 4 for DFlash2, and no hypothesis verdict changes. Every figure in
this document above this line was computed with the old form and is superseded by the values here
rather than rewritten in place.

**What is still approximate.** The corrected form reproduces the server's printed value on about
70 % of requests. The other 30 % need `F` one smaller still, which is what truncation at the
token cap looks like: a verification step that ran and was counted, whose accepted tokens were
partly discarded because the request had reached `max_tokens`. The API cannot distinguish the two
cases, so the reported `mean_len` remains low by under 1 %, stated rather than hidden.

**Consequence for the upstream contribution.** `docs/UPSTREAM_CONTRIBUTIONS.md` argued for
exposing `n_draft_verif_steps` per request on the grounds of convenience, since an exact identity
appeared to exist. There is no exact identity. The counter the server already keeps is the only
way to get the number right, and a derivation that looks exact while being quietly wrong by a
percent is a better argument for the patch than the one that was there.

**No measurement was repeated.** The correction is arithmetic applied to counters that were
already recorded, so every result file is unchanged and only the analysis output moves.

## Declared during Phase R2, 2026-08-24: the top pin does not bind

Recorded while Phase R2 is running, at 250 of 1575 records, before any of its results are
interpreted. It is a limitation of the design, found by checking the run against its own premise
rather than by anything failing.

Phase R2 exists because a power cap is a poor compute lever: the clock it produces is an outcome,
it differs between methods, and it moves during an arm. Pinning with `nvidia-smi -lgc` was meant
to make the clock a setting. Measured on the first pass:

| condition | between methods | within an arm | power drawn |
|---|---|---|---|
| `sm600` | 0.00 % | 0.00 % | 182 to 207 W |
| `sm1200` | 0.00 % | 0.00 % | 248 to 278 W |
| `sm1700` | 0.89 % | up to 2.72 % | 381 to 401 W, peaks at 420 |

At 600 and 1200 MHz the pin holds exactly: mean equals min on every record, so every power sample
sat on the requested clock, and all three methods met the same clock. `sm600 -> sm1200` is
therefore a matched two-fold compute sweep, which is what H2' needs and what Phase R never had.

At 1700 MHz the pin does not bind, because the 420 W power limit binds first. The baseline draws
381 W and holds 1710 MHz; `mtp-n3` draws 401 W with peaks at exactly 420 and falls to 1650. So the
top condition reverts to being an outcome, and the methods land 0.89 % apart. Phase R's
corresponding figures were 30.0 % and 35.8 % apart with within-arm drift of 71.6 % and 118.9 %, so
this is a 34-fold improvement rather than a failure, but it is not zero and elasticities that
cross `sm1700` carry it. The analysis reports matching per condition for that reason.

**Not corrected mid-run.** Raising the power limit for the top condition alone would vary power
and clock together, which is the confound the phase was built to remove. Lowering the top pin
would need a restart and would cost the range. The clean interval already exists at
`sm600 -> sm1200`, so the top point is kept for range and read with its caveat.

**A side observation that supports the mechanism.** At the same pinned 1700 MHz, `mtp-n3` draws
401 W against the baseline's 381 W. Same clock, more power, so more work per unit time. That is a
fourth independent line of evidence for speculation converting a bandwidth-bound decode into a
compute-bound verify, alongside the bandwidth elasticity ratio, the compute elasticity ratio, and
the higher clock reached under a shared power cap. It was not predicted in advance and is
recorded as an observation rather than as a test.

## ADDENDUM, registered 2026-08-24: a mechanism for the width partition, and what it predicts

Registered before `phase_nmax` has produced a single record. It is queued behind Phase C and its
result file does not exist.

Phase A found that fork positions partition the speculative arms into exactly two groups by
verification width, `{3, 4}` against `{5, 6, 8}`, identically in all five passes, with the
partition shared by the target's own MTP head and by a structurally unrelated 1.1 GB
block-diffusion drafter. That was reported as an observation without a mechanism.

Reading `ggml/src/ggml-cuda/mmvq.cu` supplies one. `calc_nwarps` selects how many warps take part
in the MMVQ reduction, from a table chosen by architecture. An RTX 3090 is sm_86: it is not RDNA,
GCN, CDNA or DGX Spark, and the Turing table is gated on `arch >= TURING && arch < AMPERE`, so it
falls through to `MMVQ_PARAMETERS_GENERIC`. That table reads

    ncols_dst 1 to 4  -> 4 warps
    ncols_dst 5 to 8  -> 2 warps
    ncols_dst > 8     -> 1 warp

`ncols_dst` is the verification width. The number of warps sets the shape of the summation tree,
so it sets the order in which partial products are added, and floating-point addition is not
associative. The boundary between 4 and 5 is exactly where the measured partition falls.

This is the same shape of cause the maintainers traced on Vulkan, where the `soft_max` reduction
order changes with batch size, and it is on the side that is still open: ggerganov asked on
2026-08-21 for a reproduction isolating the width at which it starts on CUDA, and noted that
nobody had posted the boundary.

**H8 (the partition follows `calc_nwarps`).** `phase_nmax` runs MTP at n-max 1 through 8, so
widths 2 through 9. Fork positions will partition into exactly three groups: `{2, 3, 4}`,
`{5, 6, 7, 8}`, and `{9}` alone.

- Width 2 joins the low group and width 7 joins the middle one. Both are new: Phase A measured
  neither, and both are forced by the table.
- Width 9 separates for a second reason as well, and the two cannot be told apart here.
  `calc_nwarps` drops it to one warp, and `ggml_cuda_should_use_mmvq` returns
  `ne11 <= MMVQ_MAX_BATCH_SIZE`, which is 8 on this fallthrough, so width 9 leaves the MMVQ
  kernel family altogether. Either alone predicts a separate group.
- Falsified if width 2 or width 7 lands in the wrong group. That would say the partition tracks
  something other than this table, and the mechanism would be wrong even though the Phase A
  boundary happens to coincide with it.
- Falsified differently if width 9 joins `{5, 6, 7, 8}`, which would say the reduction shape is
  not what moves the fork positions.

**What this does not establish.** That the warp count is the cause rather than a correlate. The
table changes at the same width as the measurement, on this architecture, for this quantisation,
which is consistent with cause and is not proof of it. Proof needs the kernel forced to a fixed
warp count across widths and the divergence disappearing, which is a build change and belongs
upstream rather than in a study that must not rebuild its own trees mid-run. The claim registered
here is the boundary and its coincidence with a named code path, not the causal step.

**Also unestablished: that it matters.** Divergence is not error. Every arm in Phase A was
deterministic and reproduced exactly across five passes, and nothing here says the speculative
output is worse, only that it is different. The reason it is worth reporting is that greedy
decoding is documented as reproducible, and a user comparing two runs has no way to know a
verification width changed the arithmetic under them.

## Registered 2026-08-24, after posting to llama.cpp #25618 and before `phase_nmax` runs

The comment posted to that thread says the onset width is not something this study has, because
Phase A never ran `--spec-draft-n-max 1`. `phase_nmax` does run it, as `mtp-n1`, which is
verification width 2. So the answer arrives in a few hours and is worth committing to first.

Width 2 is exactly where the Vulkan onset sits. `frizikk` traced it to a `MUL_MAT` that takes
`mul_mat_vec_q8_0_f32_f32` at `N=1` and `quantize_q8_1_x4` plus MMVQ at `N=2`, and `Ankk98` said
the same thing in the opening comments from behaviour alone: `n_max=1` was fine and `n_max>=2` was
not. Both are Vulkan.

**H8a (the onset is the same on CUDA).** `mtp-n1`, at width 2, diverges from the non-speculative
baseline on at least one prompt.

- If it does, CUDA and Vulkan agree on where divergence begins, and the `{2,3,4}` group of H8 is
  a grouping among widths that all already diverge.
- If `mtp-n1` is byte-identical on all 25 prompts, CUDA's onset is above Vulkan's. That would be
  a real backend difference and more interesting than the grouping, and it would need saying in
  the thread promptly, since the comment already there implies the onset is settled.
- Registered as a genuine question. Phase A's shallowest arm was width 3 and it diverged on 21 of
  25 prompts, which says nothing about width 2 either way.

**A control that arrives for free.** `phase_nmax` runs both an MTP and a DFlash2 arm at widths 3,
5, 7 and 9. `snick525` established drafter independence at one width on Vulkan; this repeats it at
four widths on CUDA without costing an extra arm. If the two drafters ever disagree at the same
width, the width-grouping account fails and H8 goes with it.

## Disclosure, 2026-08-24: an external process ran during part of Phase R2

Noticed while checking process health, not because anything looked wrong. Recorded because a
study that refuses to run on an overclocked card should not quietly ignore a second process
touching the same GPU.

A `watch -n 1 nvidia-smi` was started from a VS Code terminal at 21:18:37, forty-three minutes
into a run that began at 20:35:27. It was still running at the time of writing. So pass 1's
`sm600` arms and the first two `sm1200` arms were measured without it, and everything after
`mtp-n3@sm1200` was measured with it.

Checked rather than argued away. The sensitive number is the no-speculation baseline's
coefficient of variation across prompts, which is small enough to see a perturbation:

| arm | watch running | tok/s | CV | SM clock mean/min |
|---|---|---|---|---|
| `baseline@sm1200` | no | 37.580 | 0.073 % | 1200 / 1200 |
| `baseline@sm600` | no | 21.497 | 0.115 % | 600 / 600 |
| `baseline@sm1200-bwlo` | yes | 36.676 | 0.067 % | 1200 / 1200 |
| `baseline@sm1200-bwhi` | yes | 38.306 | 0.075 % | 1200 / 1200 |
| `baseline@sm1700` | yes | 41.268 | 0.069 % | 1710 / 1710 |
| `baseline@sm1700-bwlo` | yes | 39.856 | 0.081 % | 1710 / 1710 |

The variation with the poller running is the same as without it, or slightly smaller, and every
record on both sides reports SM clock mean equal to min, so the pin held throughout. A one-hertz
read-only driver query costing tens of milliseconds of CPU on a 24-thread host is not visible in
a GPU-bound decode loop.

**Left running rather than killed.** It is demonstrably not perturbing anything, and stopping it
now would introduce a change partway through a run for no measurable gain. The uneven coverage
across pass 1 is disclosed here rather than corrected.

**What would have changed the decision.** A baseline CV rising above the 0.24 % that Phase A saw
within a prompt, any record where SM clock mean exceeded min at a pinned condition, or a step in
a baseline arm's per-prompt series. None of those are present.

## Registered 2026-08-24: the competing explanation for the width grouping, and what separates them

The obvious objection to H8 is that acceptance falls monotonically with width, so a threshold on
acceptance would produce the same two groups without any reference to warp counts. It is worth
writing down before `phase_nmax` lands, because that run is what tells the two apart.

Phase A's five widths cannot. Acceptance across them:

| width | drafter | acceptance | fork group |
|---|---|---|---|
| 3 | mtp | 0.6645 | A |
| 4 | mtp | 0.5579 | A |
| 5 | dflash2 | 0.4737 | B |
| 6 | mtp | 0.4125 | B |
| 8 | dflash2 | 0.3394 | B |

The largest gap between neighbours is 0.1066, from width 3 to width 4, and there is no group
boundary there. The boundary sits at 0.0842, the second largest. So a single acceptance threshold
does not naturally pick out the observed split. It is not excluded either: any cut between 0.474
and 0.558 reproduces the grouping exactly, and five points cannot distinguish that from a cut on
width.

**What `phase_nmax` decides.** It runs widths 2 through 9. Acceptance continues to fall smoothly
across them, so an acceptance account needs two thresholds placed precisely, and width 9 has to
separate on its own despite sitting one small step below width 8. The warp table predicts exactly
that with no free parameters, because `calc_nwarps` drops to one warp above `ncols_dst` 8.

- **Acceptance is the better account** if the groups turn out to be split at acceptance
  discontinuities that do not line up with 4-to-5 and 8-to-9.
- **Neither survives** if the groups do not partition cleanly at all, for instance if two widths
  with the same warp count and similar acceptance give different fork vectors.
- Reported as a comparison of the two, not as a test of H8 alone. `harness/width_groups.py` prints
  the observed partition next to the table's prediction and withholds a verdict when the
  two-drafter control fails, so the acceptance comparison is added to its output rather than
  argued afterwards.

## Cross-check, 2026-08-24: Phase R and Phase R2 agree where they overlap

Phase R's `stock` condition and Phase R2's `sm1700` sit close together: same memory clock at
9501 MHz, same 420 W limit, core clocks 4 % apart. They were produced hours apart by different
mechanisms, Phase R letting the clock settle under a power cap and Phase R2 pinning it with
`nvidia-smi -lgc`, so agreement between them is a real check on both.

| method | run | n | tok/s | core MHz | power W |
|---|---|---:|---:|---:|---:|
| baseline | R stock | 75 | 41.56 | 1785 | 416 |
| baseline | R2 sm1700 | 25 | 41.27 | 1710 | 381 |
| mtp-n3 | R stock | 75 | 63.32 | 1716 | 411 |
| mtp-n3 | R2 sm1700 | 25 | 63.37 | 1695 | 401 |
| mtp-n7 | R stock | 75 | 45.33 | 1732 | 413 |
| mtp-n7 | R2 sm1700 | 25 | 45.13 | 1708 | 398 |

Throughput agrees to 0.7 %, 0.08 % and 0.4 %. For two runs on different days through different
clock mechanisms that is close, and it says the pinning did not introduce an artefact.

**A local elasticity between these two points is not estimable, and the first attempt to compute
one was a mistake worth recording.** The core clocks differ by 4.2 % for the baseline but only
1.2 % and 1.4 % for the speculative arms, which is the same order as the throughput difference
being measured. Dividing one small number by another produced 0.162 for the baseline, -0.065 for
`mtp-n3` and 0.314 for `mtp-n7`. The negative value is not a finding that lowering the clock made
`mtp-n3` faster; it is 63.32 against 63.37, which is noise across a 1.2 % clock step. Only the
baseline's span is wide enough to carry any signal at all, and 0.162 there is consistent with a
bandwidth-bound workload near the top of its clock range and with the 0.491 that Phase R measured
over the far wider and lower `pw-lo` to `stock` span, since elasticity falls as the workload
leaves compute starvation.

The usable comparison is the one Phase R2 was designed for: `sm600` to `sm1200`, a matched
two-fold span with the pin holding exactly on every record.

## ADDENDUM, registered 2026-08-25: a second RTX 3090 on a different host

A second 3090 became available on a separate machine reached over Tailscale. It is worth being
precise about what a second card of the same architecture can and cannot settle, because the
temptation is to treat it as more samples of the same thing, and it is not.

### What differs, and why absolute numbers cannot be pooled

| | host A (primary) | host B (second 3090) |
|---|---|---|
| GPU | RTX 3090, GA102, sm_86 | RTX 3090, GA102, sm_86 |
| default power limit | 420 W | **350 W** |
| CUDA toolkit | 13.3 | **12.0.140** |
| host compiler | gcc 14.2 (Debian 13) | **gcc 13.3 (Ubuntu 24.04)** |
| glibc | 2.41 | 2.39 |
| driver | **610.43.02** | **580.173.02** |

The two cards are different SKUs of the same part: 350 W against 420 W is not a setting, it is the
board's own `power.default_limit`. The drivers are a major version apart, 610 against 580, which is
a wider gap than the rest of the table and wider than was first written here. Every absolute throughput number therefore belongs to its host
and to nothing else, and no result here pools tok/s across the two. This is the same constraint the
three-host fleet already imposed and it is restated because the second card makes it tempting to
forget.

The binary also cannot travel. Host A's `llama-server` links eight in-tree shared objects against
glibc 2.41 and CUDA 13.3; host B has glibc 2.39 and CUDA 12.0, and glibc is not backward
compatible in that direction. llama.cpp is therefore rebuilt on host B from the same revision,
`c060ca974c773c7c3d17fd1b66dc9d312bc292c0`, with the same flags except `LLAMA_CURL`, which is off
because the URL loader is not used and enabling it would pull in a dependency the host does not
have. **The consequence is that host B differs from host A in the toolchain as well as the card,
so a difference between them is not attributable to the hardware alone.** That is stated here
rather than discovered later.

The model bytes do travel, and are checked. The target, the MTP head and the DFlash2 drafter were
copied from host A and verified by SHA-256 against host A's recorded digests, so the weights are
identical even though the code that runs them is not.

### RH1: the #27572 reproduction on an independent machine

`repro/llamacpp_27572.py` runs unchanged apart from its paths. The issue concerns behaviour in the
server's parallel path, so it is a property of the code and should not depend on the card.

- **Reproduces on host B with the same qualitative pattern** - the strongest available evidence
  short of a maintainer reproducing it, and it is reported to the thread either way.
- **Does not reproduce on host B** - then host A's result is either configuration-specific or
  toolchain-specific, and the comment already posted to that issue needs a correction naming host
  B as the disconfirming case. Absolute timings are expected to differ because of the 350 W cap;
  only the pattern is being replicated.

### RH2: does the width partition survive an independent host and toolchain?

The registered mechanism (see the 2026-08-24 addendum) is `calc_nwarps` in the CUDA MMVQ path:
`ncols_dst` of 1 to 4 takes four warps, 5 to 8 takes two, above 8 takes one. Verification width
`w = n_max + 1` enters as `ncols_dst`, so widths 3 and 4 share a kernel configuration, and 5, 6 and
8 share a different one. That is an architecture property of sm_86, not a CUDA-version property,
so the prediction carries to host B.

Fork positions are compared **within host only**: each host's speculative arms against that host's
own greedy baseline. Comparing host A's fork positions to host B's would confound the card, the
driver and the compiler, and no such comparison is made.

- **The grouping reproduces** - w3 equals w4, w5 equals w6 equals w8, and the two groups differ
  from each other on host B as they do on host A. This is the registered prediction. It would show
  the partition is not an artefact of one machine's numerics.
- **The grouping does not reproduce** - the partition is toolchain- or machine-specific, and the
  claim in llama.cpp #25618 must be narrowed to host A in a correction posted to that thread.
- **The grouping reproduces and the absolute fork positions also match host A** - stronger than
  required, and would say CUDA 12.0 and 13.3 produce bit-identical reductions here. Not predicted;
  recorded if observed.
- **A width lands in neither group cleanly** - reported as a failure of both H8 and the acceptance
  competitor, exactly as `harness/width_groups.py` already handles on host A.

Because greedy decoding at a fixed seed is deterministic, one pass per arm is sufficient for the
fork-position comparison and no repetition is run for it. Throughput on host B is recorded but is
not used for any cross-host claim.

### Disclosure: host B's driver was broken and repairing it changed 15 packages

Host B could not run any CUDA workload on arrival. Its NVIDIA kernel modules existed only for
kernel 6.17.0-22-generic while the machine was running 7.0.0-30-generic, because the HWE tracking
package `linux-modules-nvidia-580-open-generic-hwe-24.04` had not followed the kernel. `nvidia-smi`
failed for every user on the box, not only for this study.

Installing `linux-modules-nvidia-580-open-7.0.0-30-generic` fixed it and, as a dependency
consequence, upgraded the userspace driver from 580.126.09 to 580.173.02: 15 packages upgraded, 2
installed, and `linux-modules-nvidia-580-open-6.17.0-22-generic` removed. The machine's owner
authorised this after being shown the exact package list. **The upgrade is not reversible**: version
580.126.09 has been superseded and is no longer fetchable from the Ubuntu archive, and an attempt
to retrieve it from snapshot.ubuntu.com returned nothing. The pre-change package versions are
recorded on that host at `~/.nvidia-rollback/manifest.txt`.

This matters to the results in one way and it is recorded rather than left implicit. Host B's
driver is now 580.173.02 while host A runs 610.43.02, so the two hosts are a **major driver
version** apart, not a point release apart, and the upgrade narrowed that gap without closing it.
"Different driver" therefore joins "different CUDA toolkit" and "different compiler" in the list of
things that separate the two hosts, and it is the largest of the three. It does not affect any within-host
comparison, which is the only kind RH1 and RH2 make.

## Correction 4, registered 2026-08-25, BEFORE `phase_nmax` runs: H8's third group cannot exist

H8 was registered on 2026-08-24 as a three-way partition of fork position by verification width,
`{2,3,4}` against `{5,6,7,8}` against `{9}`, from the `calc_nwarps` table in
`ggml/src/ggml-cuda/mmvq.cu`, which returns 4 warps up to `ncols_dst` 4, 2 warps from 5 to 8 and
1 above 8.

Reading the dispatch rather than the table shows the third group cannot be produced by that
mechanism. `ggml_cuda_mul_mat_vec_q` is selected only when

    src1->ne[1] <= MMVQ_MAX_BATCH_SIZE

and `mmvq.cuh` line 3 defines that as **8**, with the comment "Max. batch size for which to use
MMVQ kernels." A verification width of 9 therefore never reaches MMVQ at all; it is served by a
different kernel. The `default: return 1;` arm of `calc_nwarps` is unreachable on this path for
the GENERIC table, which is the table an sm_86 device selects.

**What this changes.** The registered prediction for widths 2 to 8 stands unaltered: two groups,
split between 4 and 5, from 4 warps against 2. The prediction for width 9 is **withdrawn**. If
width 9 forks differently from `{5,6,7,8}`, that is a boundary between two different kernels and
not evidence about the warp count, and `harness/width_groups.py` must report it as such rather
than as the third arm of H8. If it forks the same as `{5,6,7,8}`, that is also uninformative
about warps for the same reason.

This is recorded before `phase_nmax` has produced a record; `phase_kv` is still running and
`phase_nmax` is behind it in the chain.

## Registered 2026-08-25, before any of it runs: the forced-warp intervention

Everything measured so far is observational. The width grouping is consistent with `calc_nwarps`,
but consistency is not causation: any quantity that changes at the same width would fit the same
data. The experiment that separates them is to change the warp count **without changing the
width**.

`calc_nwarps` is `constexpr` and its result is used as a compile-time constant at `mmvq.cu` line
562, which sizes the cross-warp reduction buffer at line 680

    __shared__ float tmp_shared[nwarps-1][ncols_dst][rows_per_cuda_block][warp_size];

and bounds the reduction loop at line 710, `for (int l = 0; l < nwarps-1; ++l)`. So `nwarps` is
not merely correlated with the reduction order, it **is** the width of the reduction tree. Editing
the table therefore changes the summation order and nothing about the width, the drafter, the
weights or the prompt.

Three builds, from the same revision `c060ca9`, in a clone kept separate from `llamacpp-master`
so the primary tree is never rebuilt:

| build | GENERIC table | prediction |
|---|---|---|
| **control** | unmodified: 1-4 -> 4, 5-8 -> 2 | reproduces this host's own grouping |
| **forced-up** | 1-4 -> 4, **5-8 -> 4** | widths 5, 6, 8 move to the `{3,4}` fork positions |
| **forced-down** | **1-4 -> 2**, 5-8 -> 2 | widths 3, 4 move to the `{5,6,8}` fork positions |

Registered outcomes:

- **Both directions follow the forced warp count and not the width.** The warp count is the cause.
  Nothing else in the build changed, and the two directions rule out an accidental one-way effect.
- **Neither moves.** `calc_nwarps` is not the mechanism, H8's account is wrong however well the
  observational data fitted it, and the llama.cpp #25618 comment needs a correction.
- **One direction moves and the other does not.** Reported as such and treated as unresolved. The
  asymmetry would itself need explaining, and no story is prepared for it in advance.
- **The forced build changes fork positions for widths it did not touch.** The intervention is not
  clean; something else in the build differs, and nothing is concluded until that is found.

The greedy baseline is included in every build as a control: it runs at width 1, which the table
maps to 4 warps in all three builds, so **its output must be byte-identical across the three**. If
it is not, the builds differ by more than the table and the comparison is void.

## Observed 2026-08-25, not predicted: the partition travels, the positions do not

The forced-warp intervention's **control** build ran on host C, the A6000, and its purpose was to
give that host its own baseline before the two modified builds are compared against it. It also
produced a comparison nobody registered, so it is recorded as an observation rather than dressed
up as a prediction.

| | host A | host B | host C |
|---|---|---|---|
| device | RTX 3090 | RTX 3090 | **RTX A6000** |
| SMs | 82 | 82 | **84** |
| compute capability | 8.6 | 8.6 | 8.6 |
| memory bandwidth | 936 GB/s | 936 GB/s | **768 GB/s** |
| CUDA / gcc / glibc | 13.3 / 14.2 / 2.41 | 12.0 / 13.3 / 2.39 | 12.9 / 12.2 / 2.36 |
| partition clean | 25/25 | 25/25 | **25/25** |
| groups differ on | 14 | 14 | **18** |
| fork positions | identical to host B | identical to host A | **different from both** |

Two 3090s that share nothing else - different board SKU at 350 W against 420 W, three CUDA
versions between them, different compilers, a major driver version apart - produce fork positions
that match to the character on all 25 prompts. The A6000 produces the same two-group structure and
different positions.

That is the shape the `calc_nwarps` account predicts without having been asked to. The table is
selected by compute capability and indexed by `ncols_dst`, both of which are identical on all
three, so the grouping should travel. The positions are an argmax outcome of a reduction whose
order depends on how blocks are actually scheduled across the device, which is not identical:
the A6000 has 84 SMs against the 3090's 82, and 768 GB/s against 936.

**What this does not establish.** Which of those device differences moves the positions. SM count
is the obvious candidate because it changes block residency directly, but bandwidth and clock
differ too, and one observation cannot separate three variables. Nothing here is offered as a
cause; it is a fact about three machines.

It also does not weaken the registered endpoint. RH2 compared fork positions **within** host,
which is what the addendum specified, and the cross-host agreement between A and B was registered
in advance as the stronger-than-required outcome that would be recorded if seen. It was seen.

## Correction 5, 2026-08-25: the forced-down direction is void, and one of its controls was impossible

Both problems were found by running the comparison the registration asked for, not by inspection.

### The baseline identity control cannot hold for forced-down

The registration says the greedy baseline "runs at width 1, which the table maps to 4 warps in
all three builds, so its output must be byte-identical across the three". The table it specifies
for forced-down, in the row directly above that sentence, is `1-4 -> 2`. Width 1 is inside that
range. The baseline in the forced-down build therefore runs at two warps, not four, and cannot be
byte-identical to the control's by construction. It is not: 20 of 25 differ.

Both statements were written at the same time and only one can be true. The table is what was
compiled, so the table stands and the control does not. `warp_intervention.py` reports that gate
as not applicable for a build that changes width 1, rather than as a failure, because failing it
would blame the measurement for a contradiction in the design.

The control is still valid for forced-up, which leaves width 1 at four warps, and it passes there:
25 of 25 baselines byte-identical.

### Forced-down changes widths it did not touch

Forced-down leaves `5-8 -> 2`, the stock value, so widths 5, 6 and 8 run at the same warp count in
that build as in the control. Their output should be identical. It is not:

| widths | forced_down == control | expected |
|---|---|---|
| 1, 3, 4 | 5/25, 4/25, 4/25 | differ; the edit changed them from 4 warps to 2 |
| **5, 6, 8** | **5/25 each** | **25/25; the edit did not touch them** |

This is the fourth registered outcome, and it was registered precisely so this would not be
explained away: "The forced build changes fork positions for widths it did not touch. The
intervention is not clean; something else in the build differs, and nothing is concluded until
that is found." The forced-down direction is therefore void and the bidirectional result the
design was built for is not available.

Two things narrow where to look. Control and forced-up agree byte for byte at every width
forced-up left alone, 25 of 25 at width 1 and 50 of 50 across widths 3 and 4, so the build process
is deterministic and the effect is specific to this edit rather than to rebuilding. And the three
libraries differ in size, not only in content:

    control      57765192 bytes
    forced_up    57806152 bytes   +40960
    forced_down  57621832 bytes   -143360

`calc_nwarps` is `constexpr` and its result is a template argument, so the set of instantiations
the compiler emits is a function of the table. Forced-down collapses the GENERIC table to a single
warp count and its library is the smallest of the three, which is consistent with instantiations
disappearing. Whether that is what moves the untouched widths is not established here and no claim
is made from it; it is where the next test should go.

### What survives

Forced-up passes all three of its gates and is scored. Forced-down is void. A single direction
cannot separate the warp count from an accidental one-way effect, which the registration also
says, so there is no bidirectional causal result to report.

## Correction 6, registered 2026-08-25 06:28, before the run: why forced-down was void, and the rebuild

Correction 5 recorded that forced-down changed widths it did not touch and left the direction
void with no explanation. There is now an explanation, it is not a build fault, and the design is
at fault.

Disassembling all three libraries settles what the edit did to the binary. `calc_nwarps` is
`constexpr` and derived from `ncols_dst` inside the kernel rather than being a template argument,
so every build carries the same 294 `mul_mat_vec_q` symbols with the same names; what changes is
the code inside them. Hashing the SASS of each instantiation:

| | ncols_dst 1-4 | ncols_dst 5-8 |
|---|---|---|
| control vs forced-up | identical, 179 of 179 | changed, 92 of 92 |
| control vs forced-down | changed, 179 of 179 | identical, 92 of 92 |

Both edits are surgical in the binary. Every kernel each one should have left alone is
byte-identical machine code. So forced-down's widths 5, 6 and 8 ran exactly the control's code and
still produced different text on 20 of 25 prompts each.

The reason is that `ncols_dst` is not the verification width. It is the number of columns in the
matrix-vector product, and **a drafter generates one token at a time, so a drafter runs at
`ncols_dst` 1**. Every speculative arm has a drafter. Setting the whole 1-4 row to two warps
therefore perturbs every arm through its drafter regardless of its verification width, and moves
the greedy baseline as well, which is why the registered baseline control could not hold.

That control was not a stray sentence. It was load-bearing, and the forced-down table specified
alongside it violated it. Reading the two together should have caught this before anything ran.

### The rebuild, and what is predicted before it finishes

`forced_down2` splits the row:

    case 1, 2       -> 4     unchanged: the drafter and the greedy baseline are untouched
    case 3, 4       -> 2     the intervention
    case 5, 6, 7, 8 -> 2     unchanged

Registered now, with the run in progress and no results seen:

- **The greedy baseline must be byte-identical to the control's on all 25 prompts.** Width 1 is at
  four warps in control, forced-up and forced_down2 alike. If it is not, the drafter account above
  is wrong and so is this build.
- **Widths 5, 6 and 8 must be byte-identical to the control's.** Their kernels are unchanged and
  their drafters are now unchanged too. This is the gate the first forced-down failed, and it is
  the whole reason for the rebuild.
- **Widths 3 and 4 must change**, or the edited row never reached the kernel.
- The prediction under test is unchanged: **widths 3 and 4 adopt the {5,6,8} fork positions** if
  the warp count is what puts the widths into two groups. Forced-up already failed the mirror of
  this on 15 of 18 informative prompts, so the honest expectation is that this fails too; that is
  written down here so that a failure cannot later be presented as the expected result and a
  success cannot be presented as a surprise.

If widths 5, 6 and 8 come back identical and widths 3 and 4 do not follow, then both directions
agree that the warp count changes the numerics without carrying the fork positions, and H8's
account of the partition is refuted rather than merely unsupported.

## Correction 7, registered 2026-08-25 07:30, before either run finishes

Two results landed together and both go against what was registered.

### Forced-down2 failed the gates its rebuild was designed to pass

Correction 6 predicted, before the run, that a table splitting the 1-4 row so widths 1 and 2 keep
four warps would leave the greedy baseline and widths 5, 6 and 8 byte-identical to the control,
and named what a failure would mean. The run finished and:

| width | registered | observed | |
|---|---|---|---|
| 1 (baseline, no drafter) | identical to control | 2 of 25 | **FAIL** |
| 3, 4 | must change | changed | PASS |
| 5, 6, 8 | identical to control | 5 of 25 each | **FAIL** |

Correction 6 says in as many words that width 1 moving means the drafter account is wrong. It
moved. The baseline has no drafter at all, so that account cannot explain it and is withdrawn.

Disassembly makes it worse rather than better. Hashing the SASS of all 294 `mul_mat_vec_q`
instantiations, control against forced_down2: `ncols_dst` 1 is **byte-identical, 110 of 110**,
`ncols_dst` 2 identical, 3 and 4 changed, 5 through 8 identical. The edit was surgical in the
binary. A width-1 request runs byte-identical machine code in the two builds, on the same card,
with the same weights, at greedy, and produces different text on 23 of 25 prompts.

### The 3090 reproduces the A6000, so it is not one card

| | A6000 | RTX 3090 |
|---|---|---|
| forced-up validity gates | all pass | all pass |
| registered prediction held on | 3 of 18 | 4 of 14 |
| forced-down | void, untouched widths moved | void, untouched widths moved |

Different architecture generation, different toolchain, same shape. **`calc_nwarps` is not
supported as the cause of the fork-position grouping on either device.**

### The control that should have run first

Every one of these comparisons assumes a build produces the same output twice, and nothing has
tested it. The control build is being re-run now on host C, with the same binary
(`libggml-cuda 5f77e30401ac31ab`, unchanged), against the same prompts. Registered before it
finishes:

- **It reproduces byte for byte, 150 of 150.** Then the build is deterministic, every difference
  between builds is a build difference, and forced_down2 changing the width-1 baseline while its
  width-1 kernels are identical SASS is a real effect that this study has not yet named. The next
  step is hashing every kernel in the library rather than only `mul_mat_vec_q`, since something
  outside that set has to be carrying it.
- **It does not reproduce.** Then run-to-run variation is on the same scale as what was attributed
  to the table edit, and every forced-warp comparison here has to be re-read against it - including
  the forced-up gates that passed 25 of 25 and 50 of 50, which would then be measuring a
  reproducibility this control says does not exist. That would not touch the throughput results,
  which are paired and averaged over passes, but it would end the intervention as designed.
- **It reproduces on some widths and not others.** Reported as such. The widths that fail are the
  ones every other comparison has to drop, and no account is prepared for a partial result.

Writing this down now matters more than usual: the second outcome would invalidate a line of work
this repo has spent a day on, and the temptation to find a reason the control does not apply would
be strongest exactly then.

## Correction 7a, registered 2026-08-25 08:05, with the control still at 100 of 150

The 3090 reproduced the A6000's forced_down2 gate failures - baseline 7 of 25 identical against
2 of 25, widths 5, 6 and 8 at 4 of 25 against 5 of 25 - so whatever it is, it is not one card.

Then a confound turned up in how the runs were scheduled, and it is written down here before the
control finishes because it is exactly the kind of thing that would otherwise be noticed
afterwards and used to explain away whichever result was inconvenient.

| comparison | how it ran | width-1 gate |
|---|---|---|
| control against forced-up | same chain, back to back at 04:33 | **25 of 25** |
| control against forced-down | same chain, back to back at 05:14 | 4 of 25, and that build did change width 1 |
| control against forced_down2 | a separate invocation at 06:29, 37 minutes later | **2 of 25**, and that build did not change width 1 |

The only comparison that holds at width 1 is the back-to-back one. Build variant is confounded
with invocation session, and the failing gate is the one where the two are not the same session.

The determinism control now running is control against control in a separate invocation, which is
the contrast that separates them. Its outcomes were registered in Correction 7; this narrows what
each of them means:

- **150 of 150.** A separate invocation does reproduce, the session is not the confound, and
  forced_down2 changing widths whose machine code is byte-identical is a real effect still without
  a name.
- **Fewer than 150.** A separate invocation does not reproduce on its own, and forced_down2's gate
  failures are session effects rather than build effects. The forced-up gates would survive, since
  they were taken back to back, but no comparison across invocations in this study can be read,
  and the intervention needs re-running with every build in one session.

Under either outcome the throughput results are untouched: they are paired within a run and
averaged over passes, which is what that design is for.

## Correction 7, resolved 2026-08-25 08:13: the build is deterministic, so the effect is real

The control came back **150 of 150 byte-identical**. The same binary, the same card, the same
prompts, at greedy, run again as a separate invocation almost four hours after the first:

| width | arm | identical between the two runs |
|---|---|---|
| 1 | baseline | 25 / 25 |
| 3 | mtp-n2 | 25 / 25 |
| 4 | mtp-n3 | 25 / 25 |
| 5 | mtp-n4 | 25 / 25 |
| 6 | mtp-n5 | 25 / 25 |
| 8 | mtp-n7 | 25 / 25 |

That is the first registered outcome, and it **refutes the session confound registered in 7a two
hours earlier**. A separate invocation reproduces perfectly, so forced_down2 running in its own
session is not what made its gates fail. Every difference measured between builds is a build
difference.

Which leaves the thing that has no name yet, now with nowhere left to hide:

- forced_down2 leaves the width-1 row of the GENERIC table alone.
- Its 294 `mul_mat_vec_q` instantiations at `ncols_dst` 1 are byte-identical SASS to the
  control's, 110 of 110, and at 5 through 8 as well.
- Its width-1 greedy baseline, which has no drafter at all, differs from the control's on 23 of
  25 prompts on the A6000 and 18 of 25 on the 3090.
- The build reproduces itself exactly.

So a width-1 request runs byte-identical `mul_mat_vec_q` code in two builds that each reproduce
themselves, and produces different text. The carrier is outside the symbols that were hashed.
Hashing every kernel in `libggml-cuda.so` rather than only `mul_mat_vec_q` is the next test and is
running.

If that comes back with only `mul_mat_vec_q` differing, and only at the widths the edit names,
then no kernel's machine code carries the difference and the search moves to the host side: launch
configuration, the order the scheduler assigns blocks, or something in the module that is not a
kernel at all.

## Correction 8, 2026-08-25 08:50: forced-down2's gate failures are a build-configuration artefact

The thing with no name has one. It is not the table, not the drafter, not the card, not the
session, and not run-to-run variation. It is that `forced_down2` was compiled under a different
cmake configuration from the build it was compared against.

The chain of tests that got there, each one narrowing rather than explaining:

| test | result |
|---|---|
| same binary, separate invocation, four hours apart | **150 of 150 byte-identical** - execution is deterministic |
| every kernel in `libggml-cuda.so`, control against forced_down2 | 6202 kernels, **46 differ, all `mul_mat_vec_q`**, only at `ncols_dst` 3 and 4 |
| every shared library in the two build directories | **only `libggml-base.so` differs**, and it is not CUDA |
| patch `mmvq.cu` and rebuild `libggml-base` | unchanged at `386e6470`; the table edit does not touch it |
| what kind of difference | `.text` and `.rodata` both differ, 8813 bytes: machine code, not metadata |
| six full rebuilds of the unchanged tree | **one distinct hash, `386e6470`**, matching control |
| the build logs | `warp_build_forced_down2.log` **begins with a cmake configure**; the restore build after it does not, and the first chain's three builds took 6 to 72 seconds each, which is incremental |

A cmake reconfigure regenerates `flags.make` for every target, so all eleven `ggml-base` sources
recompiled and produced `64cae0a5`. control, forced-up and forced-down were built incrementally
inside one session and all three carry `386e6470`.

So the forced-down2 comparison is between two builds produced under different configure states,
and "the only thing that differs is the table" is false for it. That is sufficient to explain a
width-1 greedy baseline, which has no drafter and whose `mul_mat_vec_q` machine code is
byte-identical, coming out different: the CPU-side library was not the same one.

### What survives and what does not

- **Forced-up stands.** It was built in the same session as the control with no reconfigure
  between, all three of its registered gates pass, and its registered prediction held on 3 of 18
  informative prompts. `calc_nwarps` remains unsupported as the cause of the fork grouping, on
  both devices.
- **Forced-down and forced_down2 are withdrawn**, on both hosts. Neither can be read.
- **The registered design needs one more constraint**, which nothing wrote down because nobody
  thought of it: every build in an intervention must come from one configure. Building them in
  separate sessions makes the comparison a comparison of configure states.
- **Nothing here touches the throughput results**, which are paired within a run and averaged over
  passes.

### Registered for the re-run, before it starts

All four builds - control, forced-up, forced-down2 and a second control - produced back to back
inside one configure, then run back to back. The second control is there so the next version of
this question has an answer already in the file: two builds from one configure that differ in no
source at all must produce byte-identical output, and if they do not, no comparison in the set
means anything.

## Correction 8a, 2026-08-25 09:05: two confirmations, one of them of my own mistake

**The reconfigure account is confirmed independently.** Stripped of symbols and build id, the
`libggml-base.so` code hashes are:

| build | how it was built | stripped code |
|---|---|---|
| control, forced-up, forced-down | incrementally, inside the 03:11 session | `2d4c5212` |
| forced_down2 | after a cmake configure | `67f70901` |
| the new v2 control | after a cmake configure | **`67f70901`** |

A fresh configure produces a different `libggml-base.so` than an incremental build of the same
tree, and it produces the *same* different one twice. That is Correction 8's claim, arrived at a
second time from the other direction.

**The first attempt at the re-run stopped on a gate that was wrong.** It compared whole-file
sha256 and found control and control2 differing in `libggml-cuda.so` despite identical source
under one configure. Before reporting that the CUDA build is not reproducible - which would also
have turned the earlier finding that 6156 of 6202 kernels are identical into noise - the two files
were compared directly:

    sizes equal, four bytes differ, at offset 56408893
    control  "1586"      control2  "37ec"
    build ids identical
    stripped content byte-identical

Four hex characters deep inside the file, from an identifier nvcc derives from its temporary
filenames. The code is reproducible; the identifier is not. The gate now hashes stripped content
and the offset is written into the script so the next person does not have to find it again.

Worth keeping the gate rather than loosening it out of embarrassment: it is what forced the
comparison that produced the confirmation above. A gate that fires and turns out to be too strict
still did its job, provided the response is to look rather than to lower it.

## Correction 9, 2026-08-25 19:40, BEFORE Phase M runs: H6a is anchored on the wrong number

H6a registers that Phase M's replication arm holds if `moe-draft08b-n8` "is worse than -25 %"
against the MoE baseline, anchored on the predecessor's -44.6 %. Both halves are wrong, and the
gate as registered would fail a correct replication.

**The -44.6 % belongs to a different method.** It is the predecessor's `06_dflash_max8`, DFlash
with a BF16 drafter. Phase M runs no DFlash arm. The arm it replicates is `draft-q35-08b-max8`,
the 0.8B draft-then-verify one, published at 121.06 against 135.69 tok/s: **-10.8 %** raw. On the
class-stratified estimand this repo uses throughout, `docs/METHODOLOGY_AUDIT.md` puts the same
data at **-21.5 %**. A faithful replication should therefore land near -21.5 %, which is inside
the region the registered gate calls a failure.

The gate moves to a band of -12 % to -32 % around the stratified figure. `run_remaining.sh` is
updated with it, and now writes `results/phase_m_anchor_ok` only when the band is met; the 22 GB
MoE target is no longer deleted while the anchor is unresolved, since that is exactly when the
model is needed.

**The MoE premise is separately contradicted by this repo's own Phase C.** H6 and H6a are built on
the predecessor's account that the loss comes from MoE expert loading. Phase C ran the same 0.8B
drafter at the same n-max 8 against the DENSE 27B target, on this harness, this card and this
build, and measured:

```
draft08b-n8    baseline@master   -29.81 [-33.09, -26.39]  SLOWER
```

Worse than the MoE arm, in a model with no experts, no routing and no union to load. Expert
saturation cannot be the cause of a loss that reproduces at least as badly without any experts.
This is a cross-study comparison of the MoE side, so it is not decisive on its own; it is decisive
enough that Phase M must not be reported as identifying MoE routing.

**And the "acceptance near 100 %" that motivated the account is a broken counter.** Every
predecessor config has `total_draft == total_accept` exactly, because the log line divided
accepted by accepted. The adjacent line gives 115 accepted of 214 drafted, 53.7 %. The current
build prints a real ratio, so the two studies do not measure the same quantity.

H6a is re-anchored as above. H6 stands as registered but is exploratory: it is a near-zero test,
and at n = 25 prompts the interval half-widths in this repo run 3 to 6 points, so any true effect
between -8 % and 0 is not resolvable. H4a's "isolating MoE routing" is withdrawn; the addendum at
line 479 already conceded that model size is not controlled, and Phase C now removes the premise
as well. Phase M keeps its value as a replication of the predecessor's arm on a controlled
harness. It is not a dense-versus-MoE identification, and all nine of its arms are MoE.

**Phase Q was never registered.** It has no entry anywhere above. The README's follow-up table is
corrected to say so rather than letting the table's preregistered framing cover it. Phase Q also
has no inferential machinery: `harness/cost_model.py` reports `c` as a bare point estimate with no
interval, so a difference in `c` between quantization rungs currently has nothing to be compared
against. Both are prerequisites before it runs.

## Correction 10, 2026-08-25 21:10, BEFORE Phase M runs: the phase is given the arms its name claims

Correction 9 recorded that Phase M could not identify MoE routing. It did not fix the reason. The
matrix declared one model and every one of its nine arms ran it, because `Arm` had no `model`
field and `bench.py` passed the matrix default to every server it started. A phase called "dense
against MoE" therefore had no dense arm in it, and its dense side would have come from Phase A and
Phase C, measured one and two days earlier on different builds of the harness.

`Arm` now carries an optional `model`, `bench.py` uses `arm.model or model` when it starts a
server, and because the harness already starts a fresh server for every arm-pass, changing the
model between arms costs nothing beyond the load it was going to do anyway. Runs where no arm
overrides record exactly what they recorded before: `arm_models` is empty and `env.model` is
unchanged. Where an arm does override, every distinct file is hashed into `arm_models`, so a
two-model run is auditable rather than being described by a single `env.model_sha256` that would
have been true of only part of it.

Three dense arms are added: `baseline-dense`, `dense-mtp-n2` and `dense-draft08b-n8`. A pair now
differs in the model and nothing else: same harness, same card, same prompts, same build, same
hour. `BASELINE_MAP` is rewritten to score each arm against the baseline that ran its own model;
as first written it sent every arm to `baseline-moe`, which would have scored the dense arms
against an MoE baseline and reported the difference between two models as a speculation effect.
`test_every_arm_is_paired_with_a_baseline_on_its_own_model` now fails on that mapping.

Two more arms are added to the draft-then-verify side, `moe-draft08b-n2` and `-n6`. Its existing
n_max of 4, 8 and 16 are verification widths 5, 9 and 17, and only 5 is inside the MMVQ dispatch
path, so a fit over that path had one point and could not produce a `c` at all. Widths 3 and 7
make it fittable, which is what H6b needs.

`--spec-draft-n-min 4` is added to every draft-then-verify arm. The predecessor's v1 passed
`--draft-min 4`; this matrix passed nothing, and the current default is 0
(`common/common.h:326`), so the replication arm differed from the thing it replicates in a
parameter neither study had noticed.

The matrix goes from 9 arms to 14 and from 675 records to 1050, about three hours rather than two.
`expect_for` in `run_remaining.sh` derives the count from `len(ARMS)` and needs no change.

None of this is registered as a new hypothesis. H6, H6a and H6b are unchanged in content; what
changes is that the matrix can now address them. H6a keeps the -12 % to -32 % band from
Correction 9.

**Phase Q, same audit.** Its ladder is three unsloth UD dynamic quants and one uniform `Q8_0`, so
bit width is confounded with quantization scheme; results are to be plotted against each file's
measured effective bits per weight rather than its label. On a 24 GB card only the first two rungs
are reachable, a span of about one bit, where H2' predicts roughly -8 % in `c` against the 2.7 %
run-to-run drift already seen in `c` between `phase_a` and `phase_nmax` - three times the noise,
from two points. `phase_qsmall` spans Q4_K_M to BF16, about four times the bit range, and is the
better instrument on this hardware. H2' is also not stated in the same units as `c`: theirs is a
per-extra-token throughput cost from `llama-batched-bench`, `c` is a slope in serial-decode-step
equivalents, and no derivation relates them, so the ladder tests the direction of the claim and
not its figures. The `UD-Q4_K_XL` rung repeats arms `phase_a` and `phase_nmax` already ran on the
same file; it is kept as the same-session control and is not new evidence about Q4.

## Correction 11, 2026-08-25 21:35, BEFORE Phase M runs: the dense side needed a ladder, not a point

Correction 10 gave Phase M a dense side so the comparison would happen in one session. It gave it
three arms: a baseline, one MTP depth and the anchor depth. That is enough to compare LEVELS and
not enough to compare SLOPES, and H6b is about a slope.

H6b registers that fitting `k(w) = k0 + c(w-1)` on the MoE gives a marginal cost per verified
position exceeding the dense model's. Measured on the matrix as Correction 10 left it:

```
MoE   draft-mtp     on-MMVQ widths [2,3,4,6,8]   c fittable
MoE   draft-simple  on-MMVQ widths [3,5,7]       c fittable
DENSE draft-mtp     on-MMVQ widths [3]           c CANNOT be fitted
DENSE draft-simple  on-MMVQ widths []            c CANNOT be fitted
```

Correction 10 added widths to the MoE side, because that was the defect the audit named, and left
the dense side at one point. A run would have produced 1350 records answering H6 and H6a and not
H6b, and c_dense would have had to be borrowed from `phase_nmax`, measured in another session on
another day, which is the comparison these arms exist to remove. The fix would then have been
another full run.

The dense side now carries the same ladder as the MoE side on both paths: `draft-mtp` at n_max 1,
2, 3, 5 and 7, and `draft-simple` at n_max 2, 4 and 6. The anchor arm stays at n_max 8, which is
width 9 and off the MMVQ path, so it contributes a level and not a slope. Both paths are matched
on both models, which is what a paired comparison of `c` needs.

The draft-then-verify ladder is included on the dense side rather than only MTP because that is
the path the predecessor's loss lives on. A difference in `c` there says more about where the loss
comes from than the same difference on a path the predecessor never ran.

Phase M is now 21 arms and 1575 records, about 4.8 hours. No hypothesis changes; the matrix can
now address the ones already registered.

`test_every_method_on_both_models_has_matched_fittable_widths` fails on the Correction 10 shape,
naming the unmatched widths.

## Correction 12, 2026-08-25 22:05, BEFORE Phase M runs: run order was confounded with the model

Corrections 10 and 11 built Phase M's dense side by appending arms to the end of the list. That
put the model on the same axis as the run order.

`bench.py` rotates arm order by one position per pass, `rot = (p_idx - 1) % len(arms)`. With 21
arms and 3 passes the dense arms sat at positions 11-20, 10-19 and 9-18 and never once ran in the
first nine. Whatever varies with position in a four-hour session - card temperature, the clock
drift this repo has measured at 1.8 % across a ladder, page cache - would have varied with the
model, which is the one thing the phase is comparing. The invocation in `run_remaining.sh` passes
no `--settle-floor`, so there is no thermal gate absorbing it either, and that script cannot be
edited while the chain is reading it.

The arm list is reordered so each matched pair is adjacent. Adjacency does not remove the position
effect; it makes both halves of a pair meet the same one, which is what a paired comparison needs.
Measured over the three rotations, every pair is one position apart and the median position is 10,
11, 12 for the MoE side against 11, 10, 9 for the dense side, where before it was 0-10 against
11-20.

The two baselines lead, then the anchor pair. The anchor is the tightest configuration in the
matrix, the 21.3 GiB MoE target plus a 0.5 GiB drafter against a 24 GiB card, so if it does not
allocate that is known about ten minutes in rather than an hour, and H6a gates every other reading
in the phase anyway.

`moe-draft08b-n16` keeps no dense twin. It is a replication point for the predecessor's second
depth, not a comparison point.

No hypothesis changes and no arm is added or removed. `test_a_pair_never_runs_far_apart` fails on
the previous order, naming the pair and the two positions.

## Correction 13, 2026-08-26 00:45, DURING Phase M: `c` was reported as differing on an interval that cannot say

The README, `docs/COST_MODEL.md` and `docs/UPSTREAM_CONTRIBUTIONS.md` all stated that the completed
n-max ladder shows the two methods' marginal costs to be different -- `c = 0.2481` for
`draft-dflash` against `0.2904` for `draft-mtp` -- and cited the paired interval
`[-0.0434, -0.0413]` as clearing zero. One of them drew the inference that part of the marginal
cost moves with the drafter. A test enforced the claim. All of that is withdrawn.

The interval came from `cost_model.fit_ci`, which redraws which PROMPTS contribute to each width's
mean `k`. It never asks whether a straight line is the right shape for those means, and a slope
comparison is a question about shape. Taken against the fits' own residuals across widths the
numbers are `se(c) = 0.0181` on **one** residual degree of freedom for `draft-dflash` and `0.0084`
on five for `draft-mtp`, combined `0.0199` -- twenty times the bootstrap's half-width. The 15 %
gap is 2.1 combined standard errors at 1.5 Welch degrees of freedom, where the two-sided 95 % point
is 12.7.

So the comparison is short of WIDTHS, not of prompts, and it resolves nothing either way. Phase A's
two-point near-agreement (0.2784 against 0.2829) carried the opposite inference and is equally
unsupported. `cost_model.py` now prints both uncertainties for every fit and takes the verdict from
the width residuals; `test_the_prose_does_not_settle_c_on_the_wrong_interval` enforces the rule in
both directions and refuses silence, because silence reads as agreement.

Two things found in the same pass, both recorded here because they change what earlier numbers
mean rather than only how they are printed:

**`k(w=1)` is an exact anchor that no fit reproduces.** At zero draft depth a cycle is a plain
decode step, so `mean_len` and `speedup` are both 1 and `k = 1.0` exactly. No fit sees that point.
Every fit on the dense target extrapolates below it -- 0.7187, 0.7799, 0.7825, 0.8888, 0.8937,
0.8986, 0.9443 across the five completed matrices -- and a cycle cheaper than a decode step does
not exist, so `k(w)` is concave and the first extra position costs more than `c`. Refitting with the
anchor moves `c` by 3.0-3.4 % and holds r^2 above 0.99, so the linear form is sound over the widths
it is fitted on. What does not survive is reading `k0` as a fixed overhead, on either method or
either architecture.

**TEST 1's rejection bound was decided by a point estimate's sign, with `r^2` printed and never
consulted, and its summary took `max` over the arms' point estimates.** The maximum of several
noisy estimates is biased upward and bounds nothing, and that same quantity gated whether TEST 1's
conclusion was printed at all, so on the live Phase M data one arm clearing zero by 0.10
half-widths suppressed the finding. The model also held the draft length at `n_max`, which the
server does not: it reuses a surviving draft tail instead of re-drafting
(`tools/server/server-context.cpp:2893`), and on the 0.8B `draft-simple` arms the realised length
is **4.20 against an `n_max` of 8 and correlates with acceptance at +0.94**. The regressor is inside
the response there, and the induced bias is negative in `r`, which is TEST 1's own conclusion. The
bound is now the largest upper confidence limit over arms whose draft length is stable, converted to
the share of the cycle it would account for: 0.08 % to 1.37 % across the five completed matrices.

No hypothesis changes. H6b's endpoint is unchanged and is now reported against the wider of its two
uncertainties: on Phase M the difference between the MoE's and the dense target's marginal cost per
verified position is bounded to about +/-14 %, against an expert-saturation account that predicts
the MoE's should be the larger.

## Correction 14, 2026-08-26 01:20: Correction 13 was itself wrong, and the reason names a third defect

Correction 13 withdrew the finding that `c` differs between `draft-dflash` and `draft-mtp`. That
withdrawal is withdrawn. The finding stands, at a larger magnitude than either earlier version, and
the argument used to withdraw it was a real statistical error that was silently biasing every
comparison in this file toward a null.

**What Correction 13 got wrong.** It took each fit's residual across widths as a standard error on
that fit's slope and added the two in quadrature. Over widths 3, 5 and 7 the two arms' residuals
are `+0.0209, -0.0418, +0.0209` and `+0.0210, -0.0420, +0.0210`. Those are the same numbers. The
residual is not independent noise on each fit; it is curvature in `k(w)`, it is deterministic, and
it is shared between two arms measured on the same card. Adding a shared quantity in quadrature
inflates the bound on a *difference* that it largely cancels from -- here by a factor of about
twenty.

Two reasons the residual vectors had to look alike, and only the second is evidence: with three
equally spaced widths the residual must be orthogonal to the constant and the linear direction, so
it is forced to be proportional to `[1, -2, 1]` and its *shape* carries nothing; what is
informative is that the *magnitudes* agree to 0.5 %.

Taking the difference first: the two `k(w)` curves differ by a straight line to within `2.4e-4`,
and the slope of that difference is `-0.04729` with a residual standard error of `0.000104` -- 456
standard errors from zero against a two-sided 95 % point of 12.71 at one degree of freedom. The
paired prompt bootstrap on the same restricted range agrees: `-0.0473 [-0.0489, -0.0456]`.

**The third defect, which is why both earlier readings were off.** `k(w)` is curved -- every fit in
this study lands below the floor a zero-depth cycle must cost -- so a fitted slope is a CHORD, and
a chord over widths 3 to 7 is not the same quantity as a chord over 2 to 8. The two methods were
being compared over whatever widths each happened to run. Matched on the shared widths the
difference is `-0.0473` rather than `-0.0424`, a sixth of the effect. **Phase A is the sharper
case: it fits DFlash2 on {5, 8} and MTP on {3, 4, 6}, which share no width at all**, so its
"the two coefficients agree to within 1.7 %" compared chords of disjoint arcs. That comparison is
now refused rather than printed, and the inference it once carried is void on those grounds rather
than on Correction 13's.

**What the corrected procedure is.** Restrict both fits to the widths they share; use the paired
prompt bootstrap for sampling uncertainty; check shape on the DIFFERENCE, not on each fit. Where
the curvature cancels, the bootstrap interval decides. Where it does not, the shape bound binds and
is reported as binding.

**H6b is unchanged in verdict and better supported in method.** Phase M's two targets share all
five widths, so no range mismatch arises. Their difference is `+0.0029 [-0.0007, +0.0064]`, but
their curves are *not* parallel -- residuals reach 0.15 -- so the shape bound of `+/-0.0775` binds
and the comparison is **not resolved**. That rules out a large architecture effect, against an
expert-saturation account predicting the MoE's marginal cost per verified position should be
clearly the larger. It does not establish equality, and the README no longer says +/-14 %, which
came from the quadrature error.

**Two smaller items from the same review.**

`k(w=1)` was described in Correction 13 as "the exact 1.0". It is a floor, not a value: a
zero-depth cycle is a decode step plus a drafter that costs at least nothing, so `k(1) >= 1.0`. The
exact 1.0 belongs to the baseline arm, which runs no drafter and is a different configuration. The
conclusion is unaffected -- every dense fit lands *below* the floor, which is impossible whatever
the drafter costs -- but the pinned refit is a bound-constrained sensitivity check and not an
anchor.

The matched-acceptance pairs added in the same pass were reported without checking that the two
arms verify at the same width. Phase M's headline pair -- `moe-draft08b-n4` at 38.7 % acceptance
against `moe-mtp-n5` at 38.6 % -- runs at **3.32 columns against 5.97**. The pair is now flagged,
and the flag carries a direction, because a bare "confounded" would tell a reader to discard a
finding that survives: the arm verifying 2.6 more columns per cycle, at about 0.80 extra
decode-steps priced at the fitted `c`, is the one that is 76 points of baseline *faster*. The
confound runs against the gap and cannot explain it.

No hypothesis changes.
