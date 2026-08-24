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
| power cap | 350 W (stock) | 350 W (stock) | **420 W default; reset to stock — see below** |

The 420 W default is a different board SKU/vBIOS from the 350 W cards used in v1-v3. Absolute
tok/s are **not** comparable across these three hosts. Only within-host paired deltas are.

**Overclock, found and corrected mid-study (2026-08-24).** Ten minutes into the first full
Phase A run the card was discovered to be carrying `GPUMemoryTransferRateOffset=800`
(memory +400 MHz), `GPUGraphicsClockOffset=100`, and a 450 W limit against its 420 W default —
while this document described it as stock. **That run was discarded, not kept.** The offsets were
zeroed and the limit returned to 420 W, which restored the maximum memory clock to 9751 MHz, the
exact stock figure recorded in the predecessor repo's `BENCHMARK_ENV.md`. Full record in
`docs/GPU_AS_FOUND.md`.

This is not a cosmetic correction. Batch-1 decode is memory-bandwidth-bound while speculative
verification is comparatively compute-dense, so a memory overclock moves the baseline and the
speculative arms by *different* amounts — precisely the kind of differential that a paired
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

**H2' (competing explanation, from the PR #27342 author — must be discriminated, not assumed away).**
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
| depends on rejection rate? | **yes — cost is paid only on rejection** | no — cost is paid per drafted token regardless |
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
  net loss — isolating MoE routing, not consumer Ampere, as the cause.

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
  PR #27342 (DFlash2, **open**), issue #22947 (llama-bench spec-decode support — closed as not planned),
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

so `k` — the cost of one speculative verification step in units of a plain decode step — is
recoverable per request. Measured:

| arm | verification width | k | k spread across 5 prompt classes |
|---|---:|---:|---:|
| mtp-n2 | 3 | 1.4494 | 1.06 % |
| mtp-n3 | 4 | 1.7420 | 1.15 % |
| dflash2-n4 | 5 | 1.8871 | 1.79 % |
| mtp-n5 | 6 | 2.2929 | 1.38 % |
| dflash2-n7 | 8 | 2.7142 | 2.39 % |

(Figures corrected 2026-08-24 after a review of the estimator itself — see "Estimator
correction" at the end of this entry. The earlier numbers used llama.cpp's reported `mean len`
field and were biased by a prompt-dependent amount.)

**Test of H2.** A state-rollback account charges the overhead to *rejection*. Modelling that as
`k = k_verify + r * n_max * (1 - acceptance)` makes `r` estimable from the slope of k against
acceptance. Acceptance spans 0.096–0.918 in this data — nearly a ten-fold range — and every arm
returns `|r| <= 0.0024` with r² between 0.001 and 0.065 — no relationship at all. No
rejection-proportional cost is detectable on either drafter.

**This does not say rollback is free.** It bounds how much of the measured overhead rollback can
account for, and the bound is approximately nothing. H2 as stated — rollback as the dominant term
setting the n-max ceiling — is not supported.

**What replaces it.** Fitting `k = k0 + c * (w - 1)`:

- `draft-mtp`, widths 3/4/6: k0 = 0.8937, **c = 0.2803**, r² = 0.9998
- `draft-dflash`, widths 5/8: k0 = 0.7844, **c = 0.2757** (two points only: a straight line
  through two points is perfect by construction and this r² carries no information; Phase N adds
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

the field is high by +0.17 % to +0.81 %, **and the error varies by prompt** — the same order as
the cross-class constancy of `k` that this entry uses as evidence. In other words the estimator
was contaminating the exact quantity the claim rests on.

An anomaly noted in the first version — that k sloped slightly *positive* against acceptance on
every arm — turned out to be that bias, not a physical effect: the field's error is largest on
high-acceptance prompts. Recomputing from the API counters removes it. The slopes now straddle
zero (−0.0167 to +0.0033) with r² between 0.001 and 0.065, the within-arm spread of k tightens
(e.g. mtp-n2 1.36 % → 1.06 %), and the conclusion is unchanged but better supported.

Deriving from the API counters also removed an unverified assumption: the earlier version aligned
log lines to prompts by position. The log is now used only as an independent cross-check, and it
agrees with the API counters on **150/150 requests, to the token**.

### 2026-08-24: Phase A complete (875/875, 0 incidents), and two corrections

**Result.** Every speculative arm is faster than its own-tree baseline, all five intervals
excluding zero, with within-prompt run-to-run CV at or below 0.3 %:

| arm | width | tok/s | vs own-tree baseline (95 % CI) | k | tok/J | J/request |
|---|---:|---:|---|---:|---:|---:|
| baseline@master | — | 41.55 | — | — | 0.1005 | 3980 |
| baseline@pr27342 | — | 41.55 | — | — | 0.1005 | 3979 |
| mtp-n2 | 3 | 66.39 | **+59.77 [+56.95, +62.75] %** | 1.4497 | 0.1627 | **2503 (−37.1 %)** |
| mtp-n3 | 4 | 63.29 | +52.32 [+48.47, +56.48] % | 1.7425 | 0.1549 | 2684 |
| dflash2-n4 | 5 | 63.13 | +51.94 [+45.56, +58.17] % | 1.8874 | 0.1554 | 2835 |
| mtp-n5 | 6 | 54.89 | +32.10 [+26.38, +37.75] % | 2.2939 | 0.1343 | 3228 |
| dflash2-n7 | 8 | 50.95 | +22.63 [+14.68, +30.37] % | 2.7156 | 0.1251 | 3786 |

The dual-tree control is exact: the two baselines agree to 41.55 tok/s and produce **byte-identical
output on 125/125 prompt-passes**, so nothing in the DFlash2 comparison is attributable to the
unmerged branch.

**H4a is supported, and the class breakdown matters more than the headline.** `dflash2-n7` is
+22.6 % overall while being a *net loss* on three of five prompt classes (prose −11.1 %,
chat −4.3 %, zh −28.7 %). `mtp-n5` and `dflash2-n4` are within noise of zero on Chinese. A single
overall figure conceals a sign change — the same failure this repo documented in the predecessor's
headline (`docs/METHODOLOGY_AUDIT.md` A1), reproduced here in the opposite direction.

**Cost model, final.** `draft-mtp` (widths 3/4/6): k0 = 0.8934, **c = 0.2806**, r² = 0.9998.
`draft-dflash` (widths 5/8, two points): k0 = 0.7831, **c = 0.2761**. `c` agrees to 1.6 % between
unrelated drafters while `k0` differs by 14 %. Within-arm spread of `k` across five prompt classes
and five passes is 0.35–0.54 %. Independent cross-check: the API counters and llama.cpp's own log
lines agree on **625/625 requests, to the token**.

**H2 is not supported.** Over an acceptance range of 0.096–0.918, the rejection-proportional cost
`r` is at most +0.0028 decode-steps per rejected draft token, with r² between 0.001 and 0.060.
The overhead is charged per position verified, not per draft rejected. This bounds rollback's
contribution; it does not prove rollback is free.

**Losslessness.** Speculative arms are byte-identical to baseline on only 25–30 of 125
prompt-passes — 76–80 % diverge, forking at a median 23 % into the text — but every arm is
**100/100 reproducible across passes**. The divergence is deterministic, not noise. This is
consistent with, and corroborative of, llama.cpp #25618 rather than novel.

---

### CORRECTION 1: the baseline bandwidth elasticity is 0.75, not ~1.0

An earlier entry inferred a bandwidth elasticity of about 1.0 for the no-spec baseline from the
overclock removal: memory +400 MHz was removed and throughput fell 4.1 %. That step was **not a
bandwidth-only lever** — it also removed +100 MHz of core offset and dropped the power limit from
450 W to 420 W. Attributing the whole 4.1 % to bandwidth was wrong.

Phase R's pre-flight measures it properly, moving memory clock alone at a fixed 420 W:

| condition | memory clock under load | tok/s |
|---|---:|---:|
| bw-lo (−800 offset) | 9101 MHz | 40.52 |
| stock | 9501 MHz | 41.87 |
| bw-hi (+800 offset) | 9901 MHz | 43.15 |

That is +4.2 % / −4.2 % of memory clock for +3.1 % / −3.2 % of throughput: **elasticity ≈ 0.75**.

A further subtlety, recorded because it limits even the controlled lever: at a fixed power cap,
raising the memory clock takes power from the core. SM clock under load reads 1922 MHz at stock
against 1886 and 1881 MHz in the two bandwidth arms, so the bandwidth arms are a net of "+4 %
memory, −2 % core" not a pure bandwidth step. Phase R's full run quantifies this; the
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

_(none yet — the two items above are corrections to this document's own interim reasoning, not to
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
its net yield against the MoE baseline is worse than −25 %.

- This runs first in the report for a reason. If the anchor does not reproduce, the difference
  between the two studies is in the harness, the build, or the card, and no Phase M result may be
  read as a statement about MoE until that is resolved. The predecessor's figures were 138.9 tok/s
  baseline against 77.0 tok/s at K=8, a 44.6 % loss, taken at 16384 context on a different 3090.
  This runs at 8192, so exact agreement is not expected and is not the test; the sign and the
  rough magnitude are.

**H6b (the cost model localises the difference).** Fitting k(w) = k0 + c(w−1) on the MoE gives a
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
requests of Phase A, the old form is systematically low: mean gap −0.0204, and the sign never
changes. The corrected form sits at −0.0050, which is inside the log's own `%5.2f` printing
precision.

**Why the existing integrity check did not catch it.** `cross_check_against_log` compared the API
counters against the log's counters and reported 0 mismatches out of 625, correctly, both before
and after this correction. The counters were never wrong. What was wrong was the arithmetic
applied to them, and nothing compared the derived quantity against the server's own value of the
same quantity. That comparison is now part of the check, and it was verified to fire on the old
formula and pass on the new one.

**What it moves.** `k` rises by 0.33 % at n-max 2 to 0.59 % at n-max 7. Because the bias grows
with depth it inflates the fitted `c` by about 0.8 %: `draft-mtp` k0 0.8934 → 0.8937 and
c 0.2806 → 0.2829; `draft-dflash` k0 0.7831 → 0.7825 and c 0.2761 → 0.2784. Both `c` still read
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
