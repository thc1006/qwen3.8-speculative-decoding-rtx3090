# Upstream contribution map

Surveyed 2026-08-24 with `gh` against the live issue trackers, not from search snippets, and
revised twice the same day as reading the actual threads narrowed what this study can claim.
Comment counts and issue states quoted below were read on **2026-08-25**; a tracker moves, so
treat them as a snapshot rather than a current reading.

Ranked by *how directly this study's data answers a question that is still open after the
existing discussion*.

The recurring lesson, recorded because it kept happening: every item here shrank once the
upstream thread was read rather than skimmed. Priority for an observation almost always belongs
to someone already in the thread.

Contribution etiquette: PR #27342's thread already contains an AI-assisted report posted with an
explicit disclosure ("Written by Claude ... posted from the account of the human who ran the
hardware"). Anything filed from this work follows the same convention, and reports only numbers
actually measured on this machine.

---

## 1. Greedy divergence: CLAIM NARROWED TWICE AFTER READING THE THREAD

**What this repo does NOT get to claim.** A first pass at this section treated
[#27407](https://github.com/ggml-org/llama.cpp/issues/27407) (open 2026-08-19, no comments at the
time, RTX 3090 Ti) as an unanswered orphan and this study's divergence data as a fresh
confirmation. Reading the tracker properly shows otherwise. The parent thread is
[#25618](https://github.com/ggml-org/llama.cpp/issues/25618) (open 2026-07-13, **18 comments as of
2026-08-25**), and it already establishes:

- **The phenomenon**, scoped: greedy speculative output diverges from vanilla on **quantized**
  targets while a **bf16 target preserves parity**; ngram speculation stays lossless on the same
  quantized target. Our target is `UD-Q4_K_XL`, i.e. squarely inside the known-affected regime.
  Our divergence numbers are corroboration, not discovery.
- **The mechanism**, already argued in-thread: a batched decode over M positions must be bitwise
  identical to M serial decodes, and is not, so batched verification changes the reduction order.
- **Drafter independence**, already demonstrated: `snick525` built PR #27342 specifically to get
  DFlash2 as a structurally different drafter and showed the divergence persists against the
  built-in MTP head on Qwen3.8-27B Q6_K_XL. That is the same observation this study made
  independently on Q4_K_XL, and they published it first.
- **One root cause, on Vulkan**: `Ankk98` traced an instance to the GQA-packed flash-attention
  path, which treated `N <= 8` as safe to pack and remapped workgroups as if there were a single
  query token, valid only for single-token decode, so multi-token verify drifted, with **wrong
  drafts appearing at window >= 4**. Fixed by gating the packing on `neq1 == 1`.

**The thread moved on 2026-08-21 and 2026-08-24, and it changes what is worth saying.**

ggerganov, 2026-08-21: the numerical difference between computing logits for N tokens at once and
N separate decodes is inherent to batched evaluation and is not a bug; what would be useful is a
reproduction isolating *the width at which it starts, on CUDA*, since the Vulkan case was traced
to the `soft_max` reduction order changing with batch size and nobody had posted the CUDA
boundary.

`frizikk`, 2026-08-24, answered part of it on Vulkan with an operator-level bisect: the first
boundary is a `MUL_MAT` at `linear_attn_out-0`, Q8_0 weight against F32 activation, where `N=1`
takes `mul_mat_vec_q8_0_f32_f32` and `N=2` takes `quantize_q8_1_x4` plus a Q8_0 x Q8_1 MMVQ.
Replaying `N=2` through the `N=1` path made row 0 bit-exact. So on Vulkan the onset is at `N=2`
and the cause is the path switch plus F32-to-Q8_1 staging.

`F-Mangini`, 2026-08-22, showed the same at `n_max=1` on LFM2.5 DSpark, Vulkan, Q8_0 target, and
that an F16 target preserves greedy parity, so a quantised target is necessary.

**What this study has that is not in the thread, stated narrowly.** Not the onset width. Phase A
never ran `n_max=1`, so it cannot say where divergence starts, and on the Vulkan evidence the
answer is already `N=2`. What it has is a *second* boundary, further up, on CUDA: fork positions
on sm_86 partition into exactly two groups by verification width, `{3, 4}` against `{5, 6, 8}`,
identically across five passes, with the partition shared by the target's own MTP head and by an
unrelated 1.1 GB block-diffusion drafter. Every one of those widths diverges; what changes at the
boundary is *where*.

`ggml/src/ggml-cuda/mmvq.cu` has a candidate mechanism at exactly that width.
`ggml_cuda_should_use_mmvq` on this card returns `ne11 <= MMVQ_MAX_BATCH_SIZE`, which is 8, so
every width from 3 to 8 stays inside MMVQ and the kernel family is not what changes. `calc_nwarps`
does change. sm_86 is not RDNA, GCN, CDNA or DGX Spark, and the Turing table is gated on
`arch >= TURING && arch < AMPERE`, so it falls through to `MMVQ_PARAMETERS_GENERIC`:

    ncols_dst 1 to 4  -> 4 warps
    ncols_dst 5 to 8  -> 2 warps
    ncols_dst > 8     -> 1 warp

`ncols_dst` is the verification width, the warp count sets the shape of the summation tree, and
floating-point addition is not associative. That is the same shape of cause as the Vulkan
`soft_max` finding, in a different kernel, and its boundary is where the measurement's is.

**What would make it a claim rather than a coincidence.** `phase_nmax` runs MTP at widths 2
through 9 and is registered as H8 in `PREREGISTRATION.md` before its data exists. The table
forces three groups, `{2, 3, 4}`, `{5, 6, 7, 8}` and `{9}`; widths 2 and 7 were never measured
and are the real test, and width 9 is confounded because it drops to one warp and leaves MMVQ at
the same time. Causation needs the warp count forced constant across widths and the divergence
disappearing, which is a build change and belongs upstream, not in a study that cannot rebuild
its own trees mid-run.

**Filing discipline for this one:** comment on #25618, where the expertise is. Open by agreeing
that batched evaluation is not bit-identical and that this is not filed as a bug. Do not claim
the onset width; `frizikk` has it on Vulkan and this study did not run `n_max=1`. Lead with the
second boundary, the code pointer, and the registered prediction, and say plainly that the
coincidence is not proof. Credit `snick525` for drafter independence, `Ankk98` for the Vulkan
root cause, `frizikk` for the operator bisect and `F-Mangini` for the quantisation dependence.

## 1b. Missing per-request counter in the server API: small, concrete, and self-motivated

Computing the cost of a verification step needs the number of verification steps. The server
tracks it (`server_slot_stats::n_draft_verif_steps`, incremented at `server-context.cpp:3860`)
and uses it for the `mean len` log line and for a Prometheus metric, but
`server_slot_stats::to_json()` exposes only `draft_n` and `draft_n_accepted`, so a consumer of
`/v1/chat/completions` cannot get it.

Two obvious workarounds are wrong in practice:
- `draft_n / n_max` assumes every step drafts the full width; steps that draft fewer break it.
- Recovering it from the log's `mean len` fails on precision: that field prints at `%5.2f`, and
  back-solving `steps = accepted / (mean_len - 1)` from two decimals gave this study a spread of
  +/-0.4 steps and produced physically impossible negative step counts.

A third one looks exact and is not, and finding that out is the strongest argument for the field.

The reasoning is that every verification step emits one non-draft token, so tokens produced are
`verif_steps + accepted` and `verif_steps = predicted_n - draft_n_accepted`. This study used that
form and it is wrong, by one, every time: the first generated token comes out of the
prompt-processing pass rather than a decode forward, so the correct form starts at
`predicted_n - draft_n_accepted - 1`. Nothing looked wrong. The numbers were plausible, stable
across five passes, and consistent with a cost model that fits at r^2 = 0.9998. The existing
integrity check compared the API counters against the log's counters and reported 0 mismatches
out of 625, correctly, because the counters were never the problem.

It was caught only by comparing the derived mean length against the `mean len` the server prints,
where the gap was -0.0204 with a sign that never changed. Correcting it moved the fitted marginal
cost `c` by 0.8 %, recorded as Correction 3 in `PREREGISTRATION.md`.

Even corrected it is approximate. The `- 1` form reproduces the server's printed value on about
70 % of requests; the rest need one step fewer still, which is what truncation at `max_tokens`
looks like, and the API cannot distinguish the two. So there is no exact identity to appeal to,
and the remaining error is under 1 % and irreducible from outside the server.

That is the case for the patch, and it is a better one than convenience. A derived quantity that
reproduces plausible numbers while being quietly wrong by a percent is exactly what an exposed
counter prevents, and the server already keeps the counter.

A patch is prepared at `upstream/0001-server-expose-draft-verification-steps-per-request.patch`
against `c060ca9`. It is one line in `to_json()` and adds no state. It is deliberately **not**
applied to `llamacpp-master/` in this repo: rebuilding that tree mid-study would leave later
phases running a different binary from earlier ones, which is exactly the build confound the
dual-tree design exists to avoid. It gets built and tested in a separate clone once the
measurement queue is clear.

Related but distinct prior requests:
[#26516](https://github.com/ggml-org/llama.cpp/issues/26516) (speculative counters in
`/metrics`) and [#24850](https://github.com/ggml-org/llama.cpp/issues/24850). Both are
**server-wide aggregates**; benchmarking needs the **per-request** value, so this should be
raised as a comment on #26516 not as a new issue.

## 2. The clearest single contribution: the CUDA acceptance-and-cost curve vs n-max (sm_86)

[#26750](https://github.com/ggml-org/llama.cpp/issues/26750) (open 2026-08-08, 2 comments) claims
`draft-mtp` acceptance **collapses on CUDA**: 35.8-40.7 % against ~92 % on Vulkan, same files,
same build, same prompts, turning MTP from +128 % into -32 %. Their setup: `Qwen3.5-9B-Q4_K_M`,
RTX PRO 4000 **Blackwell**, `--spec-draft-n-max 6`, 108 runs per cell, read from
`timings.draft_n` and `draft_n_accepted`, **the same two fields this study reads.**

What their report does not contain is an **acceptance-versus-depth curve on CUDA**, so there is
no way to tell whether 40 % at n-max 6 is a collapse or simply where the curve sits.

This study measures, on sm_86 CUDA with Qwen3.8-27B Q4_K_XL:

| n-max | pooled acceptance |
|---:|---:|
| 2 | 0.665 |
| 3 | 0.558 |
| 5 | **0.412** |

**0.412 at n-max 5 sits inside their 0.358-0.407 band at n-max 6.** Different model and different
CUDA architecture, so this is not a refutation and must not be presented as one. It does supply
the missing control: on a CUDA card where MTP is unambiguously a large *win* (+59.8 % at n-max 2),
acceptance still falls to ~0.41 by n-max 5. That reframes the open question from "why is CUDA at
40 %?" to "why is Vulkan at 92 %?", and 92 % at n-max 6 implies a mean accepted length of
1 + 6x0.92 ~ 6.5, which is high enough to be worth checking, especially since #25618 already
root-caused a **Vulkan** flash-attention packing bug that made multi-token verify disagree with
sequential decode.

Phase N walks n-max 1-8 and produces the full curve, plus the per-step cost k at each depth. That
is a measurement with no interpretive step in it, which is what makes it the most defensible
thing this study has to offer upstream.

## 2b. The n-max disagreement in PR #27342, turned into a measurable coefficient

`lance0` posted RTX 5090 numbers on 2026-08-23, `UD-Q6_K_XL` target, and concluded that n-max 7
is the right DFlash2 setting because the drafter's `block_size` is 8 and lower values discard
tokens the block already paid for. That reasoning is about the drafter and is sound on its own
terms. This study measures the opposite ordering on an RTX 3090 at `UD-Q4_K_XL`: n-max 4 gives
1.520x and n-max 7 gives 1.228x.

Both can hold, and the cost model says what has to differ rather than which report is wrong.
With `speedup = mean_len(w) / (k0 + c(w-1))`, width 8 beats width 5 only when

    mean_len(8) / (k0 + 7c)  >  mean_len(5) / (k0 + 4c)

On the acceptance curve measured here, `mean_len` 2.883 at w=5 and 3.353 at w=8 with k0 = 0.7825,
that requires **c < 0.0543**. Phase A's two DFlash2 points give `c` = 0.2784, 5.1 times over the
threshold; the completed ladder gives 0.2481 over widths 3, 5 and 7, 4.6 times over. Either way the
shallower setting wins here. Phase R2 measures what `c` travels with: over the tested GA102 clock
ranges the baseline responds mostly to the memory clock and the speculative arms mostly to the SM
clock. That is consistent with `c` being dominated by arithmetic, and the 5090 sits further along
that axis, but elasticity is a response measurement and does not establish which resource either
card is bound by.

**Why this is worth posting rather than a correction.** It converts "n-max 7 is right" and "n-max
4 is right" into one number that either card can measure, from one baseline and three widths, and
it predicts the crossover instead of tabulating it. The thread currently has per-card tables and
no way to tell whether a new card should follow the 5090 or the 3090.

**What it assumes, which must be said when posting.** The threshold uses this card's `mean_len`
curve at `UD-Q4_K_XL`. A Q6_K target may accept more, which raises `mean_len` at depth and lowers
the required `c`. That confound is not resolved here; Phase Q walks the target ladder to separate
`c` from acceptance, and until it runs the honest statement is that the two reports differ by at
least a factor in `c`, target quantisation uncontrolled.

**Filing:** a comment on #27342 addressed to `lance0`'s numbers, crediting the `block_size`
argument rather than disputing it, and leading with the coefficient rather than with this card's
ordering. The 5090 result is more interesting than this one; it is the second point that makes
the model testable.

## 3. Energy: no prior art at all

`joule`, `tok/J` and `watt` appear **zero** times across PR #27342's 60-comment thread, and no
study in the prior-art sweep publishes an energy figure for this model. Measured here, decode-only
with prefill energy subtracted:

| arm | tok/J | J per 400-token request | vs baseline |
|---|---:|---:|---:|
| baseline | 0.1005 | 3980 | - |
| mtp-n2 | 0.1625 | **2506** | **-37 %** |
| dflash2-n4 | 0.1555 | 2833 | -29 % |
| mtp-n5 | 0.1344 | 3224 | -19 % |
| dflash2-n7 | 0.1253 | 3785 | -5 % |

Not a bug report; a data contribution to PR #27342 and to the community tables, where "is it
worth enabling" is currently answered on throughput alone.

## 4. The cost model, as an explanation not a table

`speedup = mean_len / k` with `k(w) = k0 + c*(w-1)` fits to r^2 = 0.9958 for the built-in MTP
head over widths 2-8 and 0.9947 for the DFlash2 drafter over 3, 5 and 7, on the completed n-max
ladder. It explains *why* deep drafting stops paying, in a form that predicts an optimum rather
than tabulating one.

An earlier two-point fit on Phase A put the two marginal costs within 1.7 % of each other, and
this section used to read that as the per-position cost belonging to the verification path and the
fixed cost to the drafter. That reading is withdrawn on a technicality worth stating: Phase A fits
DFlash2 on widths 5 and 8 and MTP on 3, 4 and 6, which **share no width at all**, and `k(w)` is
curved, so the two numbers are chords of disjoint arcs. `cost_model.py` now refuses that
comparison rather than printing it.

The completed ladder shares widths 3, 5 and 7. Fitted there, `c` is **0.2954** for MTP against
**0.2481** for DFlash2 and the difference is **-0.0473 [-0.0489, -0.0456]**. The two `k(w)` curves
differ by a straight line to within 2.4e-4, so whatever curvature they carry is shared and cancels
out of the comparison. Part of the marginal cost moves with the drafter.

`k0` is 0.8888 against 0.9443. Both sit below 1.0, which is the floor a zero-depth cycle must
cost, so neither intercept is a measured fixed cost and the pair should not be read as an
attribution.

Relevant to [#25187](https://github.com/ggml-org/llama.cpp/issues/25187) (FR-Spec draft-vocab
trimming research), which is about reducing drafter cost, meaning `k0` rather than `c`.

## 5. Multimodal: confirm and extend, on Linux and on the other drafter

[#27408](https://github.com/ggml-org/llama.cpp/issues/27408) (2 comments) reports that with
`--spec-type draft-dflash` plus an mmproj, **every image request stalls ~500 s and returns HTTP
500**, root-caused to mtmd image chunks leaving a positional hole in the draft KV cache. Reported
on **RTX 3090 Ti, sm_86, Windows**, the same architecture as this study on the other OS.

Two gaps in that report this study can close cheaply: whether it reproduces on **Linux/sm_86**,
and whether **`draft-mtp`**, the built-in head that needs no separate draft context, is
affected at all. They tested only `draft-dflash`.

## 6. Confirmations, ranked by what they add

| target | what this study adds | status |
|---|---|---|
| [#25618](https://github.com/ggml-org/llama.cpp/issues/25618) divergence | the CUDA width boundary (see section 1) | Phase A + Phase N |
| [#27623](https://github.com/ggml-org/llama.cpp/issues/27623) ~25x decode cliff past ~80 K, 1 comment | reproduction on sm_86, and whether speculation survives it | Phase L |
| [#27572](https://github.com/ggml-org/llama.cpp/issues/27572) acceptance -> 0 under `-np N`, 10 comments, 3 of them this study's | sm_86 confirmation; also closes the predecessor repo's own untested concurrency caveat | Phase X |
| [vLLM #52475 / #53180](https://github.com/vllm-project/vllm/issues/52475) degenerate output on hybrid GDN | the baseline-relative degeneracy screen used here is the methodology those reports need | Phase V |

---

## 7. SGLang: two unbounded device-side walks, one of them submitted

Not llama.cpp, and not in this table until now, which was an omission: the work exists in
`upstream/sglang/` as two patches, two findings write-ups and a reproducer set, and one of the two
changes has been open upstream since 2026-08-24.

The entry point was [sglang#35822](https://github.com/sgl-project/sglang/issues/35822), a hang in
`tree_speculative_sampling_target_only` with native Qwen3.5/3.8 MTP. The first guess -- that it
duplicates the known EAGLE tensor-parallel divergence -- is wrong, and the versions say so: that
bug is in the **greedy** branch, and on the reporter's v0.5.17 the **sampling** branch, which is
what their py-spy stack names, already broadcasts.

What is actually there is two `while (cur_index != -1)` walks over `retrive_next_sibling` with no
other exit and no check that `cur_index` is a position in the request's row before it is used as
one. `VerifyTreeGreedy` in `eagle_utils.cu` and `TreeSpeculativeSamplingTargetOnly` in
`speculative_sampling.cuh`. Both were confirmed non-terminating on sm_86 rather than argued for
from the source.

| what | where | state |
|---|---|---|
| Ancestor walk bounded, and stopped when the ancestor is absent | [sglang#36201](https://github.com/sgl-project/sglang/pulls/36201), +297/-32 | **open**, awaiting a maintainer to apply `run-ci`, which [#31478](https://github.com/sgl-project/sglang/issues/31478) is also waiting on |
| Sibling walks bounded, with a range check and a vocabulary check | `upstream/sglang/0002-bound-sibling-walks.patch` | **held back on purpose**: [#35771](https://github.com/sgl-project/sglang/issues/35771) is already open against that kernel's accept condition, and a second change to the same lines would collide |

One thing worth repeating from `upstream/sglang/FINDINGS.md`, because it is the kind of error a
test suite does not catch. A `PARENT_VALID` fixture for the second request was copied from the
first without accounting for its `selected_index` being `[2, 4, 0]` rather than `[4, 2, 0]`, so
its ancestor resolved to itself: `test_valid_chain` was testing a third looping tree, and the PR
body would have claimed a before-equals-after result that does not hold. Caught in the last check
before opening.

## What this study CANNOT contribute, stated so nobody spends time on it

- **The quantization axis of the divergence.** #25618 establishes that a bf16 target preserves
  greedy parity while quantized targets do not. Testing that here would need a less-quantized
  Qwen3.8-27B: `UD-Q6_K` is 25.3 GB, `Q8_0` 29 GB, BF16 50 GB, so none fit in 24 GB alongside a KV
  cache. The decisive control for that question is not available on this hardware and this study
  does not attempt it.
- **`draft-eagle3`.** Accepted by `--spec-type` in this build, but no EAGLE3 drafter has been
  published for Qwen3.8-27B (HF hub, checked 2026-08-24). Worth stating publicly so others stop
  looking; there is nothing to benchmark.
- **Multi-GPU / tensor-parallel.** Several open issues (#27366, #27577, #26339, and the
  `-sm tensor` reports in PR #27342) are multi-GPU. Single card here.
