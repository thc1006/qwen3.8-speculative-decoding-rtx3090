# Upstream contribution map

Surveyed 2026-08-24 with `gh` against the live issue trackers, not from search snippets, and
revised twice the same day as reading the actual threads narrowed what this study can claim.
Ranked by *how directly this study's data answers a question that is still open after the
existing discussion*.

The recurring lesson, recorded because it kept happening: every item here shrank once the
upstream thread was read rather than skimmed. Priority for an observation almost always belongs
to someone already in the thread.

Contribution etiquette: PR #27342's thread already contains an AI-assisted report posted with an
explicit disclosure ("Written by Claude … posted from the account of the human who ran the
hardware"). Anything filed from this work follows the same convention, and reports only numbers
actually measured on this machine.

---

## 1. Greedy divergence: CLAIM NARROWED TWICE AFTER READING THE THREAD

**What this repo does NOT get to claim.** A first pass at this section treated
[#27407](https://github.com/ggml-org/llama.cpp/issues/27407) (open 2026-08-19, zero comments,
RTX 3090 Ti) as an unanswered orphan and this study's divergence data as a fresh confirmation.
Reading the tracker properly shows otherwise. The parent thread is
[#25618](https://github.com/ggml-org/llama.cpp/issues/25618) (open **2026-07-13, 16 comments**),
and it already establishes:

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

**What is left, and it is narrow but real.** That root cause is a **Vulkan** path. Nobody has
localised the analogous threshold on **CUDA**. This study's Phase A shows fork positions on
sm_86 partitioning into exactly two groups by verification width, `{3, 4}` against `{5, 6, 8}`,
with the partition shared by two unrelated drafters, placing a CUDA boundary between width 4
and width 5. That is close to, but not the same as, the Vulkan `>= 4` window condition, which is
what makes it worth pinning down rather than assuming they are the same bug.

Phase N walks MTP over widths 2–9 and DFlash2 over 3/5/7/9 precisely to state the CUDA boundary
as "between width W and W+1" instead of "somewhere". A precise boundary on a backend whose
sibling bug has already been root-caused is a usable pointer to a specific kernel; a vague one
is a fourth report that output differs.

**Filing discipline for this one:** comment on #25618 (the parent, where the expertise is), not
on #27407; lead with the CUDA width boundary; explicitly credit `snick525` for drafter
independence and `Ankk98` for the Vulkan root cause rather than restating either as new.

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
  ±0.4 steps and produced physically impossible negative step counts.

A third one is exact, and stating it honestly makes this a smaller contribution than the two
failures above suggest. Every verification step emits one non-draft token, so the tokens produced
are `verif_steps + accepted`, and `verif_steps = predicted_n - draft_n_accepted` follows. That
identity is what this study actually uses, and it is exact rather than approximate. The case for
the field is therefore convenience and durability, not necessity: the identity depends on every
verification step emitting exactly one non-draft token, which is an implementation detail that
no documentation guarantees and that a future scheduling change could break silently, since the
arithmetic would keep producing plausible numbers.

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
`draft-mtp` acceptance **collapses on CUDA**: 35.8–40.7 % against ~92 % on Vulkan, same files,
same build, same prompts, turning MTP from +128 % into −32 %. Their setup: `Qwen3.5-9B-Q4_K_M`,
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

**0.412 at n-max 5 sits inside their 0.358–0.407 band at n-max 6.** Different model and different
CUDA architecture, so this is not a refutation and must not be presented as one. It does supply
the missing control: on a CUDA card where MTP is unambiguously a large *win* (+59.8 % at n-max 2),
acceptance still falls to ~0.41 by n-max 5. That reframes the open question from "why is CUDA at
40 %?" to "why is Vulkan at 92 %?", and 92 % at n-max 6 implies a mean accepted length of
1 + 6×0.92 ≈ 6.5, which is high enough to be worth checking, especially since #25618 already
root-caused a **Vulkan** flash-attention packing bug that made multi-token verify disagree with
sequential decode.

Phase N walks n-max 1–8 and produces the full curve, plus the per-step cost k at each depth. That
is a measurement with no interpretive step in it, which is what makes it the most defensible
thing this study has to offer upstream.

## 3. Energy: no prior art at all

`joule`, `tok/J` and `watt` appear **zero** times across PR #27342's 60-comment thread, and no
study in the prior-art sweep publishes an energy figure for this model. Measured here, decode-only
with prefill energy subtracted:

| arm | tok/J | J per 400-token request | vs baseline |
|---|---:|---:|---:|
| baseline | 0.1005 | 3980 | — |
| mtp-n2 | 0.1625 | **2506** | **−37 %** |
| dflash2-n4 | 0.1555 | 2833 | −29 % |
| mtp-n5 | 0.1344 | 3224 | −19 % |
| dflash2-n7 | 0.1253 | 3785 | −5 % |

Not a bug report; a data contribution to PR #27342 and to the community tables, where "is it
worth enabling" is currently answered on throughput alone.

## 4. The cost model, as an explanation not a table

`speedup = mean_len / k` with `k(w) = k0 + c·(w−1)` fits to r² = 0.9998 on the built-in MTP head,
and the marginal cost `c` agrees to 1.7 % between MTP (0.2803) and the structurally unrelated
DFlash2 drafter (0.2757) while the fixed cost `k0` differs by 14 %. That says the per-position
cost belongs to the verification path and the fixed cost belongs to the drafter, and it explains
*why* deep drafting loses, in a form that predicts the optimum instead of tabulating it.

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
| [#25618](https://github.com/ggml-org/llama.cpp/issues/25618) divergence | the CUDA width boundary (see §1) | Phase A + Phase N |
| [#27623](https://github.com/ggml-org/llama.cpp/issues/27623) ~25× decode cliff past ~80 K, **0 comments** | reproduction on sm_86, and whether speculation survives it | Phase L |
| [#27572](https://github.com/ggml-org/llama.cpp/issues/27572) acceptance → 0 under `-np N`, **0 comments** | sm_86 confirmation; also closes the predecessor repo's own untested concurrency caveat | Phase X |
| [vLLM #52475 / #53180](https://github.com/vllm-project/vllm/issues/52475) degenerate output on hybrid GDN | the baseline-relative degeneracy screen used here is the methodology those reports need | Phase V |

---

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
