# TODO - Qwen3.8-27B speculative decoding

Rewritten 2026-08-25 03:50 against what has actually run. The original plan is in git history;
it was written before any measurement and several of its spearheads have since been answered,
narrowed, or deleted with a reason.

Three machines are in use and they do not pool. Absolute tok/s belongs to its host; only
dimensionless quantities and within-host deltas cross. See `docs/A6000_PLAN.md` and the
second-host addendum in `PREREGISTRATION.md`.

| host | GPU | toolchain | what it runs |
|---|---|---|---|
| **A** `thc1006-debian13` | RTX 3090, 420 W | CUDA 13.3 | gcc 14.2 | glibc 2.41 | the primary phase chain |
| **B** `3090` (tailscale) | RTX 3090, **350 W** | CUDA 12.0 | gcc 13.3 | glibc 2.39 | cross-host replication, #27572 |
| **C** `mailer.cirda` | **RTX A6000 48 GB** | CUDA 12.9 | gcc 12.2 | glibc 2.36 | forced-warp intervention, Phase Q |

---

## Done

- [x] **Phase A** - 875 measurements, 0 incidents, 0 excluded. Every speculative arm faster,
      every interval clear of zero. MTP n-max 2 at +59.8 % [+57.0, +62.8].
- [x] **Phase R** - 1125 measurements. Bandwidth and power separable on this card, confirmed by
      its own pre-flight. Its compute axis is superseded by R2.
- [x] **Phase R2** - 1575 measurements, 0 incidents, SM clock pinned instead of power-capped.
      The elasticities nearly swap between baseline and speculative arms, and the regime change
      sits between 1200 and 1710 MHz.
- [x] **Phase KV** - the f16 control for the width partition.
- [x] **Cost model** - `k = k0 + c(w-1)`, `c` agreeing to 1.6 % across two unrelated drafters.
      Rejection-cost hypothesis registered and reported unsupported.
- [x] **Correction 3** - the derived `mean_len` was low by a forward pass; caught by comparing
      against the server's own printed value on 625 requests.
- [x] **Correction 4** - H8's third group withdrawn before `phase_nmax` ran: `MMVQ_MAX_BATCH_SIZE`
      is 8, so width 9 never reaches MMVQ and `calc_nwarps` cannot produce that group.
- [x] **Figures** - six, generated from the result files through the same functions the text
      reports use, sized for GitHub's column, light and dark.
- [x] **`--settle-floor` fixed** - it measured the device's idle floor, printed it, and never
      passed it to `settle_gpu`, so the flag raised on every card it was meant for.
      `run_phase_q.sh` and the whole A6000 plan depend on it.

## Running now

- [ ] **Phase n-max** (host A) - MTP 1..8 and DFlash2 2,4,6,8. Delivers the H8/H8a verdicts and
      the `mtp-n1` arm Phase V needs.
- [ ] **#27572 reproduction** (host A, then host B) - the extended sweep reaches the reported
      19 000-token prompts and an `-np` sweep beyond 4, which the first pass never covered.
- [x] **RH2 cross-host replication** (host B) - done. `phase_a_hostB.json`, 175 records: partition
      clean 25/25, groups differ on 14, fork positions identical to host A on all 25 prompts.
- [ ] **Forced-warp replication** (host B, until ~09:36) - the same three builds on the second
      3090. The A6000 result is one device, and the two 3090s are known to agree on fork positions
      where the A6000 does not, so this separates a property of the table from one of that card.
      Host B disappears afterwards; nothing else it was holding is still only there.
- [ ] **Forced-warp intervention** (host C) - three builds of the same revision differing only in
      the `calc_nwarps` GENERIC table. Registered before any of it ran, with the outcomes and the
      baseline identity control written down first.

## Next, in order

- [ ] **Phase C** - drafter quantization. Also decides whether the predecessor's v3.0 needs an
      erratum.
- [ ] **Phase L** - the context-depth ladder to 96 K, against llama.cpp #27623.
- [ ] **Phase M** - dense against MoE under one protocol, anchored on reproducing the
      predecessor's -44.6 %.
- [ ] **Phase Q** - the target-quantization ladder. Needs host C to itself; `UD-Q6_K_XL` and
      `Q8_0` do not fit on 24 GB, and 29 GB free is not enough for `Q8_0` either.
- [ ] **Phase V** - vLLM, matched at K=1, so it waits on the n-max ladder.

---

## Upstream

> `ggml-org/llama.cpp` ships an `AGENTS.md` that binds contributions there. It forbids AI-written PR **descriptions, commit messages and reviewer responses**, and forbids an agent running `git push` or `gh pr create` on the author's behalf; the stated penalty is a contributor ban. Prepare the branch and the body, then run the submit step by hand. Comment rules it enforces: 1-2 lines, no hard-wrapping, never split a sentence across lines, ASCII only.

| item | state | next |
|---|---|---|
| **SGLang #36201** | open, CI gated on `run-ci` | body rewritten; four lookup sites and both kernels fixed; test covers 3 mask modes x 5 outputs; real-op sanitizer clean. Needs someone to trigger CI. |
| **SGLang sampler walk** | not filed | `TreeSpeculativeSamplingTargetOnly` hangs and reads out of bounds on sm_86, both shown. Separate PR: bound the walk **and** range-check `cur_index` and `draft_token_id` - a bound alone does not fix the candidate-id case. Warning must come from one thread; the walk runs on all 1024. |
| **llama.cpp #25618** | commented | the width partition, and the forced-warp result when it lands. Reply on the existing thread rather than opening a new "not lossless" issue. |
| **llama.cpp #27572** | commented | the extended sweep decides whether the CUDA negative holds. Do not claim graph-copy alternation as the root cause. |
| **llama.cpp [#27676](https://github.com/ggml-org/llama.cpp/pull/27676)** | open | the verification-step counter. One line in `server_slot_stats::to_json()` plus assertions in `tools/server/tests/unit/test_speculative.py`. Verified both ways on a CPU-only build of `c060ca9`: unpatched `KeyError: 'draft_n_verif_steps'`, patched `1 passed`. Motivation is exact accounting, not benchmark convenience. |
| **llama.cpp `output_reorder()` gate** | verified, not filed | the `embd_nextn` swap is unconditional but the buffer is token-indexed when `embeddings_nextn_masked` is off. One line, plus a masked/unmasked regression. Does not claim to fix #27572. |
| **llama.cpp #25618 / #27407 / #27623** | tracked | comment counts in `docs/UPSTREAM_CONTRIBUTIONS.md` are dated; re-read before quoting. |

## Deleted from scope, with reasons

Recorded so they are not quietly revived.

- **A public SpecDecode Trace schema.** AIPerf already defines an engine-neutral, tree-agnostic
  `SpecDecodeAcceptanceRecord` with an adapter protocol. Raw llama.cpp artifacts stay immutable;
  issue-specific CUDA and kernel debugging records stay engine-native and are not presented as a
  stable cross-engine format. No AIPerf export is written until something consumes it.
- **A standalone vLLM Qwen3.8 batch-invariance study.** `GDN_ATTN` fails closed under
  `VLLM_BATCH_INVARIANT=1` on the tested commit, so there is no matrix to run. Blocked, not
  refused; a narrow Ampere validation may follow once upstream converges.
- **A custom adaptive speculation controller.** Every measurement is single-stream with constant
  queue depth and batch size, so the data cannot identify a queue- or concurrency-aware policy,
  and repeated passes are repeated measures rather than new serving states. The cost model stays
  what it is: a fixed, single-stream deployment calibration.
- **A public `specdecode-lab` platform.** One engine adapter has been exercised end to end, so
  the abstraction is unvalidated. The harness is modularized internally instead. Extraction waits
  for two engines, several studies, and an external user who needs it.

## Standing constraints

- Never rebuild `llamacpp-master/` or `llamacpp-dflash2/`. Experimental builds go in a separate
  clone; host C's three warp variants are built that way.
- Host B and host C are shared machines. Work lives in one directory that can be removed whole,
  and results come back before anything is deleted.
- Do not ping maintainers. CI on SGLang is triggered by a documented comment command; that is the
  owner's call, not something to do unasked.
