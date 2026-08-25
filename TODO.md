# TODO - Qwen3.8-27B speculative decoding

Rewritten 2026-08-25 03:50 against what has actually run. The original plan is in git history;
it was written before any measurement and several of its spearheads have since been answered,
narrowed, or deleted with a reason.

Three machines are in use and they do not pool. Absolute tok/s belongs to its host; only
dimensionless quantities and within-host deltas cross. See `docs/A6000_PLAN.md` and the
second-host addendum in `PREREGISTRATION.md`.

| host | GPU | toolchain | what it runs |
|---|---|---|---|
| **A** `thc1006-debian13` | RTX 3090, 420 W | CUDA 13.3 / gcc 14.2 / glibc 2.41 | the primary phase chain |
| **B** `3090` (tailscale) | RTX 3090, **350 W** | CUDA 12.0 / gcc 13.3 / glibc 2.39 | cross-host replication, #27572 |
| **C** `mailer.cirda` | **RTX A6000 48 GB** | CUDA 12.9 / gcc 12.2 / glibc 2.36 | forced-warp intervention, Phase Q |

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

- [x] **Phase n-max** (host A) - complete, 1050 records. - MTP 1..8 and DFlash2 2,4,6,8. Delivers the H8/H8a verdicts and
      the `mtp-n1` arm Phase V needs.
- [x] **#27572 reproduction** - complete on both hosts; host B's 21 result files are under `repro/hostB/`. - the extended sweep reaches the reported
      19 000-token prompts and an `-np` sweep beyond 4, which the first pass never covered.
- [x] **RH2 cross-host replication** (host B) - done. `phase_a_hostB.json`, 175 records: partition
      clean 25/25, groups differ on 14, fork positions identical to host A on all 25 prompts.
- [x] **Forced-warp replication** (host B) - complete; it reproduced the A6000's pattern, including the failure now attributed to a cmake reconfigure. - the same builds on the second 3090.
      control done, forced-up running, then the original forced-down and then forced_down2. The
      A6000 result is one device and the two 3090s agree on fork positions where the A6000 does
      not, so this separates a property of the table from one of that card.
- [ ] **forced_down2** (host C now, host B after its chain) - forced-down was void because its
      1-4 row includes width 1, and a drafter decodes one token at a time, so it perturbed every
      arm through its drafter. SASS hashing shows both original edits were surgical in the binary
      (1-4 identical for forced-up, 5-8 identical for forced-down), which is what forced the
      explanation. forced_down2 splits the row and leaves 1,2 at four warps. Predictions are
      registered in PREREGISTRATION.md Correction 6, written before the run finished.
- [ ] **Truncation, extended-cap run** - not started; the design is verified and needs no code.
      `harness/truncation_audit.py` finds 15 of 25 prompts in phase_a holding a 'same' the
      400-token cap could have produced. The partition survives on the 10 that do not, so the
      grouping does not depend on generation length, but the other 15 are unrecovered.

      Run it as `bench.py --matrix <same> --max-tokens 1600` on the host that produced the file.
      Nothing else changes. Three things were checked before writing this down:

      - `max_tokens` is applied at the single measured request, the same line for every arm, so
        the baseline is extended too. Comparing a 1600-token arm against a 400-token baseline
        would manufacture a fork at the baseline's end, and that is the one mistake that would
        make the whole run worse than useless.
      - divergence refers to `(baseline, prompt, pass)` from the same run, so the reference is the
        baseline that ran under the same cap.
      - 1600 tokens because the densest prompt, `zh_tea`, is 1.37 chars/token and that gives it
        2188 characters against the 1537-character study threshold. Every other prompt clears it
        by more.

      The control is free: divergence is deterministic here, 150 of 150 arm-by-prompt cells agree
      across all five passes of phase_a, so every already-resolved fork must come back at the same
      character. One that does not means something other than the cap changed.
- [x] **Forced-warp intervention** (host C) - forced-up stands; forced-down withdrawn. See Correction 8. - three builds of the same revision differing only in
      the `calc_nwarps` GENERIC table. Registered before any of it ran, with the outcomes and the
      baseline identity control written down first.

## Next, in order

- [x] **Phase C** - complete 2026-08-25 10:43, 750/750, 0 incidents. Drafter quantization barely
      matters and the highest precision is the slowest: q8 +53.4 %, q4k +52.0 %, bf16 +48.5 %
      against baseline, so a bf16 drafter costs about five points to run. The class effect is
      larger than the quantization effect: code +117 %, reason +90 %, zh +0.8 %. Both baselines
      agree to 0.01 tok/s across the two trees.

      The three n-gram arms are three different failures and the drafting counters separate them.
      `ngram-mod` has `t_draft_n = 0` on all 75 records and output byte-identical to baseline on
      all 75: the flag was accepted and did nothing, which is the predecessor's
      `draft-qwen3-0.6b` failure exactly, caught this time because the guard records instead of
      asserting. `ngram-map-k` drafts on 6 of 75, 288 tokens for 24 accepted. `ngram-cache` drafts
      on 63 of 75, 9699 tokens, and accepts none of them, so its -8.3 % is drafting cost with no
      return. For contrast DFlash2 accepts 41.1 % and draft08b-n8 21.0 %, the latter below n4's
      35.2 % and slower for it. That `ngram-mod` is silently inert is worth an upstream issue.
- [ ] **Phase L** - the context-depth ladder to 96 K, against llama.cpp #27623. Started 10:43:24;
      rung 1 of 5 at 11.9 s/record. Rungs 8192, 32768 and 65536 sit below the reported ~80 K
      cliff, so the fourth rung is what decides whether the ladder takes four hours or thirteen.
      `.ladder_budget_s` holds a seconds count that stops it at a rung boundary.
- [ ] **Phase M** - dense against MoE under one protocol, anchored on reproducing the
      predecessor's -44.6 %.

      Audited before it runs, since the n-gram guard already showed one bad arm can stop a phase.
      Two risks checked, one cleared and one open.

      Cleared: the MoE carries a real MTP block. `blk.40.nextn.{eh_proj,enorm,hnorm,
      shared_head_norm}` matches the dense target's `blk.64` set exactly, so the five `moe-mtp-*`
      arms will not repeat `ngram-mod` and come out as silent baseline duplicates.

      Open: the three `moe-draft08b-*` arms put a second model on the card with `-ngld 99`, and
      the margin is thin. llama.cpp ignores the MTP block's attention and expert tensors - the
      server log says so for the dense model, `blk.64.attn_q.weight ... ignoring` - which for this
      MoE is 0.48 GiB of `blk.40` never reaching VRAM, so the load is 20.79 rather than 21.27 GiB.
      With the 0.50 GiB drafter and 0.34 GiB of q8_0 KV at 8192 that leaves about 0.57 GiB for the
      compute buffer at 1.8 GiB, and nothing fits above about 2.4 GiB. It cannot be settled
      without the card: no server log here records a buffer allocation, and the MoE has never been
      loaded on this host. `assert_capacity` compares against total VRAM, 24.0 against the
      matrix's 23.8, so it passes and will not pre-empt an OOM at load.

      If it does OOM, `run_phase phase_m ... || exit 1` stops the chain, the 22 GB MoE is never
      deleted, and Phase Q never gets the disk it needs. The monitor greps server logs for `out of
      memory`, so it surfaces immediately rather than being found in the morning.
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

---

## Audit response, opened 2026-08-25 07:10

An external source-and-inference audit graded the repo A- on discipline and throughput, and C- to
D on mechanism attribution. Every item below was checked against the data before being accepted;
two of its claims did not hold and are recorded as such. The user authorised full re-runs where
the fix needs them.

### A. Truthfulness of what is already written

- [x] **A1** `c` no longer attributed to target verification. It is the whole cycle - verification,
      the drafter's own forwards, sampling, launch, synchronisation, output extraction, state
      management - and two drafters sharing all but one of those narrows it without identifying it.
- [x] **A2** the rollback bound scoped to the component proportional to `n_max(1 - acceptance)`.
      A fixed per-step checkpoint or per-rejection restore lands in `k_verify` and is invisible to
      a slope against acceptance.
- [x] **A3** "`c` is a compute cost" downgraded to SM-clock-sensitive. Clock elasticity also moves
      with the voltage-frequency curve, power headroom, occupancy and launch amortisation.
- [x] **A4** the 5090 paragraph relabelled a sensitivity threshold. It was already conditional on
      holding acceptance fixed; the sentence before it was not.
- [x] **A5** the -37 % decode energy marked provisional. `power.draw` and `power.draw.average`
      return the same number on this card, so the sampler integrates one-second averages at 10 Hz.
- [x] **A6** class and language scoped to this suite. `think=True` is 5 of 5 reason prompts and 0
      of the other 20, so thinking is collinear with the class, and the Chinese prompts are
      different tasks rather than translations.
- [x] **A7** the warp section carries the intervention result: forced-up passes every gate and the
      registered prediction held on 3 of 18 discriminating prompts.
- [x] **A8** done - the docstring describes the design that exists, names the think-class collinearity and the language-task confound, and points at D5. It said "3 x 5 = 15" for a 25-prompt suite and claims
      `think` is crossed with class. Rewrite to describe the design that exists and name the
      collinearity as a limitation.
- [x] **A9** done - downgraded to a described observation that also says what it does not measure. It asserted "compute-bound verify" as a conclusion. Same downgrade.

Two audit claims that did not hold: the README never says "compute-bound" (the phrase is in
`elasticity.py` and, correctly, in the preregistration as the hypothesis under test), and it did
not assert warp causation - it stated co-occurrence and called the CUDA boundary open. The gap
there was omission, not misstatement.

### B. Analyser correctness, no re-run needed

- [x] **The cache check tested one invariant and reported it as the other** - `bench.py` fired on
      any `t_cache_n > 0` with "despite cache_prompt=False" as a constant string. `phase_l` sets
      `CACHE_PROMPT = True` deliberately, so it raised one incident per request against a
      condition it never claimed: 45 records, 45 incidents when this was caught, and 900 over the
      full ladder, enough to bury a real one. Fixed in 8143dd3 to check whichever direction the
      matrix declared, and a miss under `CACHE_PROMPT = True` is now its own incident, which is
      the one the deep rungs need since it means the KV cache was evicted. Nothing measured moved:
      exclusion is decided per record by `analyze._usable()` and never reads this list, and the
      energy path had already taken its `cache_prompt` branch. Rung 1 was mid-flight and python
      imports once, so `harness/repair_cache_incidents.py` clears its 180 and keeps them under
      `incidents_repaired`.

- [x] **B1** `width_groups.py` mapped width 9 to one warp. `MMVQ_MAX_BATCH_SIZE` is 8, so that
      width never reaches MMVQ and the table predicts nothing for it. H8 now reports NOT TESTABLE
      for off-path widths and scores the rest.
- [x] **B2** `cost_model.py` fitted one line across the MMVQ boundary, dragging the MTP
      coefficient from 0.2904 to 0.2215 and the fit from r2 = 0.9958 to 0.8304.
- [x] **B3** `analyze.py` still used `mean_len = n / (n - accepted)`, the form `cost_model.py` was
      corrected away from. Two mean lengths in one repo.
- [x] **B4** `bench.py` decode tok/J used `predicted_n` against an energy figure with the first
      token subtracted out.
- [x] **B5** five algebraic invariant tests, each verified by reintroducing its defect on a copy.
- [x] **B6** `ascii_sweep.py` put a pipe inside markdown table cells and broke five rows.
- [x] **B7** `truncation_audit.py` measured in characters and reported 15 of 25 Phase A prompts as
      censored while the rest were clean. That was the unit. The design fixes the window in tokens,
      and characters per token run 1.36 to 6.17 across this suite, so in tokens every record has the
      same 400-token window and the differential censoring does not exist. The "cleaner subset"
      robustness check both analysers had grown is removed, because there is no cleaner subset.
- [x] **B8** three states, read off `finish_reason` rather than inferred from a threshold:
      diverged at token t, identical through EOS, right-censored at the cap. **5825 of 5825 records
      across every file stopped at the cap and none reached EOS**, so every identical verdict in
      the study is right-censored, uniformly. Forks resolve as late as token 334 of 400 in Phase A
      and 379 of 400 in the A6000 warp control.
- [x] **B9** pass agreement asserted rather than assumed: 150 cells in Phase A and 300 in n-max are
      measured more than once and all agree.
- [x] **B11** the repair recorded in `phase_a.json` verified rather than trusted: no measured field
      differs from `phase_a.pre_repair.json`, exactly 150 records gained a divergence and all are in
      pass 5, and recomputing those 150 independently disagrees on none. The thermal gate is also
      not a no-op: 34 of 35 arm-passes waited, median 30 s, GPU at stock throughout.
- [x] **B12** found while looking for a clock confound, and it runs the other way: every speculative
      arm boosted 2.0 to 4.2 % **lower** than its own baseline and ran 3 to 5 degrees hotter,
      because it draws more power for the same wall time. A treatment slower than its control
      deflates the effect. At matched clock `mtp-n2` would be about +64.7 % rather than +59.8 %.
      The measured figure stays the headline as the conservative one.
- [x] **B10** `analyze.py` excluded records that did not hit the cap without saying so.
      Speculation moves where a request stops - 76 to 80 % of these diverge from their baseline -
      so that selects on a post-treatment variable. `build_series_itt` applies no exclusion and
      `report()` now prints both counts every run. On Phase A they are the same 875, so the
      headline never depended on it; it starts to matter at D2's larger budget.
- [x] **B13** the log cross-check in `cost_model.py` skipped an arm-pass silently when its line
      count did not match, which turns the check off rather than failing it. It now names what it
      skipped. Verified running on Phase A: 625 requests compared, 0 mismatched, and the derived
      mean length tracks the server's printed one to within its printing precision.
- [x] **B14** acceptance checked end to end. It is parsed from the server log into
      `arm_pass_acceptance`, which holds 26 entries per arm-pass against 25 prompts because the
      drafter-evidence request runs first. The consumer requires exactly one extra and zips from
      index 1, so it is aligned; the cost model takes acceptance from the per-request timings
      regardless.

- [x] **B15** the request path verified against the claims made about it. `cache_prompt: false`
      is sent and `t_cache_n` is 0 on all 875 Phase A records. The sampler chain is pinned
      explicitly - `top_k` 1, every penalty neutral, mirostat off - so "greedy" means the same
      thing in every arm. The compared text is `reasoning_content + content`, so a divergence
      inside a thinking block is seen and the characters-per-token conversion is over the whole
      generation.
- [x] **B16** the two trees' baselines coincide, which every cross-tree comparison depends on:
      **125 of 125 byte-identical**, decode rate apart by -0.02 %, paired bootstrap
      -0.008 % [-0.029, +0.012], spanning zero.
- [x] **B17** prefill energy is subtracted. The `subtracted` flag is None on Phase A because the
      flag was added after that run, but `decode_energy_j` is below the request total on all 875
      records at a median ratio of 0.977, and phase_nmax carries the flag as True at 0.976. About
      2.3 % of request energy is prefill, so the -37 % figure is computed on decode energy.
- [x] **B18** power sampling density: 38 samples minimum, 71 median per record, so no energy
      integral rests on a handful of points. The effective rate is 0.85 of the nominal 10 Hz
      because each `nvidia-smi` call takes longer than the interval; the trapezoid handles the
      uneven spacing, and the one-second averaging behind `power.draw` remains the reason C3
      moves to the counter.

### C. Harness design, requires a full re-run

- [x] **C1** `--shuffle-prompts`: a seeded permutation per pass, identical across arms within
      it, different between passes, with `prompt_order_by_pass` and a per-record `ordinal` in the
      result. **Off by default and it has to be**: phase_a, phase_r, phase_r2, phase_kv and
      phase_nmax all ran under the fixed order, so a default that permuted would make phase_l
      onwards a different experiment from the phases it is compared against. It is for D4.
- [x] **C2** `--latin-arms` runs `len(arms)` passes so the rotation closes and every arm visits
      every position exactly once. Also off by default, since it changes how many passes a matrix
      runs. With the ordinal now recorded, the other route - adjust for position in the model
      rather than balance it by design - is open too.
- [x] **C3** done, by a different route than the audit proposed. There is no total-energy
      counter reachable from here: `nvidia-smi` rejects `total_energy_consumption` on this driver
      and neither `pynvml` nor any nvidia python package is installed, so the counter would mean a
      new dependency in a harness that deliberately has none.

      `power.draw.instant` does the job instead. Querying `power.draw` beside `power.draw.average`
      returns the same number on every sample, which is the averaging the audit named; `instant`
      differed on all 20 samples under load and carried 58 % more spread. Both are now sampled and
      both integrated, so a file has the figure the earlier phases used and the sharper one, and
      nothing becomes incomparable.

      It also bounds what the averaging was worth: over a three-second window on a loaded card the
      two integrals differ by **0.17 %**. The other half of the criticism - that the integral does
      not cover the gap between the request starting and the first sample - is untouched by this
      and remains open.
- [x] **C4** `harness/kernel_facts.py` reads the dispatch facts out of the tree that is about to
      run - `MMVQ_MAX_BATCH_SIZE` from `mmvq.cuh`, the GENERIC arm of `calc_nwarps` parsed as a
      case table, and the sha256 of `libggml-cuda.so` rather than of the 17 KB launcher - and
      `bench.py` records them per run. `cost_model.py` and `width_groups.py` prefer the recorded
      limit and print which source they used, so a run from before this existed says so instead
      of being described with today's constant. Three defects this study shipped were exactly a
      kernel fact written into Python and later untrue; this is the fix for the class rather than
      for the instances. Twenty tests, including one that the GENERIC switch has no case 9.
- [~] **C5** same-token replay. **The full version needs an upstream feature**: llama-server has
      no teacher-forcing or scoring mode, only free generation, so nothing in the harness can make
      a speculative arm replay the baseline's tokens. Like E3, that is an issue to raise rather
      than code to write.

      The part that can be done here is done. A record that came out byte-identical shares its
      whole trajectory with its baseline, and fitting `c` on those alone is the same-trajectory
      comparison. It gives 0.2898 against 0.2904 for `draft-mtp` and 0.2476 against 0.2481 for
      `draft-dflash`, 0.2 % in both, so divergence does not move the coefficient. `cost_model.py`
      prints it every run. The threat was real in principle and does not bite here; the replay
      mode would still be needed to say the same about anything measured *after* a fork.

### D. Re-runs, in dependency order

- [x] **D1** `forced_down2` ran on both hosts and both are **withdrawn**. Its gates failed at
      widths whose CUDA machine code is byte-identical to the control's, and the cause is now
      known: its build log begins with a cmake configure and the control's does not. A reconfigure
      regenerates `flags.make` for every target, all eleven `ggml-base` sources recompiled, and
      `libggml-base.so` came out different - `.text` and `.rodata`, 8813 bytes, not metadata.
      Six full rebuilds of the unchanged tree give one hash, the control's. See Correction 8.
- [ ] **D1b** the intervention re-run with all four builds from one configure, running on host C.
      control, forced-up, forced_down2 and a second control, built back to back and run back to
      back. Three build gates stop it before it can produce numbers: no build may reconfigure,
      `libggml-base.so` must be identical across all four, and the two controls must be
      byte-identical in `libggml-cuda.so` while both forced builds must differ from them.
- [ ] **D2** extended cap. Design verified, no code change: `--max-tokens 1600` on the host that
      produced the file. 1600 because the densest prompt is 1.37 chars/token against a
      1537-character threshold. The control is free: divergence is deterministic, 150 of 150.
- [ ] **D3** Phase B - `n_max` crossed with `p_min`. The only design here that can separate
      drafted volume from rejection volume, and therefore the only one that can identify the
      rollback components A2 leaves open.
- [ ] **D4** full re-run of Phase A under the C1/C2/C3 harness, once those land.
- [ ] **D5** factorial prompts: class crossed with thinking, language crossed with matched task,
      short and medium output lengths alongside the 400-token regime.

### E. Upstream

- [x] **E1** llama.cpp output-row / token-row index-space fix, prepared as
      `upstream/llamacpp/0002-output-reorder-index-space.patch` (155 lines, DCO, `libllama.so`
      compiles). `output_swaps` is built from output-row ordering and `output_reorder()` applied
      it to two buffers that are not indexed that way: `embd_nextn` when
      `embeddings_nextn_masked` is off, and `embd_layer_inp` in every mode. A test showing that
      interleaved sequences give unsorted `out_ids` - the state that makes `output_reorder()` do
      any work at all - is added to `tests/test-batch-alloc.cpp` and passes, 206 assertions in
      that file, 0 failures. **Source-level proof, not a runtime reproducer**: no scrambled
      embedding has been observed. Submitting is the author's step; `AGENTS.md` there forbids an
      agent pushing or opening a PR.
- [x] **E2** SGLang consumer-side sibling hardening, prepared as
      `upstream/sglang/0002-bound-sibling-walks.patch` plus a seven-case test matrix in
      `test_speculative_sampling_malformed.py`: self-loop, two-node cycle, out-of-row first hop,
      out-of-row sibling, a negative that is not the -1 sentinel, and a candidate id above and
      below the vocabulary, against a well-formed tree pinned as a regression.

      The audit asked for a per-request status buffer instead of the device printf. Not taken:
      upstream already reports this exact class by printf in four places with the wording this
      patch matches, there is no status-buffer pattern in these kernels, and adding one means
      changing the kernel signature, the launcher and the Python binding. The concern behind it -
      a printf from all 1024 threads - is already handled by the `tx == 0` guard, and
      `VerifyTreeGreedy` launches `dim3 block(1)`.

      **Not yet run**: it needs a free GPU and the card is mid-benchmark.
- [ ] **E3** llama.cpp acceptance histogram. **Do not write the PR.** The counters exist -
      `slot.n_accepted_per_pos[i]` is incremented for every `i < n_accepted`, so it is a survival
      count: steps that accepted at least i+1 - but they are kept out of the task result on
      purpose, and the source says so twice:

          // note: the per-position breakdown lives in server_slot, it is not needed in a task result
          // not in server_slot_stats to avoid copying to every task result

      A patch that moves the vector into `server_slot_stats` contradicts a documented decision and
      should expect to be told so. `AGENTS.md` there also asks for an issue before a feature PR.
      So: write the issue, and have it answer the reason rather than route around it. The compact
      form is k+1 integers rather than a vector per request, from
      `H[0] = steps - S[0]`, `H[j] = S[j-1] - S[j]`, `H[k] = S[k-1]`, with
      `sum(H) == steps` and `sum(j*H[j]) == accepted` as reconciliation. An opt-in verbose field
      is the other way to answer it. The AIPerf adapter comes after, against the existing
      `SpecDecodeAcceptanceRecord` schema, not a new one.
- [ ] **E4** `embd_layer_inp` index-space reproducer before any claim is made about it.
- [ ] **E5** quantized batch-invariance conformance harness across backend, quant, width, context,
      flash attention and parallelism.

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
