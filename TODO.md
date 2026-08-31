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
      `scripts/run_phase_q.sh` and the whole A6000 plan depend on it.

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
- [x] **Forced-warp intervention, v2 four-build set** - complete 2026-08-25 11:52 on the A6000.
      All four at 150/150. The registered prediction failed in both directions, and the failure is
      a null rather than a void: the intervention had no effect at all.

      The build gates that voided v1 all hold here. `libggml-base` is identical across the four
      (67f70901eb8c485f), so no cmake reconfigure moved the CPU-side library. control and control2
      are byte-identical in `libggml-cuda`, and their SASS is identical in 0 of 6202 kernels
      differing, and their 150 outputs match byte for byte. That is the control the v1 set never
      had, and it is what makes a difference from a forced build attributable to the table.

      The edit reached the machine code and only there. Hashing every kernel: forced_up differs
      from control in 92 of 6202, all 92 `mul_mat_vec_q`, template ints {5,6,7,8} at 23 quant
      types each; forced_down2 differs in 46, all `mul_mat_vec_q`, {3,4} at 23 each. So the
      registered reading "the edited table row never reached the kernel" is refuted.

      What is left is that the warp count changes nothing observable. Output: 0 of 75 records
      differ for forced_up, 0 of 50 for forced_down2. Throughput: every forced-vs-control delta
      sits inside the control-vs-control2 band, which is +/-1 %. The warp count is not what puts
      the verification widths into two groups.

      The microbenchmark settled which of those it was, and it was the second. Correction: the
      earlier reading here, that the warp count does not matter for this workload, was wrong.

      `test-backend-ops perf -o MUL_MAT` on the A6000, the same four binaries swapped by
      directory, 184 timed cases each. Its MUL_MAT sweep varies `n`, which is ncols_dst, over
      1, 2, 3, 4, 5, 8 and 512, so the controls come with the design: 1 and 2 are untouched by
      both builds and 512 leaves MMVQ. us/run relative to control, median over quant types:

          ncols_dst   control2   forced_up   forced_down2   edited by
                  1     +0.10 %     +0.01 %       +0.00 %   neither
                  2     +0.05 %     -0.10 %       +0.08 %   neither
                  3     -0.02 %     +0.03 %       -0.57 %   forced_down2
                  4     -0.05 %     -0.25 %       -1.52 %   forced_down2
                  5     +0.11 %    +13.62 %       -0.08 %   forced_up
                  8     -0.01 %    +26.68 %       +0.00 %   forced_up
                512     -0.10 %     -0.04 %       -0.09 %   not MMVQ

      control2 sets the floor at 0.17 % median absolute deviation, 0.97 % at the 95th percentile,
      which is about six times tighter than the end-to-end run. Each forced build moves only the
      widths it edited and nothing else.

      So the warp count changes this kernel a great deal, and changes the output not at all. That
      is a stronger statement than the one it replaces. Fork positions are a property of the text,
      and a build that runs 26.68 % slower at width 8 emitted the same bytes on all 150 records,
      so MMVQ's accumulation does not depend on how the reduction is split across warps. A
      mechanism that cannot change the output cannot change where two outputs diverge, and the
      warp count is out as an explanation of the width grouping on those grounds rather than on a
      measured absence.

      It also bounds what the end-to-end null was measuring. A 26.68 % change in the width-8
      kernel moved decode by less than the rebuild noise, so that kernel is at most about 4 % of
      decode time, which fits: at n_max 7 the drafter runs seven times at ncols_dst 1 for one
      verification at width 8, and every pass carries attention and the rest besides.

      Still no upstream PR. What the data says about the table is that its choice of two warps at
      widths 5 to 8 is right on Ampere, since four costs 13.6 % at width 5 and 26.7 % at width 8,
      and that two warps at width 4 is 1.5 % faster than the stock four. A 1.5 % micro-tuning on
      one shape and one GPU is far under the bar in AGENTS.md, which asks that every merged line
      be maintained across a large matrix of platforms and backends.

      A method note worth keeping: `ggml_cuda_should_use_mmvq` carves out per-type thresholds for
      Ada, Blackwell, DGX Spark and CDNA, and Ampere falls through to `ne11 <= MMVQ_MAX_BATCH_SIZE`
      for every quantized type. On an Ada card Q4_K stops using MMVQ above ne11 7, so the same
      forced_up edit would not execute at width 8 there and the intervention would fail silently.
      The card this ran on is part of what the result means.

- [x] **forced_down2** - done on both hosts and in the v2 four-build set:
      `phase_warp_forced_down2.json`, `phase_warp_forced_down2_hostB.json` and
      `phase_warp_v2_forced_down2.json`, 150 records each. forced-down was void because its
      1-4 row includes width 1, and a drafter decodes one token at a time, so it perturbed every
      arm through its drafter. SASS hashing shows both original edits were surgical in the binary
      (1-4 identical for forced-up, 5-8 identical for forced-down), which is what forced the
      explanation. forced_down2 splits the row and leaves 1,2 at four warps. Predictions are
      registered in PREREGISTRATION.md Correction 6, written before the run finished.
- [x] **Truncation, extended-cap run** - done, and it is the same item as D2; see Correction 33.
      The two sentences that stood here are withdrawn twice over. They said the audit found 15 of
      25 prompts censored and that the partition survived on the other 10: B7 had already shown
      that split was an artefact of measuring the window in characters, and the run itself has
      since shown the censoring is not a fixed property of the study at all. At a 1600-token cap
      it is 9 of 375 records on 2 of 25 prompts, against 260 of 750 at 400.

      It was run as `bench.py --matrix <same> --max-tokens 1600` on the host that produced the file.
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

      The control was free: divergence is deterministic here, 150 of 150 arm-by-prompt cells agree
      across all five passes of phase_a, so every already-resolved fork had to come back at the
      same character. It did: 125 of 125 repeated cells agree in the new file, over its three
      passes. All three checks above held in the run - every arm including the baseline ran at
      1600, and all 25 prompts carry a single `sha_ref` across arms, so each comparison is against
      a baseline under the same cap.
- [x] **Forced-warp intervention** (host C) - forced-up stands; forced-down withdrawn. See Correction 8. - three builds of the same revision differing only in
      the `calc_nwarps` GENERIC table. Registered before any of it ran, with the outcomes and the
      baseline identity control written down first.

## Next, in order

- [x] **Phase C** - complete 2026-08-25 10:43, 750/750, 0 incidents. Drafter quantization barely
      matters and the highest precision is the slowest: q8 +53.4 %, q4k +52.0 %, bf16 +48.5 %
      against baseline, so a bf16 drafter costs about five points to run. The class effect is
      larger than the quantization effect: code +117 %, reason +90 %, zh +0.8 %. Both baselines
      agree to 0.01 tok/s across the two trees.

      The three n-gram arms are activation diagnostics, not three comparable efficacy
      measurements, and the drafting counters separate them. `ngram-mod` has `t_draft_n = 0` on
      all 75 records and matches the baseline on all 75. An earlier version of this entry read
      that as a flag accepted and ignored, the predecessor's `draft-qwen3-0.6b` failure repeating;
      Correction 25 established it is the method working as designed. Its default `n_min` is 48
      and `draft_one` discards the whole draft on hitting an empty table entry before that, so it
      needs 48 consecutive matched tokens to emit anything, and a 400-token general writing, code
      and reasoning suite does not produce one. Its -0.20 % is the cost of entering the
      speculative path and drafting nothing, and its 75/75 baseline match is the absence of
      speculation rather than lossless speculation. `ngram-map-k` drafts on 6 of 75, 288 tokens
      for 24 accepted. `ngram-cache` is the only active n-gram method here: it drafts on 63 of 75,
      9699 tokens, accepts none of them, so its -8.3 % is drafting cost with no return, and it is
      the only one of the three that supports a workload-level negative result. For contrast
      DFlash2 accepts 41.1 % and draft08b-n8 21.0 %, the latter below n4's 35.2 % and slower for
      it.
- [x] **Phase L** - complete, five rungs, 900 records. The collapse does not appear, and the
      ladder now reaches past the depth the report's own worked example measures at, so the
      verdict is no longer withheld.

      Baseline decode over realised depths of 8195, 32772, 65538, 81921 and 98300 filler tokens:
      39.67, 35.14, 30.27, 28.31, 26.53 tok/s. That is 1.50x across the whole ladder where
      llama.cpp #27623 describes 25x. The two trees agree at the deepest rung, so it is not one
      build, and part of even that 1.50x is not depth: the SM clock falls 1.60 % across the ladder
      as deeper rungs take power from the core, worth 0.43 points of throughput at the compute
      elasticity Phase R2 measured. Those two figures are `analysis/phase_l_ladder.txt` line 26,
      not a fresh derivation; 1.87 % and 0.50 appeared here and came from the four-rung partial.

      Speculation survives it intact. mtp-n2 against its own baseline is +54.6 %, +53.9 %,
      +51.7 %, +53.5 % and +53.4 % (the 8 K rung's paired class-stratified interval is
      [+52.1, +57.3]), with accepted tokens per verification step at 2.294, 2.297, 2.255, 2.280
      and 2.281. The drafter holds while the baseline slows.

      An earlier version of this entry reported +46.6 %, +48.6 %, +50.2 % and +50.7 % over four
      rungs. Those were computed while the fourth rung stood at 60 of 180 records and are
      superseded; the ratio of class-stratified means and the paired bootstrap agree to 0.02
      points on the completed data, so the gap was the partial run, not the estimator.
- [x] **Phase M** - complete, 1575 records, 21 arms, dense against MoE in one session with
      matched width ladders so `c` is comparable and not just the levels. The sign belongs to the
      drafting method rather than the architecture: the built-in MTP head wins on both targets and
      a 0.8B `draft-simple` drafter loses on both. The registered replication anchor does NOT
      hold, so none of it is a statement about the predecessor's numbers. Corrections 9 and 10.
- [x] **Phase Q-small** - complete, four rungs, 1500 records: Q4_K_M, Q6_K, Q8_0 and BF16 on
      Qwen3.5-9B-MTP. It supplies the bf16 anchor the 27B ladder structurally cannot reach, and
      its Q4_K_M is the exact file llama.cpp #26750 reports on.

      The open risk logged before the run - three `moe-draft08b-*` arms putting a second model on
      the card with about 0.57 GiB of headroom - did not fire. The cleared one held: the MoE's
      `blk.40.nextn.*` set matches the dense target's `blk.64`, so the `moe-mtp-*` arms were not
      silent baseline duplicates the way `ngram-mod` was.

      H9 is supported as an effect and #26750's specific claim is refuted: bf16 diverges on 52 %
      of requests, not the parity the report describes. See Corrections 22, 26 and 27, the last
      of which marks the one arm whose interval does not clear the study's own margin.
- [x] **Phase Q** - complete for the two rungs this card can hold, 600 records: `UD-Q4_K_XL` and
      `UD-Q5_K_XL`. `UD-Q6_K_XL` and `Q8_0` do not fit in 24 GB and 29 GB of free disk is not
      enough to stage `Q8_0` either, so the ladder stops there by hardware and not by choice.
      Phase Q-small is the instrument that covers the rest of the bit span.
- [x] **Phase B** - complete 2026-08-27, 525 records, 7 arms, 3 passes, `--spec-draft-n-max` in
      {3,7} crossed with `--spec-draft-p-min` in {0,.50,.75}. Two incidents, both host contention
      and both recorded: `nvidia-smi 50 %` on pass 2, which is this run's own power sampler and a
      false positive that motivated the descent-attribution fix, and `git 162 %` on pass 3, which
      was real and was mine. `pass_stability.py` puts the second arm-pass at 0.63 % within-pass
      scatter against a phase median of 0.74 %, so it is quieter than typical; the incident stands
      recorded either way.

      The gate works and it works hard: at n-max 7 it takes drafted tokens from 23 719 to 7 016
      and acceptance from 0.276 to 0.770.

      **The cost tracks tokens DRAFTED, not tokens REJECTED.** `harness/mechanism_b.py` is the
      analyser and did not exist before this phase. The gate sweep spreads the drafted-to-rejected
      ratio 5.22x across the six arms, which is what makes two one-parameter models comparable
      whatever their regressors' correlation. Fitting each separately on the extensive form:
      7.208 ms per drafted token at r2 0.978 against 10.184 ms per rejected token at r2 0.824, and
      the RSS difference clears zero by 18.5 half-widths. Adding the drafter's own per-step
      forward pass -- which runs whether or not the gate lets it extend -- gives 4.229 ms/step
      plus 6.112 ms/drafted token at r2 0.991 against 10.740 plus 6.889 at r2 0.969, and the
      margin narrows to 3.57 half-widths. That supports H2' over H2 on this target.

      The joint two-coefficient fit is NOT identified and is reported as such: corr(drafted/fwd,
      rejected/fwd) is +0.996 and the fit answers with a negative coefficient for rejection, which
      is what least squares does when it splits one direction in two.

      On throughput, `mtp-n7-p.00` is **+8.91 % [+3.55, +14.36]** on the class-stratified endpoint,
      which `analyze.py` flags as clearing zero by only 0.66 half-widths. That single number hides
      a sign change: +68.0 % on code and +29.3 % on reason against -16.0 % chat, -16.2 % prose and
      -20.5 % zh. Its pooled median is 37.77 tok/s against the baseline's 41.39, which is the same
      data read on a statistic this study does not use as its endpoint.

      `cost_model.py` REFUSES to report `k`, `c` or `k0` for this phase: the `mean_len` derivation
      fails its integrity check here too, mean gap -0.3151 and worst 1.1471 over 450 requests.
- [x] **Phase V** - run 2026-08-27, 75 records, and what this card can produce is one arm plus
      six recorded failures. `baseline-vllm` serves at 47.52 / 47.53 / 47.52 tok/s across three
      passes, a 0.02 % spread, and that is the cross-engine anchor this study has not had: a
      decode-only rate from `vllm:request_decode_time_seconds` over
      `vllm:generation_tokens_total`, not completion tokens over wall time. Prefill is 1.29 % of
      inference here, which is the size of the error a wall-clock rate would have carried.

      Both MTP arms failed to start on all three passes, six incidents, every one the same
      2.37 GiB allocation at `qwen3_5_mtp.py:244`. The matrix marks them `may_fail` and
      `vllm_bench.py` records a failed start as the arm's result rather than aborting, so the
      failure is in the result file with its server log beside it. Reported as vllm#53887, and
      the attribution in that issue was corrected on 2026-08-26 (comment 5427886687): the
      allocation that fails is the head at `:244`, not the embedding at `:82`, which succeeds --
      the module makes two 2.37 GiB allocations, not one.

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

- [x] **A refactor verified on printed output moved one pixel in a plot** - `mean_len` moved into
      `harness/speclen.py` and the derived form changed from `(predicted_n - 1) / F` to
      `1 + accepted / F`. Those are one identity, since `predicted_n - 1 = accepted + F`, and they
      are not one floating-point expression: 174 of phase_nmax's 1050 records differ, by at most
      8.9e-16, about four ULP. The move was checked by diffing the reports of `analyze.py` and
      `cost_model.py` before and after, on two files, and they were byte-identical. That check
      cannot see four ULP, because the reports print four decimals. Regenerating the figures found
      it: `plot_dispatch_boundary.png` differs in exactly one pixel, at (494, 437), in both the
      light and dark versions, where a point sat on a rounding boundary. The commit message for
      the move said byte-identical output "is what says the refactor changed nothing", which is
      too strong. It says the refactor changed nothing at the precision the reports carry. The new
      form is kept because it is the one that stays exact once `draft_n_verif_steps` arrives and F
      is counted rather than solved for, and the difference is recorded in `speclen.py`.

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
      diverged at token t, identical through EOS, right-censored at the cap. Across every file that
      existed then, **5825 of 5825 records stopped at the cap and none reached EOS**, so every
      identical verdict was right-censored, uniformly. Forks resolve as late as token 334 of 400 in
      Phase A and 379 of 400 in the A6000 warp control. Superseded for the extended-cap regime by
      D2: `phase_a_cap1600.json` has 267 of 525 records reaching EOS and 9 of 375 censored, and
      still 0 exact identities.
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
- [x] **C3** done, by a different route than the audit proposed -- and the reason given for that
      route was wrong, which 2026-08-30 established rather than argued. What follows is kept as it
      was written, with the correction under it.

      ~~There is no total-energy counter reachable from here: `nvidia-smi` rejects
      `total_energy_consumption` on this driver and neither `pynvml` nor any nvidia python package
      is installed, so the counter would mean a new dependency in a harness that deliberately has
      none.~~

      **Both clauses are false.** `nvidia-smi` rejecting a query field says nothing about the
      library behind it: `nvmlDeviceGetTotalEnergyConsumption` returns `NVML_SUCCESS` on this card,
      and `telemetry.NvmlEnergy` reads it through `ctypes` against `libnvidia-ml.so.1`, which is
      the standard library and not a dependency. The counter and the `power.draw.instant` integral
      agree to within 0.15 % on every arm of `results/phase_e.json` across a 2.75x power range,
      which is the cross-check this item concluded could not be had. See Correction 44. The reason
      it looked unreachable is that one tool's refusal was read as the platform's, and the
      dependency argument was never checked against what `ctypes` can do.

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
- [x] **D1b** done: the four builds from one configure all produced results --
      `phase_warp_v2_control`, `_control2`, `_forced_up` and `_forced_down2`, 150 records each.
      control, forced-up, forced_down2 and a second control, built back to back and run back to
      back. Three build gates stop it before it can produce numbers: no build may reconfigure,
      `libggml-base.so` must be identical across all four, and the two controls must be
      byte-identical in `libggml-cuda.so` while both forced builds must differ from them.
- [x] **D2** extended cap. `--max-tokens 1600` on the host that produced the file; 1600 because the
      densest prompt is 1.37 chars/token against a 1537-character threshold. Run as
      `results/phase_a_cap1600.json`, 525 records, 21 of 21 arm-passes, 2 incidents. Right-censoring
      260 of 750 -> **9 of 375**, EOS 0 of 875 -> **267 of 525**, latest fork 334 -> **1396**, still
      0 exact identities, partition unchanged. The free control held: 125 of 125 repeated cells
      agree. Pointing the readers at the file exposed three defects in them, all fixed with tests
      (Correction 33). Not carried over: the same-tree `baseline@pr27342` divergence control.
- [x] **D3** Phase B - `n_max` crossed with `p_min`. Run: `results/phase_b.json`, 525 records,
      `mtp-n3` and `mtp-n7` crossed with `p_min` 0.00 / 0.50 / 0.75. It separated the two volumes:
      step + drafted tokens fits at r2 0.9912 against 0.9687 for step + rejected, 3.57 half-widths
      apart, so **the cost tracks tokens drafted rather than tokens rejected**, and the verdict
      holds across F-2 to F+1. Two `host_contended` incidents from processes of my own leave the
      file marked FAIL in the audit; the fit above should be read with that.
- [x] **D7** evidence block wired into README.md. `evidence/registry.json` holds only what a file
      cannot state -- the question, a controlled-vocabulary strength, and the claims a phase must
      not be used for -- and `harness/render_evidence.py` computes every count from the result
      files. `verify_everything.sh` section 7 regenerates and diffs it. The registry's vocabulary
      is now enforced rather than merely declared, and the generated rows are plain rather than
      bold because `| **M** |` is the findings table's own syntax and the duplicate shadowed two
      tests that grep for a phase row.
- [x] **D6** `analysis/bootstrap_coverage.txt` regenerated at 2000 replications, which is what
      `coverage_sim.py` now defaults to so the artifact reproduces from a bare invocation. Every
      row carries its own Monte Carlo standard error, 0.6 to 0.7 points. The result reverses which
      process disagrees with the recorded 800-replication figures: normal lands on 90.9 % at
      91.1 % (0.3 SE) and heavy-tailed on 88.0 % at 87.5 % (0.7), while uniform is 92.0 % against
      90.6 % (2.3). The earlier 300-replication pass had put the discrepancy on `normal` at 2.0 SE;
      that was Monte Carlo noise. Binary at n=25 is 90.2 %, inside the continuous band.
- [ ] **D4** full re-run of Phase A under the C1/C2/C3 harness, once those land.
- [ ] **D6** what the averaged-field offset actually is. Phase E establishes what it is not, over
      119 file-arm cells and 7125 measured windows in `analysis/energy_instruments.txt`: not proportional
      (r = +0.078 against total energy), not power alone (r = +0.548, and within a cap the arm
      drawing LESS power carries the larger offset), not power fluctuation (-0.106 against
      SM-clock spread), and not a per-window constant -- an
      earlier reading with an arm-dependent time constant was refused by nine files and by the
      negative offsets on `phase_m`'s `moe-draft08b-*` arms. The context ladder adds a dimension
      none of them covers: at a nearly constant 400 to 415 W the offset per watt rises from
      0.0031 s at 8k to 0.0163 s at 96k.

      **The prerequisite is done.** This said identifying it needs the shape of the power trace,
      which the record did not carry, and that adding it was cheap. `power_sd_w` and
      `power_sd_instant_w` are recorded from 2026-08-30. `power_max_w` could never supply it: while
      the card sits at its limit, max IS the cap, so `max - mean` measures how far below the cap
      the mean sits and not how much the draw moves -- which is exactly the confound that made an
      r of +0.97 look like a mechanism in the first place.

      **The run happened and the candidate died.** `results/phase_e2.json`, 450 records and 0
      incidents, carries both spreads. The variation the smoothing removed correlates with the
      offset at -0.342 pooled and -0.250 within arms -- the wrong sign -- and the quantitative
      test, `offset_J / (sd_lost x span)`, spans 0.012 to 6.807 across the six arms. Four
      candidates are now refused rather than three. Correction 46.

      The same run convicts Correction 44's best number: mean power is +0.863 pooled and **-0.239
      within arms**.

      **And the offset is a real energy difference, not an artefact of the grid.** That had been
      the live alternative to all of the above: a linear moving average preserves the integral of
      a stationary signal, so an oscillating trace should contribute nothing, and an offset that
      scales with the spread anyway could have been what trapezoidal integration does to two
      signals with different frequency content. `results/phase_e3_*.json` -- 450 records over nine
      invocations, three sampler periods over three rotated rounds -- settles it against the
      driver's cumulative counter, which is read exactly twice per window and therefore cannot
      move with the rate. Across a threefold change of grid the instantaneous integral does not
      move (0.999x, 1.000x) and stays within **0.23 %** of the counter; the averaged integral sits
      **0.31 to 1.86 %** below it and moves *further* away as the grid refines. Correction 47.

      **D6 is answered for the part that was asked, by Phase E4.** The spread was the wrong axis.
      `power.draw` is a boxcar average of `power.draw.instant`, and deconvolving one against the
      other measures its width directly, needing no assumption about the window's ends: **1.00 to
      1.10 s**, the same on both arms at every setting, rms 1.2 to 1.6 W. That figure had only
      ever been quoted from a paper as "about a second". Averaging over T is linear and preserves
      the integral of what it averages; integrating the RESULT across a window does not, and loses
      `(T/2)` times the difference between the window's two ends whatever the trace does in
      between. With T measured there is no free parameter and the closed form reproduces the whole
      unrolled offset -- 1.06 and 1.08 predicted against observed -- with 98 % and 93 % of it
      accruing inside the first T seconds and the middle carrying 0.06 and 3.56 J. Holding idle
      around the window collapses it 24.11 -> 6.43 J and 46.03 -> 6.35 J while the window
      LENGTHENS, which refuses a per-second loss and a spread-driven one in the same measurement.
      The arm-dependence needed no per-arm time constant, which is the thing nine files refused:
      T is one number and `mtp-n2`'s window ends differ by more. Correction 48.

      **D6 is now two quantities, and one of them is answered.** Phase E5 varied the step the
      window straddles by moving the power cap -- 287.4, 122.1 and 26.5 W above an idle-with-
      model draw, a range of 10.8x against the 1.4x the committed records span on their own --
      and regressed the surviving residual on it over nine (arm, pass) cell means. The fit
      gives a **slope of +19.7 ms** and an **intercept of +3.56 J**, the latter 3.30, 3.16 and
      4.21 across the three passes. Correction 49.

      **Correction 50 withdrew that split**, because the cap moves the step and the span
      together (Spearman -0.917) and a 1/span fit described the same nine cells marginally
      better with an intercept of +1.38 J. **Correction 52 restores it.** Phase E6 held the cap
      and moved the generation length instead -- 200, 400 and 800 tokens, step within 1.7 %,
      span 2.57x -- and the span model made a risky prediction and lost: it required the
      residual to FALL 6.84 J from the short cell to the long, and it ROSE 3.54, with the
      1/span slope coming out at -55.03 against E5's +101.14. Against a round spread of 3.35 J
      that is 5.5 standard errors. The step model predicted no change and the observed change
      is 1.9 standard errors, so it survives a test it could have failed. **+19.7 ms and
      +3.56 J are the figures again.**

      **The non-linearity behind it is now named.** Any linear time-invariant filter loses
      exactly `m x (end level - start level)` over a window, whatever happens inside, so at a
      4 s roll -- both ends idle -- LTI predicts nothing and the offset is 3.6 to 9.9 J. The
      relationship is therefore not LTI, and the departure is on ONE edge: fitting the width
      separately on each gives the same 1.00 to 1.10 s, but the fit's rms is under 2 W on the
      rise and up to 18 on the fall. Stacked on the instantaneous field's own step down, the
      reported average decays, then **STALLS for about 0.8 s some tens of watts above a
      boxcar** -- 30, 12.5 and 2.5 W at the three caps -- and then drops to meet it. The
      driver's average describes how power climbs and not how it falls.

      That stall scales with the step, so it feeds the slope rather than the intercept.

      **And the fixed term does not need a load transition at all.** Windows holding no rise
      and no fall -- the cell E5 could not contain, since every E5 window carried exactly one
      pair and so confounded "per window" with "per pair" -- still show the two fields apart
      by **+0.499 W** across 49 windows, with the JOULES scaling with the window length (3.6,
      7.4 and 13.2 J at 7, 14 and 28 s) and the WATTS not. That is a level difference between
      the readout paths, and no linear filter can produce it: a filter loses `m x (end - start)`
      and both ends of an idle window are the same level. `analysis/idle_offset.txt`.

      **And it is not what E5's fixed term is made of.** Correction 51 measured the levels in
      between. A resident model does not raise the floor -- 16603 MiB sits at 30 W and 210 MHz
      -- but a request leaves the SM clock at **1860 MHz for 15 to 20 s**, which is why E5's
      4 s roll reads 128 W rather than 28. Pinning the clock holds that state with no request
      in the window, and the difference collapses by a factor of **18**: +0.501 W at the
      210 MHz floor against **+0.030 W** at 1860 MHz, worth 0.2 J over E5's 6.1 s of
      non-plateau window against a fixed term of 1.4 to 3.6 J. `phase_e`'s 150 W arms bound it
      further at 0.013 W. The level difference is real, belongs to the lowest P-state, and is
      a different thing. Two candidates are already
      refused. It is not per-second -- the caps move the span 4.5x, 13.9 to 49.4 s, and the
      plateau term stays at 0.2, 1.6 and -0.4 J. And it is not a window mismatch between the
      two integrals: they cover the identical grid, with equal sample counts and identical
      timestamps in every record at every cap.

      Withdrawn from Correction 48: that the residual being equal on both arms made it a
      different kind of object. At that roll both windows hold the same excursion, so anything
      sized by it is predicted to be equal, and the equality distinguished nothing.

      `energy_j` is now correctable in principle rather than only avoidable -- the correction is
      `(T/2)` times an end-to-end difference and both terms are recorded from Phase E4 onward --
      but no committed figure applies it, because the fields it needs (`power_first_w` and the
      trace) postdate every other result file. Reading the instantaneous field or the counter
      instead remains what the published numbers do.
- [ ] **D7** an external power meter. Phase E compares three READOUT PATHS over one on-board
      sensor, so their agreement bounds the processing and says nothing about the proportional
      bidirectional sensor error the measurement literature reports. Nothing inside this machine
      can settle the absolute magnitude, and no figure here should be read as though it had.
- [ ] **D5** factorial prompts: class crossed with thinking, language crossed with matched task,
      short and medium output lengths alongside the 400-token regime.

### U. Upstream

These were `E1` to `E5` until 2026-09-01. The measurement phases are named after the letter of
their question -- `E`, `E2`, `E3`, `E4`, `E5` are the energy-instrument phases, and four of them
were added in the two days before this rename -- so `E3` meant both the sampling-rate phase and
the llama.cpp acceptance histogram, and `E5` meant both the step-scaling phase and the
batch-invariance harness. A reader chasing either landed on whichever they found first. The
section letter moves rather than the phases, because the phase ids are in `evidence/registry.json`,
`docs/PHASES.md`, the result filenames, the artifact names and four corrections, while these ids
appear nowhere but this file.

- [x] **U1** llama.cpp output-row / token-row index-space fix, prepared as
      `upstream/llamacpp/0002-output-reorder-index-space.patch` (155 lines, DCO, `libllama.so`
      compiles). `output_swaps` is built from output-row ordering and `output_reorder()` applied
      it to two buffers that are not indexed that way: `embd_nextn` when
      `embeddings_nextn_masked` is off, and `embd_layer_inp` in every mode. A test showing that
      interleaved sequences give unsorted `out_ids` - the state that makes `output_reorder()` do
      any work at all - is added to `tests/test-batch-alloc.cpp` and passes, 206 assertions in
      that file, 0 failures. **Source-level proof, not a runtime reproducer**: no scrambled
      embedding has been observed. Submitting is the author's step; `AGENTS.md` there forbids an
      agent pushing or opening a PR.
- [x] **U2** SGLang consumer-side sibling hardening, prepared as
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
- [ ] **U3** llama.cpp acceptance histogram. **Do not write the PR.** The counters exist -
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
- [x] **U4** done before the claim was made: `repro/output_reorder_ordering/` holds the probe and
      its README. 400 randomly generated batches, upstream fails 210 of them, removing the swap
      alone fails 231 (worse than upstream), permuting by token index fails 0; rerun at four more
      seeds for 2000 further cases, 0 failures. The four fixed cases that name the mechanism are
      tabulated there, including the one where upstream is correct and removal alone is not.
- [ ] **U5** quantized batch-invariance conformance harness across backend, quant, width, context,
      flash attention and parallelism.

## Deleted from scope, with reasons

- **Phase L at f16 KV on the A6000** - built, started, withdrawn the same afternoon without
  producing a record. It was to remove a KV-precision confound from the cliff test. There is no
  such confound. `docs/PHASE_L_DESIGN.md` had already settled it: `full_attention_interval` is 4,
  so 16 of 64 layers hold KV, the cache at 96 K and q8_0 is 3.27 GB, and the whole ladder runs at
  20.8 of 24 GB. q8_0 is this study's standard setting at every depth, not a compromise made to
  reach one, so it is held constant across the ladder and cannot confound a within-ladder ratio.

  Two further errors were mine. The VRAM arithmetic used 64 KV layers instead of 16 and so
  overstated an f16 cache four-fold, which is what made the 48 GB card look necessary: f16 at
  86 016 needs 23.93 GiB and the 24 GB card holds it. And the A6000 is sm_86, the same as the
  3090, while the report is from an RTX 4080 SUPER on sm_89, so it adds no architectural
  coverage either.

  A third point stands on its own: the cliff is a 25x effect and does not need 4 arms x 15
  prompts x 3 passes. That matrix exists to put confidence intervals on speedup and acceptance.
  Applying its statistical power to a question that needs almost none was how a twenty-hour run
  got planned for something a short probe would answer.

  The card was released without a record written. Nothing rests on this and nothing was lost.


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
