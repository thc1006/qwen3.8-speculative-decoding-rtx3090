# TODO: qwen3.8-speculative-decoding-rtx3090

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked · `[-]` dropped

Last updated: 2026-08-24 (Phase A running)

---

## P0: Infrastructure (blocking everything)

- [~] **P0.1 Toolchain**
  - [x] P0.1.1 Confirm host: RTX 3090 24GB, driver 610.43.02, i9-13900, 31GB RAM, Debian 13, 75GB free disk
  - [x] P0.1.2 Confirm NVIDIA CUDA Debian-13 apt repo already configured
  - [x] P0.1.3 CUDA 13.3 installed (nvcc 13.3.73)
  - [x] P0.1.4 ninja + ccache installed
  - [x] P0.1.5 master built: `c060ca9`, build 200, sm_86
  - [x] P0.1.6 PR #27342 tree built: `d1a522f`, identical flags
  - [ ] P0.1.7 Record exact commit hashes + build flags for both trees into `BENCHMARK_ENV.md`

- [~] **P0.2 Models** (running in background, `logs/download.log`)
  - [~] P0.2.1 `Qwen3.8-27B-UD-Q4_K_XL.gguf` (17.56 GB): primary target
  - [~] P0.2.2 DFlash2 drafters: BF16 (3.86) + Q8_0 (2.06) + Q4_K_M (1.14) — the quant-ladder experiment
  - [~] P0.2.3 `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (1.37 GB): insurance if UD quant lacks embedded MTP
  - [~] P0.2.4 `Qwen3.5-0.8B-Q4_K_M` (~0.5 GB): vocab-matched (248320) classic draft
  - [ ] P0.2.5 Verify sha256 of every file, record in `BENCHMARK_ENV.md`
  - [x] P0.2.6 CONFIRMED by reading the GGUF: `qwen35.nextn_predict_layers=1`, 4 `blk.64.nextn.*` tensors, block_count 65. No separate MTP module needed.
  - [ ] P0.2.7 DEFERRED to Phase 2: `Qwen3.6-35B-A3B-UD-Q4_K_XL` (21 GB): MoE contrast arm
  - [ ] P0.2.8 DEFERRED to Phase 3: Qwen3.8-27B AWQ/GPTQ int4 for vLLM (~16 GB; Ampere sm86 has no FP8)

- [ ] **P0.3 Harness** (`harness/`): the rigor differentiator
  - [ ] P0.3.1 Port `bench_runner.py` forward; keep llama-server + OpenAI-compat shape
  - [ ] P0.3.2 **Interleaved paired A/B**: alternate arms within one session, NOT all-of-A-then-all-of-B
  - [ ] P0.3.3 N>=5 complete passes; report bootstrap CI + paired effect size, not just mean+-std
  - [ ] P0.3.4 **Port-binding guard**: refuse to start if port bound; verify PID owns the port
        (a contributor in sudoingX/qwen38-mtp published 3 fake rows to a zombie server)
  - [ ] P0.3.5 Parse `draft acceptance` from server logs; store per-request, not just aggregate
  - [ ] P0.3.6 `nvidia-smi` power sampling thread -> tok/J
  - [ ] P0.3.7 Losslessness checker: greedy, hash outputs, report bitwise-identical / diverged-at-token-N
  - [ ] P0.3.8 Thermal + clock capture per run (this card is 450W cap, NOT the 350W of v2/v3 boxes)
  - [ ] P0.3.9 Emit machine-readable JSON per run + a single `results/index.jsonl`
  - [x] P0.3.10 Rotate arm order across passes (position effect, not just drift)
  - [x] P0.3.11 Pin the full sampler chain at greedy (server defaults are not neutral)
  - [x] P0.3.12 Defer baseline comparison to end-of-pass (arm rotation breaks inline reference)
  - [x] P0.3.13 GPU exclusivity assertion before and after server start
  - [x] P0.3.14 Determinism check: greedy output must be byte-identical across passes
  - [x] P0.3.15 Prefill energy measured separately (repeated) so tok/J is decode-only
  - [x] P0.3.16 VERIFIED: `enable_thinking` works: think=True emits `reasoning_content`, think=False emits none.
  - [x] P0.3.17 VERIFIED: embedded, module not required.
  - [x] P0.3.18 VERIFIED: `--spec-type` accepts none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,ngram-{simple,map-k,map-k4v,mod,cache}. `--spec-draft-n-max` default 3. `--fit` present. `draft-eagle3` unusable: no published EAGLE3 drafter for this model.

- [x] **P0.4 Pre-registration** written before measurement; corrections appended, never rewritten

---

## P1: Spearhead A: DFlash2 x consumer Ampere x Q4 target

Prior art (must cite, must not overclaim):
- **REFUTED 2026-08-24: "first RTX 3090 + DFlash2 + Q4 datapoint" is FALSE.** PR #27342's own
  comment thread already carries two single-3090 Q4 datapoints (`treo` on UD-Q4_K_XL 32K,
  `ouening` on UD-Q4_K_M 128K/Windows). Do not claim priority. Claim the controlled protocol,
  the drafter-quant ladder, losslessness/degeneracy, and energy -- none of which they ran.
- llama.cpp PR #27342 (OPEN since 2026-08-18): H200 + BF16 in the PR body; 1.824x, acceptance 28.1%
- `zptalk0221-cpu/llama-cpp-dflash2-qwen3.8-windows`: RTX 4090 48GB, Q6_K target, ~70 vs ~60 tok/s,
  user-reported, no trial count, no acceptance data
- z-lab GGUF card: acceptance length BF16 5.28 / Q8_0 5.13 / Q4_K_M 5.39
- NOT covered anywhere: RTX 3090 / 24GB / Q4 target / controlled N>=5

- [ ] P1.0 **Dual-tree baseline**: run a no-spec baseline on BOTH master and the PR #27342 tree
      with identical flags. Without it, "DFlash2 vs baseline" conflates the method with the
      branch. Report the baseline-to-baseline offset as its own number.
- [x] P1.1 DFlash2 RUNS on sm86: smoke test 91.92 tok/s, accept 0.67, mean draft len 5.64
- [ ] P1.2 `--spec-draft-n-max` sweep {1,2,3,4,5,7} at block size 8
- [ ] P1.3 **Drafter-quant ladder** BF16 vs Q8_0 vs Q4_K_M against the SAME Q4 target
      -> directly tests the v3.0 claim "Q4 target collapses DFlash"
      -> z-lab's own numbers say Q4 drafter accepts BEST; if that reproduces, v3.0 needs an erratum
- [ ] P1.4 Report acceptance length AND realized tok/s: the whole point is that they dissociate
- [ ] P1.5 VRAM accounting: drafter footprint vs context ceiling on 24GB

---

## P2: Spearhead B: Gated DeltaNet state-rollback cost  [HIGHEST NOVELTY]

Prior art: SpecLA (arXiv 2607.16673, 2026-07-18) — theory + H100 + GDN-1.3B toy model, up to 1.70x.
NOBODY has measured rollback cost on a shipped hybrid model, on consumer hardware.

Mechanism: Qwen3.8-27B is 48 linear_attention + 16 full_attention (`full_attention_interval: 4`).
On rejection, the 16 full-attention layers roll back cheaply (KV suffix truncate); the 48 GDN layers
must reconstruct recurrent state. Hypothesis: this is what pins the n-max optimum at 2-3 on every
24GB card, not acceptance decay alone.

- [ ] P2.1 Establish the acceptance-vs-realized-yield curve per method; look for the bend
- [ ] P2.2 **Discriminating test for H2 vs H2'**: the PR #27342 author proposes a competing
      quantization/arithmetic-intensity explanation for the same n-max ceiling. Hold acceptance
      approximately fixed with `--spec-draft-p-min` and sweep n-max:
        H2  (rollback) predicts yield keeps degrading (cost is paid on REJECTION)
        H2' (quant)    predicts degradation flattens (cost is paid per DRAFTED token)
      Report whichever wins. Do not defend H2.
- [ ] P2.3 Compare against a pure-attention control of similar size if one is affordable on disk
- [ ] P2.4 Instrument: does llama.cpp master's `ssm_scan` rollback show up in profiling?
- [ ] P2.5 Write the mechanism paragraph: this is the repo's "MoESD moment"

---

## P3: Spearhead C: resolve the published dense-vs-MoE contradiction  [REFRAMED]

The public record disagrees with itself and nobody has resolved it:
- njannasch.dev (5060 Ti 16GB, Qwen3.6): MoE +MTP = 144 t/s (1.47x), **dense +MTP = 42% SLOWER**
- sudoingX (RTX 3090 24GB, Qwen3.8): **dense +MTP = +54% to +81%**

Same architecture family, same flag, opposite sign. Candidate explanations to separate:
model generation (3.6 vs 3.8) / VRAM headroom (16 vs 24 GB) / llama.cpp build age
(the `ssm_scan` state-rollback commit) / quant.

- [ ] P3.1 Reproduce sudoingX's 3090 dense number under OUR controlled protocol (sanity anchor)
- [ ] P3.2 Phase 2: pull Qwen3.6-35B-A3B GGUF, run the MoE arm under the SAME protocol
- [ ] P3.3 Cross against this repo's own 2026-04 Qwen3.6-35B-A3B data (3rd physical 3090: disclose variance)
- [ ] P3.4 Test the build-age hypothesis: pre/post `ssm_scan` rollback commit
- [ ] P3.5 Answer plainly: was the 2026-04 net loss caused by MoE, or by consumer Ampere?

---

## P4: Full controlled method matrix (single host, single protocol)

Nobody has all of these on one box with one protocol. sudoingX is crowdsourced across
different builds/quants/OSes and discloses its own confounds; syv-ai is vLLM-only.

- [ ] P4.1 baseline (no spec)
- [ ] P4.2 `--spec-type draft-mtp`, n-max sweep
- [ ] P4.3 `--spec-type draft-dflash` (DSpark path, merged in master)
- [ ] P4.4 DFlash2 (PR #27342 tree)
- [ ] P4.5 classic `-md Qwen3.5-0.8B-Q4_K_M` (vocab 248320 == target, confirmed from config.json)
- [ ] P4.6 `ngram-cache`, `ngram-mod`, `ngram-map-k`
- [ ] P4.7 Known-crash combos: reproduce and report, don't silently skip:
      `draft-mtp,ngram-mod` segfaults/CUDA-aborts per two independent 3090 reports
- [ ] P4.8 Cold-vs-warm discipline: ngram scores its own cache on repeat prompts (documented artifact)

---

## P5: Coverage nobody has (each is explicitly "not covered" in a competitor)

- [ ] P5.1 **tok/J energy curve**: syv-ai lists it as not covered; sudoingX did a power sweep but
      published no tok/J. Sweep power limit; find the efficiency knee.
- [ ] P5.2 **thinking ON x method**: sudoingX is thinking-off almost everywhere.
      KGP has the reasoning_effort x acceptance ladder but on RTX 5090 (Blackwell).
      Agentic traffic is thinking-on; nobody has the Ampere ladder.
- [ ] P5.3 **Multimodal x spec decode**: Qwen3.8-27B is native VL (`image_token_id: 248056`).
      llama.cpp #19712 blocks `--mmproj` + `--model-draft`; PR #27342 says vision fails on
      draft/image position misalignment. Does `draft-mtp` + `--mmproj` work? Even a negative is first.
- [ ] P5.4 **Concurrency table**: everyone cites "gone by --parallel 4"; nobody published the table.
      Also closes limitation (iv) of this author's own qwen3.6 repo.
- [ ] P5.5 **Context-depth yield curve through llama.cpp #27623**: open bug: decode collapses ~25x
      past ~80K on this hybrid GDN model while prompt processing stays fast. Map spec yield across it.
- [ ] P5.6 **Losslessness on CUDA + hybrid + MTP/DFlash2**: arXiv 2607.17283 did this for
      Apple M3 + Qwen2.5 + classic SD only. ianlpaterson reports NOT bit-exact at temp=0 on 3090 Ti.
      Settle it for this model/method/hardware.

---

## P6: Phase 3: vLLM cross-engine arm (same single 3090)

syv-ai/qwen38-27b-rtx3090 explicitly lists "llama.cpp comparisons" as NOT covered. Nobody has
llama.cpp vs vLLM on the same single consumer card for this model.

- [!] BLOCKED ON DISK: needs ~28 GB more than Phases 1-2 leave. Plan: archive Phase 2 results,
      delete the 21 GB MoE GGUF, then proceed. Re-downloadable.
- [ ] P6.1 vLLM venv (Ampere sm86: no native FP8 -> AWQ or GPTQ int4, not the FP8 checkpoint)
- [ ] P6.2 vLLM MTP `num_speculative_tokens` sweep
- [ ] P6.3 Same prompts, same statistics, decode-rate-matched comparison against the llama.cpp arm
- [ ] P6.4 Check vLLM #52475 (MTP repetition collapse w/ turboquant KV on sm120) on sm86 + standard KV

---

## P7: Day-0 readiness for Qwen3.8-35B-A3B  [STRATEGIC]

Qwen3.8-35B-A3B does NOT exist yet (HF Qwen/ org has only 2.4T-A95B, 27B, and their FP8 variants).
It has been spotted in a supported-models table; weights unreleased as of 2026-08-24.

- [ ] P7.1 Keep the harness model-agnostic (config-driven, no hardcoded model assumptions)
- [ ] P7.2 Watch: HF Qwen org, llama.cpp model support PRs, vLLM model PRs
- [ ] P7.3 On release: run the full matrix same-day. This author is the only person holding a
      matched Qwen3.6-35B-A3B control set on the same hardware class.

---

## P8: Publication

- [ ] P8.1 README with the honest scope discipline of the qwen3.6 repo (state what is NOT covered)
- [ ] P8.2 Every prior-art claim above cited and linked; no "first" claim that verification refuted
- [ ] P8.3 Raw logs committed (`.gitignore` negation, as in the qwen3.6 repo)
- [ ] P8.4 CHANGELOG.md
- [ ] P8.5 Zenodo DOI
- [ ] P8.6 Feed results back upstream: llama.cpp #27623, PR #27342, vLLM #52475
- [ ] P8.7 Cross-link the qwen3.6 repo + qwen3.6-vllm-2x3090 sibling; add an ERRATUM to qwen3.6 v3.0
      if P1.3 refutes the "Q4 collapses DFlash" claim


---

## Added during execution (not in the original plan)

- [x] **X.1 Overclock discovered and removed from the primary condition.** The card arrived with
      memory +400 MHz, core +100 MHz and a 450 W limit while the write-up said "stock". The first
      Phase A run was **discarded**. `docs/GPU_AS_FOUND.md`.
- [x] **X.2 Clock throttling quantified and controlled.** The card sits on its power cap and loses
      9.3 % of SM clock over a pass. Thermal gate at arm entry + rotation + clock as covariate.
      No study in the prior-art sweep controls for this. `docs/METHODOLOGY_AUDIT.md` A9.
- [x] **X.3 Run lock.** `gpustate.apply()` refuses to move clocks while a run holds
      `.gpu-in-use.lock` — added after clock probing contaminated a live run (also discarded).
- [x] **X.4 Per-arm GPU state verification.** Recording state once at startup does not catch a
      mid-run change.
- [x] **X.5 Predecessor audit.** Re-analysis of the earlier repo's own committed data shows its
      headline effect was understated ~2x by prompt-class mixture, and that its
      "all completions reach the cap" claim fails for the four `-1000tok` configs.
      `docs/METHODOLOGY_AUDIT.md`.
- [x] **X.6 Phase R designed**: resource-response (bandwidth x compute x method). Promoted to the
      primary mechanism test because the pilot showed both resources are independently settable
      and the baseline's bandwidth elasticity is ~1.0, giving a built-in calibration point.
      `docs/EXPERIMENT_PLAN.md`, `harness/matrices/phase_r.py`, `harness/preflight_r.py`.
- [x] **X.7 Analysis pipeline validated against known ground truth** with synthetic data: which
      caught an inverted argument order that would have flipped the sign of every reported effect.
