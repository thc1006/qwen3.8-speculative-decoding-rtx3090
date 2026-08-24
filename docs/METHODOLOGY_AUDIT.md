# Methodology audit of the predecessor repo

`thc1006/qwen3.6-speculative-decoding-rtx3090` (v1.0 – v3.0, 2026-04 → 2026-05)

This audit was run **before** any measurement in the present repo, against the predecessor's own
committed raw JSON. Its purpose is to make sure this repo does not inherit design choices that
the data show to be load-bearing. It is written the same way the predecessor wrote its own v2.3
scope correction: state the finding, state whether the conclusion survives, fix it going forward.

**Summary: the predecessor's central conclusion survives every check below. Two reporting
choices understated the magnitude of its own effect, and several design choices leave it unable
to put an uncertainty on any number it published.**

---

## A1: The headline mean was diluted by prompt-class mixture (magnitude understated ~2x)

The v1 prompt set is 10 prompts, and its class balance is:

| class | n |
|---|---:|
| chat | **6** |
| code | 1 |
| prose | 1 |
| reason | 1 |
| zh | 1 |

Chat prompts do not trigger the speculative path on this model at all — measured, from the
repo's own data, baseline `135.8` vs `draft-q35-08b-max8` `135.7` tok/s on the chat class. So
60 % of the prompt set contributes exactly zero effect to the average, and the published mean is
a mixture statistic dominated by prompts where nothing happens.

Re-analysing the repo's **own committed 300-token results** with a class-stratified mean
(mean of per-class means, so each class carries equal weight):

| config | published-style raw mean | class-stratified | difference |
|---|---:|---:|---:|
| `draft-q35-08b-max8` | **−10.8 %** | **−21.5 %** | −10.8 pp |
| `ngram-cache` | **−12.2 %** | **−24.4 %** | −12.2 pp |
| `ngram-mod-n24` | −3.4 % | −3.1 % | +0.2 pp |

Per-class detail for `draft-q35-08b-max8` (tok/s): chat `135.7`, zh `135.6`, reason `135.5`,
code `65.9`, prose `59.2`. The effect is not "a 3–12 % drop"; it is "no effect on more than half
the set, and a 51–56 % collapse on code and prose".

**Does the conclusion survive?** Yes, and it strengthens: the finding was *net loss*, and
stratified it is a larger net loss. The v1 README does publish a per-prompt heatmap and does
describe the behaviour as bimodal, so the data were disclosed. The issue is that the TL;DR
number — "Mean decode drops 3–12 %" — is the diluted one, and that is the number that travels.

**Carried into this repo:** prompt set balanced 3 × 5 by class, and the primary endpoint is
defined as the class-stratified mean with per-class effects always reported beside it
(`harness/prompts.py`, `harness/stats.py`).

---

## A2: "All completions reach the cap" is false for the 1000-token variants

v1 README, Methodology notes: *"Output capped at 300 tokens (and 1000 tokens in the `-1000tok`
variants); all completions reach the cap, so `predicted_n` is constant across runs within a
config."*

Checked against `results/`: of 190 recorded requests, **28 terminate below the cap, and all 28
are in the four `-1000tok` configs** — 7 of 10 prompts in each. Observed `predicted_n` in
`baseline-1000tok` ranges `354 … 891` against a cap of 1000.

The 300-token configs do hold: every one of those requests reaches 300.

Why it matters: the four `-1000tok` rows appear in the same sorted results table as the
300-token rows. Their per-prompt generation lengths vary by 2.5x, so their mean tok/s is a
length-weighted mixture over a different KV-growth profile, and is not comparable to a row whose
`predicted_n` is pinned at 300.

**Carried into this repo:** `bench.py` asserts the cap is reached per request, records
`predicted_n`, and flags any early-terminating request as a length confound instead of averaging
it in. Configs with different caps are never placed in one ranking table.

---

## A3: N = 1 for the headline matrices, and the published "std" is not measurement uncertainty

- v1: each of the 19 configs run **once** (10 prompts, 1 warmup). Self-disclosed in Limitations.
- v3 (DFlash): "5 prompts × **1 trial** × 3 draft-max configs".
- v2 added replication (N = 3 on a subset) after review pressure.

The `std` column in the v1 results table is the spread **across prompts**, not across repeats.
Given A1, that spread is mostly the bimodality — i.e. it measures the prompt mixture, not
run-to-run noise. No published number in v1 or v3 carries an interval that would let a reader
tell a real 4 % effect from drift.

**Carried into this repo:** N ≥ 5 complete passes; intervals from a **cluster** bootstrap that
resamples prompts (passes within a prompt are repeated measures, not independent samples);
any interval spanning zero is reported as "no detected effect", never as a direction.

---

## A4: Arms were run sequentially, so drift is confounded with arm

Each config was benchmarked to completion before the next began. Any monotone drift over the
session — thermal soak, clock behaviour, background load — is perfectly confounded with config
order. The repo's own `baseline` vs `baseline-rerun` (135.7 vs 135.5) is reassuring but is a
single paired observation, and both were run within the same block.

**Carried into this repo:** arms are **interleaved within each pass** (pass 1: all arms; pass 2:
all arms; …), so drift is spread across arms rather than loaded onto whichever ran last. Per-arm
GPU temperature and clock are recorded at entry and exit.

---

## A5: Sampling settings and measurement tool change between versions, and numbers are compared across the change

| | tool | sampling |
|---|---|---|
| v1 | `llama-server` + Python client | `temperature = 0.0` (greedy) |
| v2 | `llama-cli -st -no-cnv` | `--temp 0.5 --seed 42`, `/no_think` appended to prompts |
| v3 | `llama-cli` | `--temp 0.5 --seed 42` |

Three differences at once (harness, sampling temperature, and a prompt-level reasoning switch),
and the resulting baselines — 135.7 / 139.9 / 138.9 — are discussed together. The repo does
attribute the v1→v2 gap to board-to-board variance and does document the tool change in
`BENCHMARK_ENV.md`, but with three variables moving simultaneously that attribution is not
identified by the data.

Temperature is plausibly not a neutral choice here — the standard expectation is that draft
acceptance falls as temperature rises, which would put v2/v3 at a different point of the
acceptance/throughput trade-off than v1. **This repo has not measured that, and does not assert
it.** It is registered as a testable side-question, not used as an explanation for the gap.

**Carried into this repo:** one harness for every arm; greedy for the primary endpoint (so the
losslessness comparison is meaningful at all); sampling temperature treated as an explicit
declared factor if it is varied, never as an incidental difference between versions.

---

## A6: No losslessness check and no degeneracy check

Neither exists anywhere in v1–v3. The repo reasons about *acceptance rate* (correctly, and it
verified the 100 % figure by reading `common/speculative.cpp`), but acceptance is an internal
counter, not evidence about the bytes the user receives.

This is not hypothetical for the successor model: vLLM issue #52475 reports MTP speculative
decoding producing **repetition collapse** on a hybrid Gated DeltaNet Qwen3.8 target. Collapsed
output is fast. A benchmark that records only tok/s will rank a broken arm first.

**Carried into this repo:** `harness/quality.py` — every request screened for degeneracy against
both absolute thresholds and its own baseline for the same prompt; greedy outputs compared
character-by-character against the no-spec baseline, reporting where the texts fork rather than
asserting either "lossless" or "not lossless".

---

## A7: ngram cache accumulates across prompts within a config, undisclosed

v1 runs all 10 prompts sequentially against one server instance per config. For the
`ngram-cache` / `ngram-mod` family the n-gram store persists across those requests, so later
prompts are scored against a cache warmed by earlier ones. The repo restarts the server *between
configs* (documented) but not between prompts.

Two independent third-party 3090 reports on the successor model quantify this artifact
directly: with `ngram-mod`, repeated passes of the same prompt read `111.1` cold then `124.4` and
`122.5` warm.

For v1 this cuts *against* the repo's own conclusion — the ngram arms were, if anything,
flattered — so the negative finding is not threatened. It still belongs in the methodology.

**Carried into this repo:** fresh server per arm per pass; prompt order fixed and identical
across arms; cold/warm status recorded per request so the artifact is measurable rather than
assumed away.

---

## A8: One config in the headline table never ran the feature it names

`draft-qwen3-0.6b` used a draft model with vocab 151936 against a target with vocab 248320. The
draft never attached, so the row is a duplicate baseline.

The repo handles this correctly and in public — the table row is annotated *"vocab 151936 ≠
248320, draft never attached — treat as baseline, shown for posterity"*. Recorded here only as
a positive control worth keeping: **this repo asserts drafter attachment from server logs before
accepting an arm's numbers**, rather than relying on the operator to notice.

---

---

## A9: Intra-session clock throttling: measured here, controlled by nobody in the prior art

This one is not a criticism of the predecessor specifically. It applies to every study this repo
is scoped against, and it was found by instrumenting not by reading.

Measured on this host during a single dry-run pass (7 arms, 5 prompts each, 450 W stock cap):

| position in pass | arm | SM clock (mean) | GPU temp |
|---|---|---:|---:|
| 1st | `baseline@master` | ~1929 MHz | 62 → 73 °C |
| 2nd | `baseline@pr27342` | ~1891 MHz | 74 → 80 °C |
| 3rd | `mtp-n2` | ~1789 MHz | 79 → 83 °C |
| 5th | `mtp-n5` | ~1808 MHz | 81 → 84 °C |

Full spread across the pass: **1950 → 1769 MHz, 9.3 %**. Power sat at ~445 W throughout, against
a 450 W cap — so this is **power-limit throttling, not thermal shutdown**: leakage rises with
temperature, and the same wattage buys fewer megahertz. The card never reports an error and
`/health` stays green.

Why it matters: 9.3 % is larger than several of the effects this study is trying to resolve, and
because arms occupy different positions within a pass, the position effect lands **inside every
paired comparison**. A study that runs arms sequentially assigns the whole of it to whichever arm
ran last.

No study in the prior-art sweep controls for this. The closest is
`sudoingX/qwen38-mtp` rule 7 ("a shared desktop halves everything, silently"), which is about a
competing tenant rather than about the card's own thermal trajectory, and one contributor's power
sweep, which varies the cap deliberately rather than holding entry state constant.

**Carried into this repo:** three independent controls, because none of them is sufficient alone.
1. `telemetry.settle_gpu()` gates arm entry on a measured temperature, so every arm starts from
   the same thermal state; a timeout is recorded as an incident rather than passed over.
2. Arm order rotates across passes, so residual position effect is spread rather than assigned.
3. SM clock, temperature and power are recorded per request and are available as covariates, so
   a reader can check whether any reported effect tracks clock.

---

## What this repo changes, in one table

| Predecessor | Here |
|---|---|
| 10 prompts, 60 % chat, unbalanced | 15 prompts, balanced 3 × 5 by class |
| headline = raw mean over prompts | headline = class-stratified mean, per-class always shown |
| some prompts terminate early | every prompt written to exceed the cap; early termination flagged |
| N = 1 (v1, v3), N = 3 on a v2 subset | N ≥ 5, all arms |
| std across prompts, reported as spread | cluster bootstrap CI over prompts |
| arms run sequentially | arms interleaved within each pass |
| harness and sampling change between versions | one harness, greedy primary endpoint |
| acceptance only | acceptance **and** degeneracy **and** byte-level divergence |
| ngram warm-cache artifact undisclosed | fresh server per arm-pass; cold/warm recorded |
| drafter attachment checked by eye | drafter attachment asserted from server logs |
| no port-collision guard | refuses to measure unless our own PID owns the port |
| no power measurement | power integrated over generation → tok/J |
| intra-session clock throttling uncontrolled (all prior art) | thermal gate at arm entry + rotation + clock recorded as covariate |

## Recommended follow-up in the predecessor repo

An addendum re-reporting the v1 table with a class-stratified mean. The conclusion does not
change; the published magnitude does, by roughly a factor of two, in the direction that
*supports* the paper's argument. Filing that is consistent with how that repo handled its v2.3
scope correction.
