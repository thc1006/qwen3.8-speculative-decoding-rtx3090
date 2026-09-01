# Methodology audit of the predecessor repo

`thc1006/qwen3.6-speculative-decoding-rtx3090` (v1.0 - v3.0, 2026-04 -> 2026-05)

This audit was run **before** any measurement in the present repo, against the predecessor's own
committed raw JSON. Its purpose is to make sure this repo does not inherit design choices that
the data show to be load-bearing. It is written the same way the predecessor wrote its own v2.3
scope correction: state the finding, state whether the conclusion survives, fix it going forward.

**Summary, as written on 2026-08-25.** Two reporting choices understated the magnitude of the
predecessor's own effect, and several design choices leave it unable to put an uncertainty on any
number it published.

**The verdict on the central conclusion is withdrawn, and this document is not maintained as an
assessment of it.** It said the conclusion survived every check below. Two things since say
otherwise, neither of them available when this was written. The predecessor's own v4 audit
retracts the anomaly the conclusion rested on -- its `ERRATA.md` A7, *"With acceptance measured
properly, there is no anomaly left to explain"*. And this repository's Phase M failed the
replication anchor derived from A1 below: `analysis/phase_m_anchor.txt` reads *"ANCHOR DOES NOT
HOLD. the class-stratified effect is -65.6 %, outside the registered -32 % to -12 % band"*, a
factor of three out. Read what follows as the design rationale for this repository's harness,
which is what it was written for.

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

The drafter emitted nothing at all on most of the suite, measured from the repo's own committed
rows: `draft_n` is **0 on 8 of the 10 prompts** in `draft-q35-08b-max8.json` -- all six chat
prompts plus `reasoning` and `zh_cn` -- and their effect against the baseline rerun runs from
-0.2 % to +0.5 %. So **80 % of the prompt set contributes essentially zero**, and the published
mean is a mixture statistic carried entirely by `long_explain` at -56.3 % and `code_small` at
-51.3 %.

This paragraph used to say 60 %, and to give the reason as chat prompts not triggering the
speculative path. Both were understatements of the same thing: the path was entered and the
drafter produced no draft, on eight prompts rather than six.

The same file is where Correction 9's counter defect is visible directly: every row has
`draft_n == draft_n_accepted` exactly, 102/102 and 168/168 on the two that drafted at all.

Re-analysing the repo's **own committed 300-token results** with a class-stratified mean
(mean of per-class means, so each class carries equal weight):

| config | published-style raw mean | class-stratified | difference |
|---|---:|---:|---:|
| `draft-q35-08b-max8` | **-10.8 %** | **-21.5 %** | -10.8 pp |
| `ngram-cache` | **-12.2 %** | **-24.4 %** | -12.2 pp |
| `ngram-mod-n24` | -3.4 % | -3.1 % | +0.2 pp |

Per-class detail for `draft-q35-08b-max8` (tok/s): chat `135.7`, zh `135.6`, reason `135.5`,
code `65.9`, prose `59.2`. The effect is not "a 3-12 % drop"; it is "no effect on more than half
the set, and a 51-56 % collapse on code and prose".

**Does the conclusion survive?** Yes, and it strengthens: the finding was *net loss*, and
stratified it is a larger net loss. The v1 README does publish a per-prompt heatmap and does
describe the behaviour as bimodal, so the data were disclosed. The issue is that the TL;DR
number, "Mean decode drops 3 to 12 %", is the diluted one, and that is the number that travels.

**Carried into this repo:** prompt set balanced five per class over five classes, 25 in all, and
the primary endpoint is
defined as the class-stratified mean with per-class effects always reported beside it
(`harness/prompts.py`, `harness/stats.py`).

---

## A2: "All completions reach the cap" is false for the 1000-token variants

v1 README, Methodology notes: *"Output capped at 300 tokens (and 1000 tokens in the `-1000tok`
variants); all completions reach the cap, so `predicted_n` is constant across runs within a
config."*

Checked against `results/`: of 190 recorded requests, **28 terminate below the cap, and all 28
are in the four `-1000tok` configs**, 7 of 10 prompts in each. Observed `predicted_n` in
`baseline-1000tok` ranges `354 ... 891` against a cap of 1000.

The 300-token configs do hold: every one of those requests reaches 300.

Why it matters: the four `-1000tok` rows appear in the same sorted results table as the
300-token rows. Their per-prompt generation lengths vary by 2.5x, so their mean tok/s is a
length-weighted mixture over a different KV-growth profile, and is not comparable to a row whose
`predicted_n` is pinned at 300.

**Carried into this repo:** `bench.py` records `predicted_n` and a `hit_cap` boolean per request
and prints a `SHORT(n)` flag when the cap is missed. It does not assert and does not drop the
record -- an earlier version of this sentence said it asserted, and it does not. What decides the
exclusion is the analyser, and it reports both series: a per-protocol one that drops
early-terminating requests and an intention-to-treat one that keeps them, side by side, so the
exclusion rule's effect is visible rather than assumed. Configs with different caps are never
placed in one ranking table.

---

## A3: N = 1 for the headline matrices, and the published "std" is not measurement uncertainty

- v1: each of the 19 configs run **once** (10 prompts, 1 warmup). Self-disclosed in Limitations.
- v3 (DFlash): "5 prompts x **1 trial** x 3 draft-max configs".
- v2 added replication (N = 3 on a subset) after review pressure.

The `std` column in the v1 results table is the spread **across prompts**, not across repeats.
Given A1, that spread is mostly the bimodality, so it measures the prompt mixture rather than
run-to-run noise. No published number in v1 or v3 carries an interval that would let a reader
tell a real 4 % effect from drift.

**Carried into this repo:** five complete passes on the primary phase, three on the later ones
and one on the controls -- across the 67 committed result files the split is 43 at one pass,
22 at three and 2 at five, and those two are Phase A and its pre-repair copy, so five passes
describes one run and not a standing rule. Intervals come from a **cluster** bootstrap that
resamples prompts (passes within a prompt are repeated measures, not independent samples);
any interval spanning zero is reported as "no detected effect", never as a direction.

---

## A4: Arms were run sequentially, so drift is confounded with arm

Each config was benchmarked to completion before the next began. Any monotone drift over the
session (thermal soak, clock behaviour, background load) is perfectly confounded with config
order. The repo's own `baseline` vs `baseline-rerun` (135.7 vs 135.5) is reassuring but is a
single paired observation, and both were run within the same block.

**Carried into this repo:** arms are **interleaved within each pass** (pass 1: all arms; pass 2:
all arms; ...), so drift is spread across arms rather than loaded onto whichever ran last. Per-arm
GPU temperature and clock are recorded at entry and exit.

---

## A5: Sampling settings and measurement tool change between versions, and numbers are compared across the change

| | tool | sampling |
|---|---|---|
| v1 | `llama-server` + Python client | `temperature = 0.0` (greedy) |
| v2 | `llama-cli -st -no-cnv` | `--temp 0.5 --seed 42`, `/no_think` appended to prompts |
| v3 | `llama-cli` | `--temp 0.5 --seed 42` |

Three differences at once (harness, sampling temperature, and a prompt-level reasoning switch),
and the resulting baselines of 135.7, 139.9 and 138.9 are discussed together. The repo does
attribute the v1->v2 gap to board-to-board variance and does document the tool change in
`BENCHMARK_ENV.md`, but with three variables moving simultaneously that attribution is not
identified by the data.

Temperature is plausibly not a neutral choice here. The standard expectation is that draft
acceptance falls as temperature rises, which would put v2/v3 at a different point of the
acceptance/throughput trade-off than v1. **This repo has not measured that, and does not assert
it.** It is registered as a testable side-question, not used as an explanation for the gap.

**Carried into this repo:** one harness for every arm; greedy for the primary endpoint (so the
losslessness comparison is meaningful at all); sampling temperature treated as an explicit
declared factor if it is varied, never as an incidental difference between versions.

---

## A6: No losslessness check and no degeneracy check

Neither exists anywhere in v1-v3. The repo reasons about *acceptance rate*, and the 100 % figure
it verified by reading `common/speculative.cpp` is itself a counter artefact: Correction 9 in
`PREREGISTRATION.md` records that every predecessor config has `total_draft == total_accept`
exactly, because the log line divided accepted by accepted, while the adjacent line gives 115
accepted of 214 drafted. The predecessor's own `ERRATA.md` A1 reaches the same verdict
independently. Acceptance is in any case an internal counter, not evidence about the bytes the
user receives.

This is not hypothetical for the successor model: vLLM issue #52475 reports MTP speculative
decoding producing **repetition collapse** on a hybrid Gated DeltaNet Qwen3.8 target. Collapsed
output is fast. A benchmark that records only tok/s will rank a broken arm first.

**Carried into this repo:** `harness/quality.py`, which screens every request for degeneracy against
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

For v1 this cuts *against* the repo's own conclusion, since the ngram arms were if anything
flattered, so the negative finding is not threatened. It still belongs in the methodology.

**Carried into this repo:** fresh server per arm per pass; prompt order fixed and identical
across arms; cold/warm status recorded per request so the artifact is measurable rather than
assumed away.

---

## A8: One config in the headline table never ran the feature it names

`draft-qwen3-0.6b` used a draft model with vocab 151936 against a target with vocab 248320. The
draft never attached, so the row is a duplicate baseline.

The repo handles this correctly and in public: the table row is annotated *"(vocab 151936, draft
never attached)"*. An earlier version of this sentence quoted it as *"vocab 151936 != 248320,
draft never attached, treat as baseline, shown for posterity"* -- the last two clauses appear
nowhere in that repository, so the quotation was longer than the source. The substance holds; the
wording did not. Recorded here only as
a positive control worth keeping: **this repo asserts drafter attachment from server logs before
accepting an arm's numbers**, rather than relying on the operator to notice.

---

---

## A9: Intra-session clock throttling: measured here, reported by nobody in the prior-art sweep

This one is not a criticism of the predecessor specifically. It applies to every study this repo
is scoped against, and it was found by instrumenting not by reading.

Measured on this host during a single dry-run pass (7 arms, 5 prompts each) on the card **as
found**: 450 W and overclocked, not stock, which `docs/GPU_AS_FOUND.md` records and which the
primary matrix was reset away from. The phenomenon is why the thermal gate exists; the 9.3 %
belongs to that state and has not been re-measured at the 420 W stock cap.

| position in pass | arm | SM clock (mean) | mean power | GPU temp (max) |
|---|---|---:|---:|---:|
| 1st | `baseline@master` | 1928.7 MHz | 445.1 W | 68.2 C |
| 2nd | `baseline@pr27342` | 1891.0 MHz | 445.2 W | 77.4 C |
| 3rd | `mtp-n2` | 1789.1 MHz | 439.2 W | 81.2 C |
| 4th | `mtp-n3` | 1815.6 MHz | 439.7 W | 82.8 C |
| 5th | `mtp-n5` | 1807.5 MHz | 440.6 W | 83.2 C |
| 6th | `dflash2-n4` | 1821.3 MHz | 438.3 W | 83.0 C |
| 7th | `dflash2-n7` | 1840.7 MHz | 440.1 W | 83.8 C |

**All seven rows are here now, and three of them change the reading.** An earlier version of this
table showed positions 1, 2, 3 and 5 -- the four that decline -- and omitted 4, 6 and 7, which
recover. The clock bottoms out at position 3 and climbs 51.6 MHz over the last four arms, so the
drift is **not monotone in position** and cannot all be a position effect.

What is a position effect is the first pair: positions 1 and 2 are the same non-speculative work
on two trees, which Phase A separately shows agree to 41.55 tok/s, and they differ by **1.95 %**.
That is the number a rotation has to defeat. The **1950 -> 1769 MHz, 9.3 %** quoted below it is
the per-record extreme over the whole pass and it mixes position with arm identity: the arm means
span 1928.7 to 1789.1, 7.2 %, and the speculative arms sit at 438-441 W against the baselines'
445, so they are not at the same operating point.

Power did not sit at ~445 W throughout: the two baselines did, the five speculative arms drew
438.3 to 440.6 W. It is still **power-limit throttling, not thermal shutdown** -- the card is
within 5 W of a 450 W cap at every arm, leakage rises with temperature, and the same wattage buys
fewer megahertz. The card never reports an error and `/health` stays green.

Why it matters: a 1.95 % position effect on identical work is larger than several of the effects
this study is trying to resolve, and because arms occupy different positions within a pass, it
lands **inside every paired comparison**. A study that runs arms sequentially assigns the whole of it to whichever arm
ran last.

No study in the prior-art sweep **reports** controlling for this, which is a weaker statement
than it may read as: the sweep sees what was published, not what was done, so a study that
controlled for drift and did not say so is invisible to it. The closest published thing is
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
| 10 prompts, 60 % chat, unbalanced | 25 prompts, balanced 5 per class |
| headline = raw mean over prompts | headline = class-stratified mean, per-class always shown |
| some prompts terminate early | every prompt written to exceed the cap; early termination flagged |
| N = 1 (v1, v3), N = 3 on a v2 subset | N = 5 on the primary phase, 3 on the later phases, 1 on the controls |
| std across prompts, reported as spread | cluster bootstrap CI over prompts |
| arms run sequentially | arms interleaved within each pass |
| harness and sampling change between versions | one harness, greedy primary endpoint |
| acceptance only | acceptance **and** degeneracy **and** byte-level divergence |
| ngram warm-cache artifact undisclosed | fresh server per arm-pass; cold/warm recorded |
| drafter attachment checked by eye | drafter attachment asserted from server logs |
| no port-collision guard | refuses to measure unless our own PID owns the port |
| no power measurement | power integrated over generation -> tok/J |
| intra-session clock throttling uncontrolled (all prior art) | thermal gate at arm entry + rotation + clock recorded as covariate |

## Recommended follow-up in the predecessor repo -- superseded

This section recommended an addendum re-reporting the v1 table with a class-stratified mean,
noting that the published magnitude moves by roughly a factor of two in the direction that
supports that paper's argument.

**That repository has since run its own v4 audit, which goes considerably further and reaches a
different verdict on the central claim** -- its `ERRATA.md` A7 records that with acceptance
measured properly there is no anomaly left to explain. The recommendation is left here as what
was said, not as what is still wanted.
