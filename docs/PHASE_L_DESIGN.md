# Phase L: does speculation survive the long-context decode cliff?

Target: [llama.cpp #27623](https://github.com/ggml-org/llama.cpp/issues/27623), open 2026-08-23,
**zero comments**. Reported on an RTX 4080 SUPER (sm_89): decode throughput on this model
collapses roughly 25x once the KV position passes ~80 K (33 t/s at 68 K -> 1.4 t/s at 91 K) while
prompt processing stays fast (~1300 t/s). Reproduced there across three quants.

**Its author withdrew that 25x on 2026-08-26**, after re-measuring with eval-only rather than
wall-clock timings (PREREGISTRATION.md Correction 23). This phase ran before that. Its own
non-reproduction -- a factor of 1.5 against a reported 25 -- is therefore consistent with the
withdrawal rather than independent of it, and `evidence/registry.json` already forbids reading
Phase L as a refutation of #27623 for a separate reason.

Two things the 2026-08-24 prior-art sweep found nobody had posted: a reproduction on another
architecture, and the obvious follow-up, **does speculative decoding survive the cliff, amplify
it, or mask it?** The sweep sees what was published, not what was run. DFlash2's advertised
advantage is precisely long-context retention, so the answer is not guessable.

## Feasibility, computed from the target GGUF's own metadata

| field | value |
|---|---|
| `qwen35.attention.head_count_kv` | 4 |
| `qwen35.attention.key_length` / `value_length` | 256 / 256 |
| `qwen35.block_count` | 65 (64 layers + 1 nextn) |
| `qwen35.full_attention_interval` | 4 -> **16 of 64 layers hold KV** |
| `qwen35.ssm.inner_size` / `state_size` | 6144 / 128 (the 48 GDN layers) |

KV per token = 4 heads x 256 x 2 (K and V) = 2048 elements, over 16 layers.
At `q8_0` (~1.06 bytes/element) that is ~ **34 KB/token**; at `q4_0` (~0.56) ~ **18.4 KB/token**.
The GDN recurrent state is ~75 MB total and does not grow with context.

| context | KV @ q8_0 | model + KV | fits 24 GB? |
|---:|---:|---:|---|
| 8 K | 0.27 GB | 17.8 GB | yes |
| 32 K | 1.09 GB | 18.7 GB | yes |
| 64 K | 2.18 GB | 19.7 GB | yes |
| 96 K | 3.27 GB | 20.8 GB | yes |
| 128 K | 4.36 GB | 21.9 GB + ~1.9 compute = **23.8 GB, 99 %** | **not attempted** |
| 262 K | 8.92 GB | 26.5 GB | **no** - needs q4_0 (~22.4 GB, matching the community figure of 22.2 GB) |

So the cliff at ~80 K is reachable at `q8_0`, on the same KV setting as every other phase in this
repo. No quant change is needed to cross it, which keeps the comparison clean.

## Design

Depths **{8 K, 32 K, 64 K, 80 K, 96 K}** x methods {baseline, mtp-n2, dflash2-n4} **plus
`baseline@Nk-pr`, the PR-27342-tree build control** -- four arms, not three -- with 3 passes,
**15 prompts (3 per class)** and a **160-token cap**.

That line used to read 128 K rather than 80 K and to list three arms. 128 K was dropped before the
run: with a q8_0 cache it needs 23.8 of 24 GB, which `harness/matrices/phase_l.py` records as "not
attempted". 80 K was added so the reported ~80 K cliff is straddled rather than jumped.

**`CACHE_PROMPT` is True here and False in every other phase** -- every request in an arm shares
one filler and re-prefilling 96 K tokens per request would dominate the run; the reasoning is in
the matrix. The result is 180 records per rung, 900 in all, 0 incidents.
8 K is the ladder's own anchor, at the same **depth** as Phase A but not the same protocol: 15
prompts against 25, a 160-token cap against 400, `-c 12288` against 8192, `cache_prompt` on rather
than off, 3 passes against 5 and 4 arms against 7. Absolute rates therefore do not transfer, and
`analysis/phase_l_ladder.txt` computes retention against each method's own 8 K rung rather than
against Phase A. This used to read "the anchor shared with Phase A", which claimed more than the
shared depth supports. The long-context arms attach to an already-characterised
reference not a fresh one.

## The one thing that must not be got wrong

The context has to be filled with **real, non-repeating text**. Filling it by repeating a
paragraph would hand the ngram-style predictability of that repetition to the drafter and inflate
acceptance for a reason that has nothing to do with context depth, the same artifact two
independent 3090 reports describe when `probe.py` sends a prompt three times and the n-gram cache
scores its own history on runs two and three.

Filler will therefore be assembled from distinct public-domain prose, tokenised and truncated to
an exact target length, with the actual question appended at the end. The realised prompt token
length is recorded per request as `filler_tokens` and per arm-pass as `arm_pass_filler`
(`requested`, `realised`, `chars`), with the server's own `timings.t_prompt_n` beside it. It is
**reported, not asserted**: `harness/filler.py` returns the realised count rather than raising on
a shortfall, and `analysis/phase_l_ladder.txt`'s realised-depth table is where a depth that did
not actually materialise is visible instead of assumed.

Acceptance is expected to fall with depth on its own, and vLLM #47602 reports exactly that from
the other engine, so the analysis must separate "acceptance decayed" from "the per-step cost k rose".
**`harness/cost_model.py` was not used here** and no `analysis/phase_l_*_cost.txt` exists;
`harness/analyze_depth.py` makes the separation instead, in `analysis/phase_l_ladder.txt`, which
tabulates retention against each method's own 8 K rung beside acceptance per rung.
