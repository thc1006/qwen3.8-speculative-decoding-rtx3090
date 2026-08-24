# Phase L: does speculation survive the long-context decode cliff?

Target: [llama.cpp #27623](https://github.com/ggml-org/llama.cpp/issues/27623), open 2026-08-23,
**zero comments**. Reported on an RTX 4080 SUPER (sm_89): decode throughput on this model
collapses roughly 25x once the KV position passes ~80 K (33 t/s at 68 K -> 1.4 t/s at 91 K) while
prompt processing stays fast (~1300 t/s). Reproduced there across three quants.

Two things nobody has done: reproduce it on another architecture, and ask the obvious follow-up,
**does speculative decoding survive the cliff, amplify it, or mask it?** DFlash2's advertised
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
| 128 K | 4.36 GB | 21.9 GB | tight |
| 262 K | 8.92 GB | 26.5 GB | **no** - needs q4_0 (~22.4 GB, matching the community figure of 22.2 GB) |

So the cliff at ~80 K is reachable at `q8_0`, on the same KV setting as every other phase in this
repo. No quant change is needed to cross it, which keeps the comparison clean.

## Design

Depths {8 K, 32 K, 64 K, 96 K, 128 K} x methods {baseline, mtp-n2, dflash2-n4}, 3 passes.
8 K is the anchor shared with Phase A, so the long-context arms attach to an already-characterised
reference not a fresh one.

## The one thing that must not be got wrong

The context has to be filled with **real, non-repeating text**. Filling it by repeating a
paragraph would hand the ngram-style predictability of that repetition to the drafter and inflate
acceptance for a reason that has nothing to do with context depth, the same artifact two
independent 3090 reports describe when `probe.py` sends a prompt three times and the n-gram cache
scores its own history on runs two and three.

Filler will therefore be assembled from distinct public-domain prose, tokenised and truncated to
an exact target length, with the actual question appended at the end. The realised prompt token
count is recorded per request (`t_prompt_n`) and asserted against the target, so a depth that did
not actually materialise is visible instead of assumed.

Acceptance is expected to fall with depth on its own, and vLLM #47602 reports exactly that from
the other engine, so the analysis must separate "acceptance decayed" from "the per-step cost k rose".
The cost model already in `harness/cost_model.py` does that decomposition directly.
