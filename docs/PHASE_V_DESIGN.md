# Phase V: does the dense-model result survive a change of engine?

Written 2026-08-24, before any vLLM install. Nothing here is measured.

## Why the question is worth asking

The predecessor study's headline turned out to be engine-specific, and it took a retraction to
find that out. On llama.cpp, speculative decoding on Qwen3.6-35B-A3B was a net loss in every
configuration tested. On vLLM, with matched flags and prefix caching off, MTP at k=1 came out
27.5 % faster on the same RTX 3090. Same model, same card, opposite sign. The explanation was
that llama.cpp's draft-then-verify path uses K of 5 to 64 while vLLM's MTP uses k=1, and a MoE
verify pass over K positions loads the union of K positions' expert sets.

This study's dense-hybrid result is +59.8 % on llama.cpp. Nobody has checked whether it holds on
the other engine, and the predecessor's experience is a direct reason not to assume it does.

## What this phase can and cannot separate

It cannot isolate the engine. Three things change together and the design cannot unpick them:

- **Engine.** llama.cpp against vLLM, different schedulers, kernels and verify paths.
- **Quantisation.** llama.cpp runs GGUF `UD-Q4_K_XL`; vLLM needs compressed-tensors or GPTQ.
  There is no format both engines read well for this architecture, so the weights differ.
- **Speculation depth.** vLLM's MTP is reported to work at `num_speculative_tokens: 1` and to
  error at 2. llama.cpp's optimum here is n-max 2. The engines cannot be matched on K for MTP.

So the phase answers a directional question, not a decomposition: does the sign and rough
magnitude of the dense-hybrid speculative win survive a change of engine? A yes makes the result
a property of the model and the hardware. A no makes it a property of llama.cpp's implementation,
which is what the predecessor found for the MoE and is worth knowing either way.

Depth can be matched, but not the way it first looked. The obvious route was DFlash2, which runs
on both engines: llama.cpp through PR #27342 with the GGUF drafter, vLLM through
`{"method": "dflash", "model": "incoai/Qwen3.8-27B-DFlash2", "num_speculative_tokens": 7}`. The
arithmetic kills it on this card. The vLLM speculator ships at BF16 and is 3.58 GiB, on top of
18.14 GiB of INT4 target, which is 21.72 GiB before any KV cache. vLLM's default
`gpu_memory_utilization` of 0.9 offers 21.6 GiB, so the pair does not fit even before the 0.51 GiB
this study's 8192 context needs and the activations and CUDA graphs on top of that. Raising the
utilisation to 0.95 leaves about 1.1 GiB for everything else, which is not a margin worth trusting
a measurement to. The DFlash2 cross-engine comparison needs the 48 GB card.

So the matched contrast is at K=1 instead. vLLM's MTP is reported to work only at
`num_speculative_tokens: 1`, and llama.cpp will happily run `--spec-draft-n-max 1`; `phase_nmax`
already defines that arm as part of its 1 to 8 ladder. Matching at K=1 costs the comparison the
depth llama.cpp actually prefers, which is 2, but a matched comparison at the wrong depth is
worth more than an unmatched one at the right depth, and the llama.cpp n-max ladder gives the
depth response separately. Phase V therefore depends on `phase_nmax` having been run, and its
report should refuse to state a cross-engine ratio without it.

## Weights

`RedHatAI/Qwen3.8-27B-INT4`, 18.14 GiB (17.33 for the model, 0.79 for the MTP head),
compressed-tensors, which is vLLM's native quantised
format and comes from the people who wrote the compressor. It ships `model_mtp.safetensors`
separately, 0.79 GiB of it, so the MTP head is present rather than stripped by quantisation.

The alternative is `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, whose
`quantization_config.dynamic` excludes `.*mtp.*` from quantisation and keeps the head at BF16.
That is a cleaner MTP head but a different target quantisation, so it is a fallback rather than
the first choice.

Fit on 24 GiB, as budgeted before the run: 18.14 for weights, leaving about 3.4 GiB for KV at
vLLM's default `gpu_memory_utilization` of 0.9. This model holds KV on only 16 of its 64 layers,
at 4 KV heads and 256+256 key and value length, so fp16 KV costs about 65.5 KB per token and
3.4 GiB buys roughly 55 000 tokens. An 8192 context, matching every other phase, is not close to
the limit.

**That budget was wrong for the speculative arms, and the error was not KV.** Measured
2026-08-26 on this card, five starts, every one dying identically:

    torch.OutOfMemoryError: Tried to allocate 2.37 GiB. GPU 0 has a total capacity of
    23.56 GiB of which 2.25 GiB is free ... this process has 21.28 GiB memory in use

The 0.79 GiB above is the size of `model_mtp.safetensors` on disk. It is not what the MTP module
costs in memory. Read straight out of the safetensors headers, without downloading either file:

| file | tensors | holds `lm_head`? | holds `embed_tokens`? | size |
|---|---:|---|---|---:|
| `model.safetensors` | many | yes, BF16 `[248320, 5120]`, 2.37 GiB | yes, BF16, 2.37 GiB | 17.33 GiB |
| `model_mtp.safetensors` | 15 | **no** | **no** | 0.79 GiB |

vLLM builds both anyway. `qwen3_5_mtp.py:82` gives the predictor its own
`VocabParallelEmbedding(vocab_size, hidden_size)` and `:244` gives the wrapper its own
`ParallelLMHead(...)`, and `load_weights` then re-feeds the TARGET's tables into them -- the
`elif any(key in name for key in ["embed_tokens", "lm_head"])` branch exists for exactly that.
So the head allocates 2 x 2.37 = 4.74 GiB of BF16 to hold copies of two tensors that are already
resident, on a card where the target alone is 17.33.

The head is BF16 rather than INT4 by the checkpoint's own instruction: `config.json`'s
`quantization_config.ignore` ends with `lm_head` and `re:^mtp.*`. `tie_word_embeddings` is
`False`, so nothing is shared by configuration either.

`--gpu-memory-utilization` cannot rescue it in either direction. The drafter is loaded BEFORE the
KV cache is sized -- "Loading drafter model..." then the OOM one second later, with no "Available
KV cache memory" line anywhere in the log -- and that flag governs the KV budget. Raising it to
0.95 was tried and failed with the same allocation, as did `--enforce-eager`, cutting the draft's
`max_model_len` to 2048, and forcing `quantization: compressed-tensors` onto the speculative
config. Logs for all five are in `logs/vllm_mtp_*.log` and `logs/vllm_probe_mtp*.log`.

What this card can therefore produce is one working arm, `baseline-vllm`, and two recorded
failures. The matrix marks both MTP arms `may_fail` and `harness/vllm_bench.py` records a failed
start as a result rather than aborting the phase.

## Disk, and why this phase runs last

The install is the problem, not the weights. vLLM with torch and the CUDA libraries is 12 to
15 GB, and 18.14 GB of weights on top of that does not fit in the 31 GB free today. Phase M's
MoE target is 22 GB and is deletable once Phase M is complete and verified, which takes the free
space to roughly 53 GB. So the order is forced: Phase M, verify, delete, then Phase V.

vLLM goes in its own virtualenv. The existing `.venv` holds the harness dependencies and the
huggingface client, and letting vLLM's torch pin overwrite anything in there would silently
change the environment every other phase was measured in.

## What the harness needs

The parts that transfer: prompts, the paired interleaved design, arm rotation, the thermal gate,
cluster bootstrap, the degeneracy screen, the energy sampler. None of those know what engine is
answering.

The parts that do not:

- **Launch.** `vllm serve` with `--speculative-config` as JSON, not llama.cpp's flags.
- **Acceptance.** llama.cpp reports `draft_n` and `draft_n_accepted` per request. vLLM exposes
  `vllm:spec_decode_num_draft_tokens_total` and `vllm:spec_decode_num_accepted_tokens_total` on
  `/metrics`, which are process-wide counters, not per-request. They must be read before and
  after each request and differenced, and that only works because the harness runs one request
  at a time with `--parallel 1`.
- **The drafter assertion.** The llama.cpp version greps the server log for
  `common_speculative_init_result`. The vLLM equivalent is that the draft counter moves at all: a
  `--speculative-config` that was accepted and ignored leaves it at zero. That check is stronger
  than the log-grep, and worth keeping as the primary evidence rather than a fallback.
- **Prefix caching must be off.** `--no-enable-prefix-caching`. This is the exact confound that
  forced the predecessor's retraction, through vllm #38182, and the llama.cpp side of this study
  already pins `cache_prompt: false` for the same reason.

## Registered before running

- **H7.** The dense-hybrid speculative win reproduces on vLLM: MTP at
  `num_speculative_tokens: 1` is faster than its no-speculation baseline on vLLM, with the
  interval clear of zero.
  - Falsified if it is not, which would make the llama.cpp result implementation-specific and
    would be the more interesting outcome of the two.
- **H7a.** At the matched depth of K=1, the two engines agree to within a factor of two on the
  speedup over their own baselines.
  - Compared against llama.cpp's `mtp-n1` from `phase_nmax`, not against its n-max 2 optimum.
    Each engine is measured against its own baseline, so the quantisation difference cancels to
    first order; what does not cancel is any interaction between quantisation and acceptance,
    which llama.cpp #25618 says exists.
  - Weakly held. It assumes the two engines' MTP paths cost about the same per verified
    position, which is exactly what a change of engine is free to violate.
- **H7b.** `num_speculative_tokens: 2` fails on vLLM for this model family, as reported.
  - Recorded as a result either way. It is cheap to test, it is the reason the comparison sits
    at K=1 rather than at llama.cpp's optimum, and a version where it works would change what
    this phase can do.

## Known problem on the other engine

vllm issue #40756 reports MTP speculative decoding crashing with an illegal memory access on long
sequences for Qwen3.6-27B-FP8 on vLLM 0.19.1. This study's Phase L is asking a long-context
question about the same model family on llama.cpp, where issue #27623 reports decode collapsing
past roughly 80 K. Two engines, two different failures, both in long-context speculative
decoding on this architecture. Phase V runs at 8192 and will not reach either, which is a
limitation to state rather than a problem to solve here.
