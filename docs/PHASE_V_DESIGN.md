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

One contrast is better matched than the rest. DFlash2 runs on both: llama.cpp through PR #27342
with the GGUF drafter, vLLM through `{"method": "dflash", "model": "incoai/Qwen3.8-27B-DFlash2",
"num_speculative_tokens": 7}`. Depth can be matched there even though it cannot for MTP, so the
DFlash2 comparison carries more weight than the MTP one and should be read first.

## Weights

`RedHatAI/Qwen3.8-27B-INT4`, 18.14 GiB, compressed-tensors, which is vLLM's native quantised
format and comes from the people who wrote the compressor. It ships `model_mtp.safetensors`
separately, 0.79 GiB of it, so the MTP head is present rather than stripped by quantisation.

The alternative is `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, whose
`quantization_config.dynamic` excludes `.*mtp.*` from quantisation and keeps the head at BF16.
That is a cleaner MTP head but a different target quantisation, so it is a fallback rather than
the first choice.

Fit on 24 GiB: 18.14 for weights, leaving about 3.4 GiB for KV at vLLM's default
`gpu_memory_utilization` of 0.9. This model holds KV on only 16 of its 64 layers, at 4 KV heads
and 256+256 key and value length, so fp16 KV costs about 65.5 KB per token and 3.4 GiB buys
roughly 55 000 tokens. An 8192 context, matching every other phase, is not close to the limit.

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

- **H7.** The dense-hybrid speculative win reproduces on vLLM: the DFlash2 arm at matched depth
  is faster than its no-speculation baseline, with the interval clear of zero.
  - Falsified if it is not, which would make the llama.cpp result implementation-specific and
    would be the more interesting outcome of the two.
- **H7a.** The MTP arm at `num_speculative_tokens: 1` is faster than baseline but by less than
  llama.cpp's n-max 2, since k=1 forgoes the depth that llama.cpp's optimum uses.
  - This one is weakly held. It assumes the two engines' MTP paths cost the same per verified
    position, which is exactly what a change of engine is free to violate.

## Known problem on the other engine

vllm issue #40756 reports MTP speculative decoding crashing with an illegal memory access on long
sequences for Qwen3.6-27B-FP8 on vLLM 0.19.1. This study's Phase L is asking a long-context
question about the same model family on llama.cpp, where issue #27623 reports decode collapsing
past roughly 80 K. Two engines, two different failures, both in long-context speculative
decoding on this architecture. Phase V runs at 8192 and will not reach either, which is a
limitation to state rather than a problem to solve here.
