"""Phase V arms: the same question on vLLM instead of llama.cpp.

Consumed by the vLLM driver, not by `bench.py`. The arm list is data, so it is defined and
reviewable now even though the run loop that reads it waits for an installed vLLM to be tested
against. See `docs/PHASE_V_DESIGN.md` for what this phase can and cannot separate.

Depth is matched at K=1, not at llama.cpp's optimum of 2. vLLM's MTP is reported to work only at
`num_speculative_tokens: 1`; the `mtp-k2` arm below exists to test that report rather than to
trust it, and is expected to fail. A DFlash2 arm is absent on purpose: the vLLM speculator is
3.58 GiB at BF16 on top of an 18.14 GiB target, which does not fit beside a KV cache on 24 GiB.
"""
import json

TARGET = "RedHatAI/Qwen3.8-27B-INT4"          # compressed-tensors, ships model_mtp.safetensors
SPECULATOR = "incoai/Qwen3.8-27B-DFlash2"     # 3.58 GiB BF16; needs the 48 GB card

# `--no-enable-prefix-caching` is the confound that forced the predecessor's retraction, through
# vllm #38182, and the llama.cpp side of this study pins `cache_prompt: false` for the same
# reason. `--max-num-seqs 1` is not a performance choice: vLLM's speculative counters are
# process-wide, so a per-request acceptance figure is only attributable with one sequence in
# flight.
COMMON_ARGS = [
    "--max-model-len", "8192",
    "--no-enable-prefix-caching",
    "--max-num-seqs", "1",
    # 0.95 rather than vLLM's default 0.90, and it does NOT rescue the speculative arms.
    # Measured 2026-08-26, five starts, logs in logs/vllm_mtp_*.log and logs/vllm_probe_mtp*.log:
    # 0.90, 0.95, 0.95 with --enforce-eager, 0.95 with the draft's max_model_len cut to 2048, and
    # 0.95 with quantization: compressed-tensors forced on the speculative config. Every one died
    # with the SAME allocation:
    #     torch.OutOfMemoryError: Tried to allocate 2.37 GiB. GPU 0 has a total capacity of
    #     23.56 GiB of which 2.25 GiB is free ... this process has 21.28 GiB memory in use
    # at vllm/model_executor/models/qwen3_5_mtp.py:244, `self.lm_head = ParallelLMHead(...)`.
    #
    # 2.37 GiB is exactly vocab_size 248320 * hidden_size 5120 * 2 bytes. The head is BF16 because
    # the checkpoint's own recipe says so: config.json's quantization_config.ignore ends with
    # 'lm_head' and 're:^mtp.*', so every MTP weight is excluded from the INT4 quantization that
    # makes the target fit at all. tie_word_embeddings is False, so nothing is shared -- the MTP
    # module builds its own embed_tokens AND its own lm_head, 2.37 GiB each, on top of a target
    # that already carries one.
    #
    # gpu_memory_utilization cannot fix this in either direction: the drafter is loaded BEFORE the
    # KV cache is sized, which the logs show directly -- "Loading drafter model..." appears and
    # the OOM follows one second later, with no "Available KV cache memory" line anywhere in the
    # file. The utilization figure governs the KV budget, and the KV budget is not what is short.
    #
    # An earlier version of this comment reasoned "0.95 of 23.56 GiB = 22.4 GiB; weights 18.12 +
    # head 2.37 = 20.49, leaving 1.9 GiB" and concluded that 0.95 would fit. It does not, and the
    # log proving it was already on disk when that was written. The value stays at 0.95 for the
    # baseline arm's KV headroom, not as a remedy.
    "--gpu-memory-utilization", "0.95",
    # vLLM 0.27.1 renamed this. `--disable-log-requests` no longer exists and an unknown flag
    # stops the server before it serves anything, so the whole phase would have failed at
    # startup with a message about argparse rather than about speculation. The new pair is
    # `--enable-log-requests` / `--no-enable-log-requests`, defaulting to False -- the explicit
    # negative is used anyway, for the same reason every sampler here is pinned: a default is a
    # thing that changes between versions without the result file showing it.
    "--no-enable-log-requests",
    "--seed", "20260824",
]


def _spec(cfg: dict) -> list[str]:
    return ["--speculative-config", json.dumps(cfg, separators=(",", ":"))]


ARMS = [
    {"name": "baseline-vllm", "args": [], "expects_drafter": False,
     "note": "no speculation; the reference every vLLM arm is measured against"},
    {"name": "mtp-k1", "args": _spec({"method": "mtp", "num_speculative_tokens": 1}),
     "expects_drafter": True, "may_fail": True,
     "note": "matched to llama.cpp mtp-n1 from phase_nmax; the only depth both engines run. "
             "Known to fail on a 24 GiB card: the MTP module's BF16 lm_head and embed_tokens "
             "are 2.37 GiB each and the checkpoint excludes 're:^mtp.*' from its INT4 recipe. "
             "The arm is kept so the failure is recorded rather than remembered, and so the "
             "same matrix runs unchanged on a card that can hold it"},
    {"name": "mtp-k2", "args": _spec({"method": "mtp", "num_speculative_tokens": 2}),
     "expects_drafter": True, "may_fail": True,
     "note": "reported to error on this model family. Tested, not assumed; a failure is the "
             "result and is recorded as one"},
]

# Only reachable on a card that can hold target and speculator together.
A6000_ONLY_ARMS = [
    {"name": "dflash2-k4",
     "args": _spec({"method": "dflash", "model": SPECULATOR, "num_speculative_tokens": 4}),
     "expects_drafter": True, "requires_vram_gb": 40.0,
     "note": "matched to llama.cpp dflash2-n4, which is its best depth"},
    {"name": "dflash2-k7",
     "args": _spec({"method": "dflash", "model": SPECULATOR, "num_speculative_tokens": 7}),
     "expects_drafter": True, "requires_vram_gb": 40.0,
     "note": "the depth the vLLM recipe for this speculator recommends"},
]

BASELINE = "baseline-vllm"
# 17.71 GiB of weights as vLLM reports them for the baseline arm, plus 1.38 GiB of KV at 8192 and
# the profiling activations. Measured, not budgeted: logs/vllm_probe_baseline.log says
# "Model loading took 17.71 GiB" and "Available KV cache memory: 1.38 GiB" at 0.90.
# This gate is what `vllm_bench.assert_arms_fit` checks before anything is loaded; before that
# function existed nothing read it.
REQUIRES_VRAM_GB = 21.0
