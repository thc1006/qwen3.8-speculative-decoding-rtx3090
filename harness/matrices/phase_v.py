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
    "--gpu-memory-utilization", "0.90",
    "--disable-log-requests",
    "--seed", "20260824",
]


def _spec(cfg: dict) -> list[str]:
    return ["--speculative-config", json.dumps(cfg, separators=(",", ":"))]


ARMS = [
    {"name": "baseline-vllm", "args": [], "expects_drafter": False,
     "note": "no speculation; the reference every vLLM arm is measured against"},
    {"name": "mtp-k1", "args": _spec({"method": "mtp", "num_speculative_tokens": 1}),
     "expects_drafter": True,
     "note": "matched to llama.cpp mtp-n1 from phase_nmax; the only depth both engines run"},
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
# 18.14 target + 0.51 KV at 8192 + activations and CUDA graphs
REQUIRES_VRAM_GB = 21.0
