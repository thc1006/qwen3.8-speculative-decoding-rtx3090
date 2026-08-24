"""Phase Q-small -- the quantization ladder INCLUDING bf16, on the existing 24 GB card.

llama.cpp #25618's central scoping claim is that greedy speculative output diverges from vanilla
on quantized targets but stays bit-identical on a **bf16** target. For Qwen3.8-27B that control
is unobtainable: BF16 is 50 GB, so it does not fit on 24 GB and does not fit on 48 GB either.

It is obtainable today, on the card already here, by changing model rather than hardware.
`unsloth/Qwen3.5-9B-MTP-GGUF` ships the full ladder and BF16 is 18.41 GB:

| rung    | file    | fits 24 GB |
|---------|--------:|:----------:|
| Q4_K_M  |  5.87   | yes |
| Q6_K    |  7.68   | yes |
| Q8_0    |  9.79   | yes |
| BF16    | 18.41   | yes (at modest context) |

Two things make this worth running even though the model is not this study's headline model:

- It supplies the **bf16 anchor** that the 27B ladder structurally cannot.
- `Qwen3.5-9B-MTP` `Q4_K_M` is **the exact model and quant used in llama.cpp #26750**, the report
  claiming MTP acceptance collapses on CUDA (35.8-40.7 %) versus Vulkan (~92 %). Running it here
  produces a directly comparable CUDA datapoint on a different CUDA architecture (sm_86 Ampere
  against their sm_120 Blackwell) with an acceptance-versus-depth curve their report lacks.

Rung is selected by environment variable and recorded in the result:

    QWEN_QS_TARGET=BF16 python3 harness/bench.py --matrix phase_qsmall ...
"""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import Arm  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent

# rung -> (filename, approximate VRAM with the context below plus the compute buffer)
RUNGS = {
    "Q4_K_M": ("Qwen3.5-9B-Q4_K_M.gguf",  9.0),
    "Q6_K":   ("Qwen3.5-9B-Q6_K.gguf",   10.8),
    "Q8_0":   ("Qwen3.5-9B-Q8_0.gguf",   12.9),
    "BF16":   ("Qwen3.5-9B-BF16.gguf",   21.8),
}

TARGET_RUNG = os.environ.get("QWEN_QS_TARGET", "Q4_K_M")
if TARGET_RUNG not in RUNGS:
    raise RuntimeError(f"QWEN_QS_TARGET={TARGET_RUNG!r} is not one of {sorted(RUNGS)}")

_fname, REQUIRES_VRAM_GB = RUNGS[TARGET_RUNG]
MODEL = REPO / "models/qwen35_9b" / _fname
if not MODEL.exists():
    raise RuntimeError(
        f"{MODEL} is missing. Stage it first:\n"
        f"  hf download unsloth/Qwen3.5-9B-MTP-GGUF {_fname} --local-dir models/qwen35_9b")

TREES = {"master": REPO / "llamacpp-master"}
BINARIES = {k: v / "build/bin/llama-server" for k, v in TREES.items()}

# BF16 leaves little headroom on 24 GB, so context is held at 8192 for EVERY rung rather than
# letting the small rungs use more. An unequal context across rungs would put KV depth into the
# comparison alongside quantization.
COMMON_ARGS = [
    "-ngl", "999", "-c", "8192", "-fa", "on",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-webui", "--parallel", "1", "--jinja", "--fit", "off",
]

ARMS = [Arm(f"baseline@{TARGET_RUNG}", [], tree="master",
            note=f"Qwen3.5-9B target {TARGET_RUNG}")]
for _n in (2, 3, 5, 6):
    ARMS.append(Arm(
        f"mtp-n{_n}@{TARGET_RUNG}",
        ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(_n)],
        tree="master", expects_drafter=True,
        note=f"width {_n + 1}" + ("; n-max 6 matches llama.cpp #26750" if _n == 6 else ""),
    ))

BASELINE_MAP = {a.name: f"baseline@{TARGET_RUNG}" for a in ARMS}
