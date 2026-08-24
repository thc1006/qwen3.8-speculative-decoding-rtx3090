"""Phase Q -- target-quantization ladder. Needs a 48 GB card for the upper rungs.

Why this phase exists. Two separate claims turn on target quantization and neither can be
settled on 24 GB with this model:

1. llama.cpp #25618 establishes that greedy speculative output diverges from vanilla on
   **quantized** targets while a **bf16** target preserves parity. That is a binary claim; a
   ladder turns it into a dose-response, which is more informative and harder to explain away.

2. The PR #27342 author's account of the n-max ceiling (recorded as H2' in PREREGISTRATION.md)
   is explicitly about quantization changing the compute/bandwidth ratio, measured by them as a
   per-extra-token cost of 6.7 % at BF16, 14.5 % at Q8_0 and 23.4 % at Q4_K_M. This study
   measures the same thing directly as `c`, the marginal cost per verified position in the model
   `k(w) = k0 + c*(w-1)`. Running the ladder measures `c` at each quantization, which tests
   their claim on the coefficient rather than on throughput.

Capacity, at the context this matrix actually uses (8192, not 64K - an earlier version of this
table assumed 64K and therefore overstated every rung by about 2 GB), with this model's measured
KV cost of ~34 KB/token at q8_0 and the ~1.9 GB compute buffer observed on this host:

| target        | file    | +8K KV  | +buffer | 24 GB    | 48 GB |
|---------------|--------:|--------:|--------:|:--------:|:-----:|
| UD-Q4_K_XL    | 17.56   | 17.84   | 19.74   | yes      | yes   |
| UD-Q5_K_XL    | 20.88   | 21.16   | 23.06   | **marginal (96 %)** | yes |
| UD-Q6_K_XL    | 25.30   | 25.58   | 27.48   | no       | yes   |
| Q8_0          | 29.05   | 29.33   | 31.23   | no       | yes   |
| BF16          | 49.99   | 50.27   | 52.17   | no       | **no** |

The corrected arithmetic changes the plan: `UD-Q5_K_XL` sits at 96 % of a 24 GB card at 8K
context, so it is worth *attempting* on the existing 3090 as a second rung, accepting that it may
fail to allocate. Only Q6 and Q8 genuinely require the larger card.

BF16 does not fit on 48 GB either. The bf16 control that #25618 rests on is therefore NOT
obtainable for Qwen3.8-27B on any single card considered here; it is obtainable today on the
existing 24 GB card with `unsloth/Qwen3.5-9B-MTP-GGUF` (BF16 18.41 GB), which is also the exact
model used in llama.cpp #26750. See `phase_qsmall.py`.

Disk, not VRAM, is the binding constraint for the ladder: the four target files total ~93 GB.
The runner stages them one at a time -- see `run_phase_q.sh`.

Selection is by environment variable so one matrix file serves every rung and the rung is
recorded in the result:

    QWEN_Q_TARGET=UD-Q6_K_XL python3 harness/bench.py --matrix phase_q --gpu 1 --settle-floor ...
"""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import Arm  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent

# rung -> (filename, approximate VRAM needed with 64K q8_0 KV and the compute buffer)
RUNGS = {
    "UD-Q4_K_XL": ("Qwen3.8-27B-UD-Q4_K_XL.gguf", 19.8),
    "UD-Q5_K_XL": ("Qwen3.8-27B-UD-Q5_K_XL.gguf", 23.1),
    "UD-Q6_K_XL": ("Qwen3.8-27B-UD-Q6_K_XL.gguf", 27.5),
    "Q8_0":       ("Qwen3.8-27B-Q8_0.gguf",       31.3),
}

TARGET_RUNG = os.environ.get("QWEN_Q_TARGET", "UD-Q4_K_XL")
if TARGET_RUNG not in RUNGS:
    raise RuntimeError(f"QWEN_Q_TARGET={TARGET_RUNG!r} is not one of {sorted(RUNGS)}")

_fname, REQUIRES_VRAM_GB = RUNGS[TARGET_RUNG]

# UD-Q4_K_XL is already the shared target for every other phase in this repo. Looking there
# first avoids re-downloading 17.6 GB into a second location, and avoids the risk of the two
# copies drifting apart.
_CANDIDATES = [REPO / "models/target" / _fname, REPO / "models/quant_ladder" / _fname]
MODEL = next((c for c in _CANDIDATES if c.exists()), None)
if MODEL is None:
    raise RuntimeError(
        f"target for rung {TARGET_RUNG} not found. Looked in:\n  " +
        "\n  ".join(str(c) for c in _CANDIDATES) +
        f"\nStage it with:\n"
        f"  .venv/bin/hf download unsloth/Qwen3.8-27B-GGUF {_fname} "
        f"--local-dir models/quant_ladder\n"
        f"The four rungs total ~93 GB, so run_phase_q.sh stages one at a time.")

TREES = {"master": REPO / "llamacpp-master"}
BINARIES = {k: v / "build/bin/llama-server" for k, v in TREES.items()}

COMMON_ARGS = [
    "-ngl", "999", "-c", "8192", "-fa", "on",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-webui", "--parallel", "1", "--jinja", "--fit", "off",
]

# Widths chosen to bracket the divergence boundary seen on the 3090 ({3,4} vs {5,6,8}) and to
# give three points for the k = k0 + c*(w-1) fit at every rung.
ARMS = [Arm(f"baseline@{TARGET_RUNG}", [], tree="master",
            note=f"target {TARGET_RUNG}; divergence and speedup reference for this rung")]
for _n in (2, 3, 5):
    ARMS.append(Arm(
        f"mtp-n{_n}@{TARGET_RUNG}",
        ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(_n)],
        tree="master", expects_drafter=True,
        note=f"target {TARGET_RUNG}, width {_n + 1}",
    ))

BASELINE_MAP = {a.name: f"baseline@{TARGET_RUNG}" for a in ARMS}
