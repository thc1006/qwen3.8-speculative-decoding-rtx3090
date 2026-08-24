"""Phase N -- localise the divergence threshold on a single drafter.

Phase A pass 1 showed greedy fork positions partitioning into exactly two groups, and the
partition tracked n-max rather than the drafter: `dflash2-n4`, `dflash2-n7` and `mtp-n5` shared
fork positions on 25/25 prompts, while `mtp-n2` and `mtp-n3` shared a different set. An
independent 1.1 GB block-diffusion drafter landing on the same character as the target's own
built-in nextn head rules out the drafter as the cause.

Speculative verification checks n+1 positions for n drafted tokens, so those groups correspond
to verification widths {3,4} and {5,6,8}. That places a boundary between width 4 and width 5 --
the neighbourhood where CUDA matmul paths commonly switch on batch size, which would change the
floating-point reduction order and flip argmax at near-ties.

This matrix walks n-max on BOTH drafters, holding everything else fixed, so the boundary can be
stated as "between n-max X and n-max X+1" instead of "somewhere". A precise boundary is what
makes the observation actionable for llama.cpp #27407; a vague one is just another report that
output differs.

It serves a second purpose that turned out to matter more. Phase A produced a cost model

    speedup = mean_len / k,      k(w) = k0 + c * (w - 1),   w = n_max + 1

with k constant across prompt classes to ~1.5 % while acceptance varied nearly ten-fold, and
with c agreeing to 1.8 % between the built-in MTP head (c = 0.2836 over widths 3/4/6, r2 = 0.9998)
and the independent DFlash2 drafter (c = 0.2786). But the DFlash2 estimate came from only TWO
widths, where a straight-line fit is perfect by construction and r2 = 1.0000 means nothing. A
coefficient that is about to carry a mechanistic claim cannot rest on two points, so DFlash2 gets
a real ladder here too.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import Arm  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent

TREES = {
    "master":  REPO / "llamacpp-master",
    "pr27342": REPO / "llamacpp-dflash2",
}
BINARIES = {k: v / "build/bin/llama-server" for k, v in TREES.items()}
MODEL = REPO / "models/target/Qwen3.8-27B-UD-Q4_K_XL.gguf"
DFLASH2_Q4 = REPO / "models/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"

COMMON_ARGS = [
    "-ngl", "999", "-c", "8192", "-fa", "on",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-webui", "--parallel", "1", "--jinja", "--fit", "off",
]

ARMS = [
    Arm("baseline@master", [], tree="master",
        note="divergence + speedup reference for the master-tree arms"),
    Arm("baseline@pr27342", [], tree="pr27342",
        note="same, for the DFlash2 arms; never compare across trees"),
]
for _n in range(1, 9):
    ARMS.append(Arm(
        f"mtp-n{_n}",
        ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(_n)],
        tree="master", expects_drafter=True,
        note=f"n-max {_n} -> verification width {_n + 1}",
    ))
for _n in (2, 4, 6, 8):
    ARMS.append(Arm(
        f"dflash2-n{_n}",
        ["--spec-type", "draft-dflash", "--spec-draft-n-max", str(_n),
         "-md", str(DFLASH2_Q4)],
        tree="pr27342", expects_drafter=True,
        note=f"n-max {_n} -> verification width {_n + 1}; four widths so c is fitted, not assumed",
    ))

BASELINE_MAP = {a.name: ("baseline@pr27342" if a.tree == "pr27342" else "baseline@master")
                for a in ARMS}
