"""Phase KV: repeat the width-grouping divergence test at f16 KV instead of q8_0.

Promised publicly in llama.cpp #25618 (comment 5396293373) and queued the same hour, because a
control that is announced and not run is worse than one that was never mentioned.

The reason it is needed is snick525's result in that thread: q8_0 KV moves greedy output on its
own, with no speculation on either side, so every fork position this study has reported was taken
under a cache that is not output-preserving. KV precision is constant across the arms here, so it
cannot by itself produce a grouping that follows verification width, but that is an argument and
this is a measurement.

Same arms and same prompts as Phase A so the two are directly comparable, with `-ctk f16 -ctv f16`
as the only change. One pass rather than five: divergence proved perfectly reproducible across
passes in Phase A, 125 of 125 prompt-passes, so repetition buys nothing here. Throughput from this
phase is not comparable to Phase A and is not reported; f16 KV moves the memory traffic.

f16 KV at 8192 context costs about 0.51 GiB against q8_0's 0.26, since only 16 of the 64 layers
hold KV at 4 heads and 256+256 key and value length. Nothing near the limit.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import Arm  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent

TREES = {"master": REPO / "llamacpp-master", "pr27342": REPO / "llamacpp-dflash2"}
BINARIES = {k: v / "build/bin/llama-server" for k, v in TREES.items()}
MODEL = REPO / "models/target/Qwen3.8-27B-UD-Q4_K_XL.gguf"
DFLASH2_Q4 = REPO / "models/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"

REQUIRES_VRAM_GB = 19.5

COMMON_ARGS = [
    "-ngl", "999", "-c", "8192", "-fa", "on",
    "-ctk", "f16", "-ctv", "f16",          # the only difference from Phase A
    "--no-webui", "--parallel", "1", "--jinja", "--fit", "off",
]

ARMS = [
    Arm("baseline@master", [], tree="master", note="f16 KV reference for the master arms"),
    Arm("baseline@pr27342", [], tree="pr27342", note="same, for the DFlash2 arms"),
    Arm("mtp-n2", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
        tree="master", expects_drafter=True, note="width 3"),
    Arm("mtp-n3", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
        tree="master", expects_drafter=True, note="width 4"),
    Arm("dflash2-n4", ["--spec-type", "draft-dflash", "--spec-draft-n-max", "4",
                       "-md", str(DFLASH2_Q4)],
        tree="pr27342", expects_drafter=True, note="width 5"),
    Arm("mtp-n5", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "5"],
        tree="master", expects_drafter=True, note="width 6"),
    Arm("dflash2-n7", ["--spec-type", "draft-dflash", "--spec-draft-n-max", "7",
                       "-md", str(DFLASH2_Q4)],
        tree="pr27342", expects_drafter=True, note="width 8"),
]

BASELINE_MAP = {a.name: ("baseline@pr27342" if a.tree == "pr27342" else "baseline@master")
                for a in ARMS}
