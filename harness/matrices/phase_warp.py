"""Forced-warp intervention: does the fork-position grouping follow the warp count?

Registered in PREREGISTRATION.md on 2026-08-25 before any of it ran. Everything observational so
far is consistent with `calc_nwarps` but consistency is not causation, so this changes the warp
count without changing the width and asks whether the grouping moves with it.

MTP alone spans the boundary. Widths 3 and 4 take four warps in the shipped table and widths 5 to
8 take two, so `mtp-n2` and `mtp-n3` sit on one side and `mtp-n4`, `mtp-n5` and `mtp-n7` on the
other. DFlash2 would add the two-drafter control but needs a second tree, and the causal test does
not need it.

The build is selected by QWEN_WARP_BUILD, which names a directory under warp/ holding a
llama-server and the libggml-cuda.so it was built against. The baseline runs in every build and
must come out byte-identical: width 1 maps to four warps in all three tables, so if the baseline
moves, the builds differ by more than the table and nothing else here can be read.
"""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import Arm  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
BUILD = os.environ.get("QWEN_WARP_BUILD", "control")
# QWEN_WARP_DIR lets a second attempt keep its builds beside the first rather than overwriting
# them. The first attempt's forced-down direction was withdrawn because its builds did not share
# a cmake configure, and keeping both sets is what makes that comparable rather than lost.
WARP_DIR = os.environ.get("QWEN_WARP_DIR", "warp")
TREES = {"warp": REPO / WARP_DIR / BUILD}
BINARIES = {"warp": REPO / WARP_DIR / BUILD / "llama-server"}
MODEL = REPO / "models/target/Qwen3.8-27B-UD-Q4_K_XL.gguf"

COMMON_ARGS = [
    "-ngl", "999", "-c", "8192", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-webui", "--parallel", "1", "--jinja", "--fit", "off",
]

ARMS = [
    Arm("baseline", [], tree="warp",
        note="width 1, four warps in every build; must be byte-identical across the three"),
    Arm("mtp-n2", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
        tree="warp", expects_drafter=True, note="width 3"),
    Arm("mtp-n3", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
        tree="warp", expects_drafter=True, note="width 4, the last on the four-warp side"),
    Arm("mtp-n4", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"],
        tree="warp", expects_drafter=True, note="width 5, the first on the two-warp side"),
    Arm("mtp-n5", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "5"],
        tree="warp", expects_drafter=True, note="width 6"),
    Arm("mtp-n7", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "7"],
        tree="warp", expects_drafter=True, note="width 8"),
]
BASELINE_MAP = {a.name: "baseline" for a in ARMS}
