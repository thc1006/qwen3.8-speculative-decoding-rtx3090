"""Phase A -- primary confirmatory matrix.

Answers the pre-registered primary endpoints:
  * Is MTP a net win on this card, and where is its n-max optimum?
  * Is DFlash2 a net win on consumer Ampere with a Q4 target, and how does it compare to MTP?
  * Does the unmerged PR #27342 branch move the no-spec baseline at all? (dual-tree control)

Flags verified against `llama-server --help` on this exact build (0.2.0-dev, build 200,
commit c060ca9): `--spec-type` accepts
none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,ngram-simple,ngram-map-k,
ngram-map-k4v,ngram-mod,ngram-cache. `--spec-draft-n-max` defaults to 3, not 2.

`draft-eagle3` is supported by the engine but is NOT in this matrix: no EAGLE3 drafter has been
published for Qwen3.8-27B (checked on the HF hub, 2026-08-24), so the method cannot be run.
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

# Identical for every arm. `--parallel 1` is not optional: speculative decoding is a
# single-stream optimisation, and a parallel-2 baseline reads roughly 20% low, which inflates
# every speedup computed against it. `--fit off` stops llama.cpp silently re-deciding the layer
# split between arms.
COMMON_ARGS = [
    "-ngl", "999",
    "-c", "8192",
    "-fa", "on",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-webui",
    "--parallel", "1",
    "--jinja",
    "--fit", "off",
]

ARMS = [
    # dual-tree no-spec controls
    Arm("baseline@master", [], tree="master",
        note="reference for every master-tree arm"),
    Arm("baseline@pr27342", [], tree="pr27342",
        note="reference for DFlash2 arms; its gap to baseline@master IS the build confound"),

    # built-in MTP head, n-max sweep (llama.cpp default is 3)
    Arm("mtp-n2", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
        tree="master", expects_drafter=True),
    Arm("mtp-n3", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
        tree="master", expects_drafter=True, note="engine default"),
    Arm("mtp-n5", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "5"],
        tree="master", expects_drafter=True),

    # DFlash2 via PR #27342, block size 8 -> author recommends n-max 7
    Arm("dflash2-n4", ["--spec-type", "draft-dflash", "--spec-draft-n-max", "4",
                       "-md", str(DFLASH2_Q4)],
        tree="pr27342", expects_drafter=True),
    Arm("dflash2-n7", ["--spec-type", "draft-dflash", "--spec-draft-n-max", "7",
                       "-md", str(DFLASH2_Q4)],
        tree="pr27342", expects_drafter=True, note="PR-recommended, matches block size 8"),
]

# Each arm is compared against the baseline built from the SAME tree.
BASELINE_MAP = {a.name: ("baseline@pr27342" if a.tree == "pr27342" else "baseline@master")
                for a in ARMS}
