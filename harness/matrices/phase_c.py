"""Phase C -- method breadth on one host under one protocol.

Nobody has run every speculative method available for this model on a single machine with a
single protocol. `sudoingX/qwen38-mtp` is crowdsourced across different builds, quants and
operating systems and discloses its own confounds; `syv-ai/qwen38-27b-rtx3090` is vLLM only;
the PR #27342 thread has DFlash2 but not the ngram family or a classic draft model.

Two questions this phase settles that others have only partly answered:

1. **Drafter quantization against a Q4 target.** The predecessor repo's v3.0 concluded that a Q4
   target collapses DFlash because the drafter was trained against BF16 target hidden states.
   The PR #27342 thread reports the opposite on a 32 GB card -- Q4_K_M, Q8_0 and BF16 drafters
   producing byte-identical output with identical acceptance -- and notes that BF16 is actively
   worse there because it crosses the VRAM ceiling. On 24 GB with a 17.6 GB target the headroom
   argument is sharper, so this is the case that is actually untested. If Q4 does not degrade
   acceptance here either, the v3.0 claim needs an erratum and this repo files one.

2. **Classic draft-model speculation.** `Qwen3.5-0.8B` has vocab 248320, identical to the target
   (read from both config.json files), so the vocab-matched classic path is available with a
   model this project already holds. The predecessor repo's headline finding was about exactly
   this path on an A3B MoE; running it on the dense-hybrid successor is the matched comparison.

The ngram family is included because two independent 3090 reports say `draft-mtp,ngram-mod`
crashes and that `ngram-cache` / `ngram-map-k` run but lose. Both are worth confirming under a
protocol that can put an interval on "lose".
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
D2 = REPO / "models/dflash2"
DRAFT_08B = REPO / "models/draft08b/Qwen3.5-0.8B-Q4_K_M.gguf"

COMMON_ARGS = [
    "-ngl", "999", "-c", "8192", "-fa", "on",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-webui", "--parallel", "1", "--jinja", "--fit", "off",
]

ARMS = [
    Arm("baseline@master", [], tree="master"),
    Arm("baseline@pr27342", [], tree="pr27342"),

    # ---- drafter-quantization ladder against the SAME Q4 target, n-max held at 4 ----
    Arm("dflash2-q4k", ["--spec-type", "draft-dflash", "--spec-draft-n-max", "4",
                        "-md", str(D2 / "Qwen3.8-27B-DFlash2-Q4_K_M.gguf")],
        tree="pr27342", expects_drafter=True, note="drafter Q4_K_M, 1.14 GB"),
    Arm("dflash2-q8", ["--spec-type", "draft-dflash", "--spec-draft-n-max", "4",
                       "-md", str(D2 / "Qwen3.8-27B-DFlash2-Q8_0.gguf")],
        tree="pr27342", expects_drafter=True, note="drafter Q8_0, 2.06 GB"),
    Arm("dflash2-bf16", ["--spec-type", "draft-dflash", "--spec-draft-n-max", "4",
                         "-md", str(D2 / "Qwen3.8-27B-DFlash2-BF16.gguf")],
        tree="pr27342", expects_drafter=True,
        note="drafter BF16, 3.86 GB — the VRAM-headroom case on a 24 GB card"),

    # ---- classic draft model, vocab-matched (248320 == 248320) ----
    Arm("draft08b-n4", ["--spec-type", "draft-simple", "--spec-draft-n-max", "4",
                        "-md", str(DRAFT_08B)],
        tree="master", expects_drafter=True, note="Qwen3.5-0.8B Q4_K_M, vocab-matched"),
    Arm("draft08b-n8", ["--spec-type", "draft-simple", "--spec-draft-n-max", "8",
                        "-md", str(DRAFT_08B)],
        tree="master", expects_drafter=True),

    # ---- ngram family: no separate model, target's own history ----
    Arm("ngram-cache", ["--spec-type", "ngram-cache", "--spec-draft-n-max", "4"],
        tree="master", expects_drafter=True),
    Arm("ngram-mod", ["--spec-type", "ngram-mod", "--spec-draft-n-max", "4"],
        tree="master", expects_drafter=True),
    Arm("ngram-map-k", ["--spec-type", "ngram-map-k", "--spec-draft-n-max", "4"],
        tree="master", expects_drafter=True),
]

BASELINE_MAP = {a.name: ("baseline@pr27342" if a.tree == "pr27342" else "baseline@master")
                for a in ARMS}
