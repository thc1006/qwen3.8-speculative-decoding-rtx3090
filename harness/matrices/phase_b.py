"""Phase B -- mechanism: discriminate H2 (state rollback) from H2' (arithmetic intensity).

Both hypotheses predict that a confidence gate helps and that very deep drafting hurts, so
"does gating help?" cannot separate them. What separates them is WHAT the cost is proportional
to:

    H2  (Gated DeltaNet state rollback): cost is paid when a draft is REJECTED, because the 48
        linear-attention layers cannot roll back by truncating a KV suffix and must reconstruct
        recurrent state. Yield should track the REJECTED-token count.

    H2' (quantization x arithmetic intensity, proposed by the PR #27342 author): cost is paid
        per token DRAFTED, because a 4-bit target is less memory-bound so the marginal compute
        of each extra speculative position is proportionally more expensive. Yield should track
        the DRAFTED-token count.

The harness records, per request, both `t_draft_n` (drafted) and `t_draft_n_accepted`, and the
server log additionally reports `mean len` per request. Sweeping `--spec-draft-n-max` against
`--spec-draft-p-min` moves drafted-count and rejected-count by different amounts, which is what
makes the two regressors separable. Neither hypothesis is this repo's to defend; the analysis
reports whichever the data supports.

`--spec-draft-p-min` default on this build is 0.00 (verified from --help).
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import Arm  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent

TREES = {"master": REPO / "llamacpp-master"}
BINARIES = {k: v / "build/bin/llama-server" for k, v in TREES.items()}
MODEL = REPO / "models/target/Qwen3.8-27B-UD-Q4_K_XL.gguf"

COMMON_ARGS = [
    "-ngl", "999", "-c", "8192", "-fa", "on",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-webui", "--parallel", "1", "--jinja", "--fit", "off",
]


def _mtp(n_max: int, p_min: float) -> Arm:
    return Arm(
        f"mtp-n{n_max}-p{p_min:.2f}".replace("0.", "."),
        ["--spec-type", "draft-mtp",
         "--spec-draft-n-max", str(n_max),
         "--spec-draft-p-min", f"{p_min}"],
        tree="master", expects_drafter=True,
        note=f"n_max={n_max}, p_min={p_min}",
    )


ARMS = [Arm("baseline@master", [], tree="master", note="reference")]
for _n in (3, 7):
    for _p in (0.0, 0.50, 0.75):
        ARMS.append(_mtp(_n, _p))

BASELINE_MAP = {a.name: "baseline@master" for a in ARMS}
