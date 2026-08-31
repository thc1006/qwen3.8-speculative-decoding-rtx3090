"""Phase E6 -- the span moved at a FIXED step, which Phase E5 could not do.

Phase E5 regressed the residual surviving Phase E4's roll on the step the window straddles and
reported a slope of +19.7 ms with an intercept of +3.56 J. Correction 50 withdrew that split as
a quantity: the power cap is E5's only lever and it moves the step and the span together, a
Spearman of -0.917 across the nine cells, so three models fit the same points about equally --

    residual on the step    intercept +3.56 J   r +0.846
    residual on 1/span      intercept +1.38 J   r +0.878   <- the better fit
    residual on the span    intercept +10.00 J  r -0.847

and the intercept is a property of whichever is chosen. The step-scaled reading also agrees
with Phase E4, 5.7 J over a 284 W step being 20.1 ms against 19.7 -- but E4's figure is
attributed to the step by the same assumption, so that is consistency inside one model family.

THE MANIPULATION. One arm at the stock 420 W cap, so the step is whatever the card does between
its idle-with-model shelf and the cap and does not move. The generation length moves instead:
200, 400 and 800 tokens, which at about 41 tok/s is 4.9, 9.8 and 19.5 s of decode, and with
Phase E4's 4.0 s roll on each side gives windows of about 12.9, 17.8 and 27.5 s.

    span 12.9 -> 27.5 s, a factor of 2.1, with the step held.

THE PREDICTIONS, from the two models E5 could not separate, using their own fitted
coefficients and nothing else:

    STEP-SCALED   residual = 3.56 + 0.0197 x 287 = 9.2 J at ALL THREE lengths.
    SPAN-SCALED   residual = 1.38 + 101.14 / span = 9.2, 7.1 and 5.1 J.

They agree in the MIDDLE cell, not at the short end, and that was worked out after the first
invocation rather than before it: the pre-roll sleeps outside the sampler and only the post-roll
is inside, so a window is the decode plus 4 s and not plus 8. E5's top cap sat at a span of
13.91 s, which is the 400-token cell here, so the two models coincide there and diverge either
side of it -- about -3.3 J at the short end and +3.6 J at the long. A two-sided test, which is
stronger than the one-sided one this paragraph first described, because a systematic error in
the measurement cannot push both ends the same way. Phase E5's round-to-round spread on the residual at this cap
was 16.6 %, so about 1.5 J on a 9 J value, and three rounds put the standard error near 0.9 J.

WHAT THIS PHASE MAY NOT BE USED FOR. Everything Phase E5's entry lists, and one more: the
generation length is 200 and 800 tokens as well as this study's standard 400, so no throughput,
energy or efficiency figure from it is comparable with any other phase. It is one arm and there
is no contrast; nothing here is about speculative decoding. A rolled window's energy includes
the roll, and `energy_instruments.py` refuses to sweep a file that declares one.

THE LENGTHS ARE NOT THE STUDY'S. 400 is `prompts.MAX_TOKENS` and the two others bracket it by
a factor of two each way, which is the widest range that keeps the short cell's plateau longer
than the 1.05 s averaging window it has to contain.

    scripts/run_phase_e6.sh
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import Arm  # noqa: E402
from gpustate import GpuState  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent

TREES = {"master": REPO / "llamacpp-master"}
BINARIES = {k: v / "build/bin/llama-server" for k, v in TREES.items()}
MODEL = REPO / "models/target/Qwen3.8-27B-UD-Q4_K_XL.gguf"

# Identical to Phase A's, E's, E3's, E4's and E5's.
COMMON_ARGS = [
    "-ngl", "999", "-c", "8192", "-fa", "on",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-webui", "--parallel", "1", "--jinja", "--fit", "off",
]

# The card's own default limit: the step is held, so nothing here sets GPU state.
STOCK = GpuState("pw420", mem_transfer_offset=0, core_offset=0, power_limit_w=420)

ARMS = [
    Arm("baseline@pw420", [], tree="master", expects_drafter=False, gpu_state=STOCK,
        note="the only arm; the generation length is the manipulation and it is a bench flag")
]

# Its own baseline: one arm, no contrast, and the quantity is a property of the instrument.
BASELINE_MAP = {a.name: a.name for a in ARMS}
