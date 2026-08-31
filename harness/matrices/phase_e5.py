"""Phase E5 -- does the residual that survives the roll scale with the step it straddles?

Phase E4 measured `power.draw`'s averaging width, showed the closed-form boxcar loss accounts
for the whole unrolled offset, and left **5.7 J on both arms** surviving the longest roll. It
also reported that residual as arm-independent and called that a reason to think it a different
kind of object. That inference does not hold, and this phase exists because checking it turned
up the reason.

WHY THE ARM-EQUALITY MEANT NOTHING. At roll 4 s both arms' windows hold the same excursion --
idle, up to the 420 W cap, back to idle -- and the steps are the same size to within 3 %:
287.6 W up on `baseline` against 286.4 on `mtp-n2`, 277.1 down against 279.4. Phase E4's own
running-difference table says the same thing, +168.67 J of head against +173.56 and -158.68 J of
middle against -159.32. So ANY residual whose size is set by the step is predicted to be equal on
the two arms. Equality was evidence of nothing.

WHERE THE RESIDUAL IS NOT. It is not a per-second loss. Splitting each record at the region where
the instantaneous field sits above 80 % of its own maximum and comparing the two fields' means
there, the plateau contributes **0.6 to 0.9 J of the 6.4** and the rest is in the edges -- on an
arm whose plateau runs 7.7 s and one whose plateau runs 4.1 s. A lag cannot produce a plateau
term at all, since on a flat stretch a delayed copy equals the original, and the measured plateau
term is small enough to say the residual does not live there.

WHAT THE COMMITTED DATA CANNOT SETTLE. If the residual is an edge effect that the boxcar model
gets slightly wrong -- the two fields' rise and fall not being mirror images -- then it is
`(lag asymmetry) x (step)` and scales with the step. If it is a fixed per-window quantity it does
not. Records vary in step on their own, but only over 175 to 248 W, a factor of 1.4, and
regressing the residual on it gives +78 ms on one arm and -214 ms on the other with correlations
of +0.12 and -0.19. That is not a weak answer, it is no answer: the range is too narrow and the
per-record noise too large.

THE DESIGN. The power cap sets how far the card climbs above its idle-with-model draw of about
128 W, so it sets the step directly:

    420 W cap    step about 290 W
    250 W cap    step about 120 W
    150 W cap    step about  20 W

which is a range of roughly 13x rather than 1.4x. One arm, the baseline, at three caps, with the
4.0 s roll and the traces that Phase E4 established.

    EDGE      residual falls with the step, and residual / step is one number in seconds
              across all three caps.
    FIXED     residual stays near 5.7 J at every cap.

The noise floor on the offset is 10 to 30 % round to round, so 5.7 J against a predicted 2.4 at
the middle cap and 0.4 at the lowest is a separation the design can see; 5.7 against 5.7 is too.

WHY THE ARM ORDER IS ROTATED HERE AND WAS NOT IN E3 OR E4. Those compared a phase's own arms
across invocations, so a fixed arm order inside an invocation could not touch the estimand. Here
the contrast IS between arms, because the cap is what the arm names, and a fixed order would put
the cap and the position within the pass on the same axis. Three passes over three arms makes
`rot = (p_idx - 1) % len(arms)` close, so each cap visits each position exactly once.

    python3 harness/bench.py --matrix phase_e5 --passes 3 \
        --power-roll 4.0 --power-trace --out results/phase_e5.json

`--passes 3` and NOT `--latin-arms`, which sets the same 3 and is the flag that says why.
bench.py writes its `design` block -- `"passes": passes` included -- BEFORE that flag's override
reassigns the variable, so a run under it records the pre-override count. The first attempt at
this phase recorded `design.passes = 5` while three passes ran, and `audit_results.py` correctly
called the file 225 of 375 records and failed it. Checked against the audit's own code rather
than inferred, then confirmed in the partial file the killed run left behind. The flag is worth
fixing; doing it in the hour before a two-hour run is not, because nothing there would have been
through the gate.

WHAT THIS PHASE MAY NOT BE USED FOR. The same limits as E4, and one more. A rolled window's
energy includes the roll, so no efficiency or tok/J figure here means anything, and
`energy_instruments.py` refuses to sweep a file that declares a roll. The three caps are here to
move the step, not because anyone would run the card at 150 W; nothing here is a speedup result.
And the arms differ ONLY in the cap, so this phase says nothing about speculative decoding.
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

# Identical to Phase A's, E's, E3's and E4's, so anything comparable stays comparable.
COMMON_ARGS = [
    "-ngl", "999", "-c", "8192", "-fa", "on",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-webui", "--parallel", "1", "--jinja", "--fit", "off",
]

# The same three caps Phase E used, so the levels each one produces are already on record.
CONDITIONS = [
    GpuState("pw420", mem_transfer_offset=0, core_offset=0, power_limit_w=420),
    GpuState("pw250", mem_transfer_offset=0, core_offset=0, power_limit_w=250),
    GpuState("pw150", mem_transfer_offset=0, core_offset=0, power_limit_w=150),
]

ARMS = [
    Arm(f"baseline@{_c.name}", [], tree="master",
        expects_drafter=False, gpu_state=_c,
        note=f"baseline at a {_c.power_limit_w} W cap, which sets the step the window straddles")
    for _c in CONDITIONS
]

# Each arm is its own baseline: this phase has no speculative arm and no effect to estimate.
# The quantity of interest is a property of the instrument within each arm.
BASELINE_MAP = {a.name: a.name for a in ARMS}
