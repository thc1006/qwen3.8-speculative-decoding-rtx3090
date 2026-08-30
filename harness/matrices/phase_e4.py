"""Phase E4 -- the offset happens at the window's EDGES. Is it a lag?

Phase E3 settled that the offset is a real energy difference: across a threefold
change of sampling grid the instantaneous integral did not move and stayed within
0.23 % of the driver's counter, while the averaged integral sat 0.31 to 1.86 %
below it and drifted FURTHER away. So `power.draw` is the reading that loses
energy. What it loses it TO was still unmodelled, and four candidates -- a
proportional gain error, mean power, power fluctuation, and the variation the
smoothing discards -- have each been refused.

WHAT THE COMMITTED DATA ALREADY SAYS, AND WHICH NOTHING HAS TESTED

The offset does not accumulate with the window. In 35 file-arm cells with enough
internal spread to split, the longest third of each cell has a window 1.81x the
shortest third's and an offset 1.01x it. Across 68 cells at a near-constant
400-415 W the regression is `offset_J = 28.79 + 0.339 * span_s`: almost all
intercept. The strongest single case is `phase_a_cap1600 baseline@master`, where
the window nearly doubles -- 20.3 to 39.0 s -- and the offset FALLS, 32.20 to
25.48 J.

That is observational. Window length varies inside a cell because prompts
generate at different speeds, so it is not a manipulation of the window; it is a
correlation with one. This phase manipulates it.

THE MODEL

`power.draw` is documented as a one-second rolling average. An average does two
things and only one of them is free. Smoothing is linear, so it PRESERVES the
integral of the signal under it -- which is why "the offset is the variation the
smoothing removed" could not work and did not. A LAG does not: integrating a
signal delayed by d seconds over [t0, t1] gives the integral over
[t0 - d, t1 - d], and the difference between those is

    offset  =  d * ( p(t1) - p(t0) )

whatever the trace does in between. That is a per-window quantity, it scales with
the difference between the two ENDS and not with the window's length, its mean,
its spread, or its total, and IT CAN BE NEGATIVE -- which is the one thing no
other candidate allows and which `phase_m`'s `dense-draft08b-n4` arm does
(+4.71 J on its short third, -5.79 J on its long third).

The ends are not incidental here. The sampler opens its window while the card is
falling back from the prefill calibration and closes it at full decode, so
`p(t1) - p(t0)` is very nearly the whole ramp. Taking the ramp as roughly 380 W,
the implied d is 0.068 s on `baseline@pw420` and 0.118 s on `mtp-n2@pw420`. The
RTX 3090's sensor has a 100 ms update period, which is between them. That is a
coincidence worth trying to break rather than a result.

THE INTERVENTION

`--power-roll S` holds idle inside the sampling window on BOTH sides of the
measured request. Both ends then sit in the same idle steady state, so
`p(t1) - p(t0)` goes to zero and the model's whole predicted offset goes with it.

    LAG      the offset COLLAPSES toward zero as the roll grows, and the d fitted
             from the recorded ends is the same at every roll and on both arms.
    LEAK     the offset GROWS: a per-second loss gets a longer window to work in.
    SPREAD   the offset is UNCHANGED: adding flat idle to both integrals adds the
             same energy to each, so a discarded-variation offset does not move.

Three models, three directions. Nothing here can come out ambiguous except by
landing between them, and that would itself be the finding.

Both fields' full traces are recorded (`--power-trace`), because a total cannot
say WHERE the two integrals separate: one offset is produced by a step at the
start, a step at the end, or a drift throughout, and those are three different
mechanisms. The model says the first and last tenths of a second carry all of it.

WHAT THIS PHASE MAY NOT BE USED FOR

A rolled window's energy INCLUDES the roll, so `energy_j`, `decode_energy_j`,
`sample_span_s` and every tok/J figure derived from them describe an object no
other phase measured. `design.power_roll_s` records it and
`energy_instruments.py` refuses to sweep any file that declares one. Nothing here
is a speedup or efficiency result. And as in E3, `--passes 1` does not rotate the
arm order, so arm and position within an invocation are collinear.

HOW IT IS RUN. The roll is a bench flag, not a matrix property, so this is one
matrix run three times per round, with the order of the three rolls rotated
between rounds so no roll sits in one part of the session:

    scripts/run_phase_e4.sh

WHY THESE THREE ROLLS. 0.0 reproduces E3's window exactly, which makes round-one
`roll0` a direct replication of a measurement already in the repository rather
than a new baseline to be trusted. 1.5 s is about one and a half documented
averaging windows, which is where a one-second average has to have settled if it
is a one-second average. 4.0 s is far enough past that to separate "settled" from
"still settling" without spending the night idle: at 50 records it adds 400 s to
an invocation that otherwise runs about ten minutes.
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

# Identical to Phase A's, E's and E3's, so anything comparable stays comparable.
COMMON_ARGS = [
    "-ngl", "999", "-c", "8192", "-fa", "on",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-webui", "--parallel", "1", "--jinja", "--fit", "off",
]

# The card's own default limit and no offsets, exactly as E3: this phase sets no
# GPU state that differs from stock, so nothing it measures can be blamed on one.
STOCK = GpuState("pw420", mem_transfer_offset=0, core_offset=0, power_limit_w=420)

SPECS = [("baseline", [], False), ("mtp-n2", ["--spec-type", "draft-mtp",
                                              "--spec-draft-n-max", "2"], True)]

ARMS = [
    Arm(f"{_name}@{STOCK.name}", _args, tree="master",
        expects_drafter=_drafts, gpu_state=STOCK,
        note=f"{_name} at the card's default 420 W limit, E3's arms unchanged")
    for _name, _args, _drafts in SPECS
]

BASELINE_MAP = {a.name: f"baseline@{STOCK.name}" for a in ARMS}
