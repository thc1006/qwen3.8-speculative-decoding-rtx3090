"""Phase E3 -- is the offset a physical energy difference, or how it is integrated?

Phase E2 refused a fourth candidate and left one standing. Over 450 records the
averaged field's offset tracks `power_sd_w`, and it does so at TWO scales: the
between-arm ratio `offset / sd` runs 1.57 to 3.30 s, and the within-arm
regression slope reproduces each arm's own ratio to within 10 % in four arms of
six. That is the test mean power failed -- +0.863 pooled, -0.239 within -- so the
spread is the first candidate to survive it.

It is still not a mechanism. The ratio is not constant across arms (a factor of
2.1, and systematically higher for the baselines), the high-spread arms carry
intercepts the spread does not explain (12.83 J of `mtp-n2@pw420`'s 41.12), and
every between-arm number rests on SIX cells however many records sit inside them.

WHAT THIS PHASE ASKS, AND WHY IT IS THE CHEAPEST QUESTION LEFT.

A linear moving average preserves the integral of a stationary signal, so a
purely oscillating trace should contribute nothing to a difference between
integrating the smoothed field and the sharp one. That the difference scales with
the spread at all says either the averaging is not what it is documented to be,
or -- and this is the candidate -- the offset is not an energy difference but an
artefact of TRAPEZOIDAL INTEGRATION OVER A FIXED GRID.

`power.draw` is a one-second rolling average whatever rate it is queried at.
`power.draw.instant` is not. Integrating both over the same 10 Hz samples
therefore resolves one well and aliases the other, and the difference between
those two integrals would grow as the grid gets finer -- while the underlying
energy, of course, does not.

    PHYSICAL:  both integrals converge as the grid refines. The offset settles.
               `offset / sd` is INVARIANT to the sampling interval.
    ARTEFACT:  the smooth field is already resolved at 5 Hz and barely moves;
               the sharp one is not, so the offset GROWS with the sample rate
               and `offset / sd` moves with it.

Two arms, one condition, three sampling intervals. Nothing else varies.

WHY 420 W AND NOTHING ELSE. The cap stays at the card's own default, so this
phase changes no GPU state at all: no `sudo nvidia-smi -pl`, no thermal settle
between conditions, and -- the point -- no power cap to confound the spread with,
which is the confound Phase E2's own analysis could only partly control for.

WHY THESE TWO ARMS. They are the pair whose ratios differ most at one cap, 2.09
against 1.57 s, and the only thing that differs systematically between them is
the duty cycle: MTP alternates draft steps with a verify step and the baseline
does not. A moving average's edge effect does not care about the frequency of
what it is averaging. A sampling grid does.

HOW IT IS RUN. The interval is a bench flag, not a matrix property, so this is
one matrix run three times:

    for iv in 0.05 0.10 0.20; do
      python3 harness/bench.py --matrix phase_e3 --passes 3 \
          --power-interval $iv --out results/phase_e3_iv${iv/./}.json
    done

0.10 is what every phase before this one used. The other two bracket it by a
factor of two each way, which is the widest range that keeps 5 Hz giving a
six-second generation more than thirty samples.
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

# Identical to Phase A's and Phase E's, so anything comparable stays comparable.
COMMON_ARGS = [
    "-ngl", "999", "-c", "8192", "-fa", "on",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-webui", "--parallel", "1", "--jinja", "--fit", "off",
]

# The card's own default limit and no offsets: this phase sets no GPU state that
# differs from stock, so nothing it measures can be attributed to one.
STOCK = GpuState("pw420", mem_transfer_offset=0, core_offset=0, power_limit_w=420)

SPECS = [("baseline", [], False), ("mtp-n2", ["--spec-type", "draft-mtp",
                                              "--spec-draft-n-max", "2"], True)]

ARMS = [
    Arm(f"{_name}@{STOCK.name}", _args, tree="master",
        expects_drafter=_drafts, gpu_state=STOCK,
        note=f"{_name} at the card's default 420 W limit")
    for _name, _args, _drafts in SPECS
]

BASELINE_MAP = {a.name: f"baseline@{STOCK.name}" for a in ARMS}
