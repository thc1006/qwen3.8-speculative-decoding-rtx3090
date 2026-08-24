"""Phase R2: resource response with the clock PINNED instead of squeezed.

Phase R answered the mechanism question and its two identity controls passed, but its own
review found three defects in how the compute lever was applied. All three come from using a
power cap to vary compute, and all three disappear if the clock is pinned directly.

1. A power cap does not produce a stable operating point. Within-request SM clock spread,
   measured as (mean - min) / mean, was 1.3-2.3 % at stock but 12-13 % at 250 W and
   16.8 / 27.2 / 43.8 % at 175 W for baseline / mtp-n3 / mtp-n7. At the bottom of the ladder the
   card oscillates against the cap instead of settling, so the mean is a poor description of
   where it ran, and the elasticity denominator is correspondingly soft.

2. Under one cap, different methods settle at different clocks: 906, 1081 and 1178 MHz at 250 W.
   A bandwidth-heavy workload spends more of the budget on memory, so an interval labelled the
   same for every method actually spans a different clock range for each. Since elasticity is
   regime-dependent, that is not a matched comparison.

3. At a fixed cap, raising the memory clock takes power from the core (1799 -> 1759 MHz across
   the bandwidth sweep), so the bandwidth lever moved compute in the opposite direction and its
   elasticity had to be corrected after the fact.

`nvidia-smi -lgc` pins the graphics clock exactly. Verified on this card: locked at 900 MHz it
reads 900 MHz on four consecutive samples under load. That gives a stable operating point, the
SAME clock for every method, and a core that cannot be robbed by the memory clock. The three
defects are fixed at the source instead of corrected in the analysis.

The ladder is 600 / 1200 / 1700 MHz, all exactly supported by this card (129 supported values
between 210 and 2130 MHz). 1800 was rejected as a top rung: the card runs at ~1785 MHz under
load at stock, so pinning to 1800 leaves no headroom and would risk failing the state check
mid-matrix.

The bandwidth axis is run at TWO anchors, 1200 and 1700 MHz, and that is a change forced by the
pre-flight. Pinned at 1200 MHz the no-spec baseline's bandwidth elasticity measures 0.494, well
below the 0.75 Phase R found at its natural ~1790 MHz. That is not a contradiction: at a lower
core clock the card is more compute-constrained, so bandwidth matters less. It does raise the
question the whole phase turns on, though, which is whether the baseline-to-speculative RATIO is
itself regime-dependent. One anchor cannot answer that. Two can.

Phase R is not superseded. It measured the same thing through a different lever, and agreement
between a squeezed-power and a pinned-clock design would be a stronger result than either alone.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import Arm          # noqa: E402
from gpustate import GpuState  # noqa: E402
import devices as _devices     # noqa: E402

_EXPECT_DEVICE = "3090"
_d = _devices.get_device(0)
if _EXPECT_DEVICE not in _d.name.replace(" ", ""):
    raise RuntimeError(
        f"phase_r2's clock ladder (600/1200/1700 MHz) and 420 W limit are this RTX 3090's "
        f"numbers; device 0 is {_d.name!r}. Pick rungs for that card instead of reusing these.")

REPO = Path(__file__).resolve().parent.parent.parent
TREES = {"master": REPO / "llamacpp-master"}
BINARIES = {k: v / "build/bin/llama-server" for k, v in TREES.items()}
MODEL = REPO / "models/target/Qwen3.8-27B-UD-Q4_K_XL.gguf"

COMMON_ARGS = [
    "-ngl", "999", "-c", "8192", "-fa", "on",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-webui", "--parallel", "1", "--jinja", "--fit", "off",
]

# transfer-rate offset is 2x the memory-clock delta: +-800 is +-400 MHz around stock 9751
CONDITIONS = (
    GpuState("sm600",       mem_transfer_offset=0,    power_limit_w=420, lock_sm_mhz=600),
    GpuState("sm1200",      mem_transfer_offset=0,    power_limit_w=420, lock_sm_mhz=1200),
    GpuState("sm1700",      mem_transfer_offset=0,    power_limit_w=420, lock_sm_mhz=1700),
    GpuState("sm1200-bwlo", mem_transfer_offset=-800, power_limit_w=420, lock_sm_mhz=1200),
    GpuState("sm1200-bwhi", mem_transfer_offset=+800, power_limit_w=420, lock_sm_mhz=1200),
    GpuState("sm1700-bwlo", mem_transfer_offset=-800, power_limit_w=420, lock_sm_mhz=1700),
    GpuState("sm1700-bwhi", mem_transfer_offset=+800, power_limit_w=420, lock_sm_mhz=1700),
)

METHODS = (
    ("baseline", [], False),
    ("mtp-n3", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"], True),
    ("mtp-n7", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "7"], True),
)

ARMS = []
for _c in CONDITIONS:
    for _m, _args, _drafts in METHODS:
        ARMS.append(Arm(
            f"{_m}@{_c.name}", list(_args), tree="master",
            expects_drafter=_drafts, gpu_state=_c,
            note=f"SM pinned {_c.lock_sm_mhz} MHz, mem offset {_c.mem_transfer_offset:+d} "
                 f"({_c.mem_clock_delta_mhz:+.0f} MHz), {_c.power_limit_w} W",
        ))

# Compared against the no-spec baseline measured under the SAME condition. Comparing across
# conditions would fold the resource change into the speculative effect.
BASELINE_MAP = {a.name: f"baseline@{a.gpu_state.name}" for a in ARMS}
