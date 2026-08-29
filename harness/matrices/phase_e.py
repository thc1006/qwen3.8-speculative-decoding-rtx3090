"""Phase E -- do the two energy instruments agree, and does their agreement move with load?

Every energy figure in this study comes from one instrument: `power.draw`, a field nvidia-smi
reports as a one-second rolling average, polled by a subprocess and integrated. The README asserted
-37.1 % decode energy from it and separately argued that NVIDIA's own cumulative counter "would not
settle the magnitude either", on the strength of one developer-forum report about a different card.
Probed here, `nvmlDeviceGetTotalEnergyConsumption` returns NVML_SUCCESS. This phase reads both
across the same requests.

WHAT TWO INSTRUMENTS ON ONE BOARD CAN AND CANNOT SEE. They read the same silicon at the same
instant, so a common multiplicative calibration error -- the proportional +-5 % the sensor
literature reports -- divides out of both and neither can see it. Only an external meter can. What
they CAN see is error that varies with load, and that is the term that matters, because it is the
one that does not cancel when two arms drawing different power are put in a ratio.

WHY THE POWER LIMIT AND NOT THE ARMS. The first draft of this matrix used baseline against mtp-n2
and asserted in this docstring that the speculative arm draws more power. Then the claim was
checked against Phase A, which is the order it should have been done in:

    baseline@master  415.7 W      dflash2-n4   409.8 W
    baseline@pr27342 415.7 W      mtp-n2       410.8 W

Every arm sits between 409.8 and 415.7 W -- a 5.9 W spread, 97.6 to 99.0 % of the 420 W cap. This card
is POWER-LIMITED during decode, so the arms do not differ in draw; mtp-n2 is 4.9 W BELOW the
baseline, not above it. A 1.4 % spread cannot probe a load-dependent term of the size being looked
for. The lever has to be the cap itself.

    420 W   stock, and what every other phase ran at
    250 W   the cap phase_r used; roughly 60 % of stock
    150 W   near the floor the card reports (100 W minimum)

That is a 270 W span, forty-five times the spread the arms give on their own. Both arms are kept so
that arm and cap can be told apart if they interact.

WHY NOT THE FAN. Considered and rejected. Cooling changes what the board does and both instruments
see the change, so it does not sharpen a disagreement between them. Fan state is recorded either
way as of 2026-08-29.

NOT A RE-MEASUREMENT OF PHASE A. Phase A is the preregistered primary result, deposited, and its
file is byte-identical to the copy in the v1.0.0 deposit. Nothing here replaces it. This runs
non-stock deliberately and records the state per arm-pass, which is what `--allow-non-stock` is for.

Throughput from the capped arms is NOT comparable to Phase A's and is not a finding of this phase:
a 150 W cap will make everything slower, which is the point of setting it.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import Arm  # noqa: E402
from gpustate import GpuState  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent

TREES = {"master": REPO / "llamacpp-master"}
BINARIES = {k: v / "build/bin/llama-server" for k, v in TREES.items()}
MODEL = REPO / "models/target/Qwen3.8-27B-UD-Q4_K_XL.gguf"

# Identical to Phase A's, so anything that is comparable stays comparable.
COMMON_ARGS = [
    "-ngl", "999", "-c", "8192", "-fa", "on",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-webui", "--parallel", "1", "--jinja", "--fit", "off",
]

# Clock offsets stay at zero throughout. Only the cap moves, so a difference between the two
# instruments cannot be attributed to an overclock that also changed.
CONDITIONS = [
    GpuState("pw420", mem_transfer_offset=0, core_offset=0, power_limit_w=420),
    GpuState("pw250", mem_transfer_offset=0, core_offset=0, power_limit_w=250),
    GpuState("pw150", mem_transfer_offset=0, core_offset=0, power_limit_w=150),
]

SPECS = [("baseline", [], False), ("mtp-n2", ["--spec-type", "draft-mtp",
                                              "--spec-draft-n-max", "2"], True)]

ARMS = []
for _c in CONDITIONS:
    for _name, _args, _drafts in SPECS:
        ARMS.append(Arm(
            f"{_name}@{_c.name}", _args, tree="master",
            expects_drafter=_drafts, gpu_state=_c,
            note=f"{_name} at a {_c.power_limit_w} W cap"))

# Each arm against the baseline AT ITS OWN CAP. Comparing a 150 W arm to a 420 W baseline would
# fold the cap into the effect, which is the confound the same-tree baselines exist to avoid one
# level up.
BASELINE_MAP = {a.name: f"baseline@{a.gpu_state.name}" for a in ARMS}
