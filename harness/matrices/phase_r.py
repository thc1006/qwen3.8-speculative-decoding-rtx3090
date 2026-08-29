"""Phase R -- resource response. The decisive test between H2 and H2'.

Both candidate explanations for the n-max ceiling predict that deeper drafting eventually hurts,
so a throughput table cannot separate them. They differ in WHICH resource the marginal cost is
charged to, and this card lets both resources be moved independently:

    memory clock  <- GPUMemoryTransferRateOffset   (transfer-rate units = 2x the clock delta)
    compute budget <- power limit                  (the card is power-capped, not thermally capped)

Measured calibration from the pilot: removing a +400 MHz memory overclock cost the no-spec
baseline exactly 4.1 % throughput, i.e. the baseline's bandwidth elasticity is ~1.0. That gives
every other arm a reference to be compared against.

    H2'  (quantization x arithmetic intensity, the PR #27342 author's account)
         marginal cost is COMPUTE -> speculative arms should be LESS bandwidth-elastic than
         baseline and MORE power-elastic, and the gap should widen with n-max.

    H2   (Gated DeltaNet state rollback)
         marginal cost is state reconstruction, i.e. memory traffic -> speculative arms should be
         AT LEAST as bandwidth-elastic as baseline, and no more power-elastic.

Opposite signs on the same contrast. Neither is this repo's to defend.

Safety: `bench.py` applies each condition through `gpustate.apply()`, which reads the state back
and refuses to measure if it did not take effect. Stock is restored when the run ends, including
on failure. Memory overclock is the only setting that can destabilise the card; the degeneracy
screen and the per-request integrity checks will surface instability as data rather than as a
silent corruption.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import Arm      # noqa: E402
from gpustate import GpuState  # noqa: E402
import devices as _devices  # noqa: E402

# These conditions hard-code this RTX 3090's numbers: 420 W is ITS default limit and +-800
# transfer-rate offset is +-400 MHz around ITS 9751 MHz stock memory clock. Running them on
# another card would silently mean something different, so the matrix refuses rather than
# producing conditions named "stock" that are not that card's stock.
_EXPECT_DEVICE = "3090"


def _check_device(index: int = 0) -> None:
    d = _devices.get_device(index)
    if _EXPECT_DEVICE not in d.name.replace(" ", ""):
        raise RuntimeError(
            f"phase_r is calibrated for an RTX 3090 (420 W default, 9751 MHz stock memory) "
            f"but device {index} is {d.name!r} ({_devices._n(d.power_default_w)} W default, "
            f"{_devices._n(d.clocks_max_memory_mhz)} MHz). Write a device-specific condition set rather "
            f"than reusing these numbers.")



# NOT called at import. A module-level hardware call makes the matrix definition unreadable without
# a GPU: it broke the CPU-only CI job on its first run, and with it every test that imports the
# matrices to check their arms against their baselines. `bench.py` calls this after importing the
# matrix and before measuring anything, so the protection is unchanged and the file can be read
# anywhere.
PRECHECK = _check_device

REPO = Path(__file__).resolve().parent.parent.parent

TREES = {"master": REPO / "llamacpp-master"}
BINARIES = {k: v / "build/bin/llama-server" for k, v in TREES.items()}
MODEL = REPO / "models/target/Qwen3.8-27B-UD-Q4_K_XL.gguf"

COMMON_ARGS = [
    "-ngl", "999", "-c", "8192", "-fa", "on",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-webui", "--parallel", "1", "--jinja", "--fit", "off",
]

# transfer-rate offset is 2x the memory-clock delta: +-800 => +-400 MHz around stock 9751
CONDITIONS = (
    GpuState("stock",  mem_transfer_offset=0,    core_offset=0, power_limit_w=420),
    GpuState("bw-lo",  mem_transfer_offset=-800, core_offset=0, power_limit_w=420),
    GpuState("bw-hi",  mem_transfer_offset=+800, core_offset=0, power_limit_w=420),
    GpuState("pw-lo",  mem_transfer_offset=0,    core_offset=0, power_limit_w=250),
    GpuState("pw-vlo", mem_transfer_offset=0,    core_offset=0, power_limit_w=175),
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
            note=f"condition {_c.name}: mem{_c.mem_transfer_offset:+d} "
                 f"({_c.mem_clock_delta_mhz:+.0f} MHz), core{_c.core_offset:+d}, "
                 f"{_c.power_limit_w} W",
        ))

# Each arm is compared against the no-spec baseline measured under the SAME resource condition.
# Comparing across conditions would fold the resource change into the speculative effect.
BASELINE_MAP = {a.name: f"baseline@{a.gpu_state.name}" for a in ARMS}
