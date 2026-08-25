"""Per-device facts and per-device safety limits.

Adding a second card breaks three assumptions that were harmless while there was only one:

1. **"Stock" is not a constant.** `gpustate.STOCK` hard-coded a 420 W power limit, which is this
   RTX 3090's default. Applying it to an RTX A6000 (300 W default) would ask for a limit the card
   may refuse or clamp, and "restore stock" would silently restore the wrong thing. Stock is now
   read from each device's own `power.default_limit`.

2. **A fixed thermal gate temperature is device-specific.** 60 C is a reachable, meaningful entry
   condition for an open-air 3090 in this chassis. A blower-cooled workstation card idles and
   cools differently; the same target could be unreachable (turning the gate into a timeout on
   every arm) or trivially met (turning it into a no-op). The gate now targets a margin above the
   device's own measured idle floor.

3. **Absolute throughput stops being comparable.** The 3090 and A6000 share an architecture
   (GA102, sm_86) but differ by roughly 18 % in memory bandwidth and 29 % in power budget, so
   tok/s means something different on each. What survives the crossing is dimensionless: speedup
   ratios, and the cost-model coefficients `k0` and `c`, which are already expressed in units of
   a plain decode step on the same device.

That third point is also what makes the second card scientifically interesting rather than merely
bigger: it is a physical ~18 % bandwidth step, against the ~4 % that clock offsets can reach.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass


def _n(v, fmt: str = "{:.0f}") -> str:
    """Format a field the driver may not report, without turning it into a number."""
    return fmt.format(v) if v is not None else "?"


@dataclass(frozen=True)
class Device:
    # The first five are what makes a device a device. The rest are reported by most cards and
    # by no means all of them, and are None when the driver says [N/A].
    index: int
    name: str
    vram_total_mib: float
    compute_cap: str
    driver: str
    power_default_w: float | None
    power_min_w: float | None
    power_max_w: float | None
    clocks_max_memory_mhz: float | None
    clocks_max_graphics_mhz: float | None

    @property
    def vram_gb(self) -> float:
        return self.vram_total_mib / 1024.0

    @property
    def model_tag(self) -> str:
        """Model name only: 'rtx3090', 'rtxa6000'."""
        return (self.name.lower()
                .replace("nvidia", "").replace("geforce", "")
                .replace(" ", "").replace("-", ""))

    @property
    def short(self) -> str:
        """Filename-safe tag. Includes the index because two identical cards in one box would
        otherwise produce the same tag and overwrite each other's outputs."""
        return f"{self.model_tag}_{self.index}"

    def describe(self) -> str:
        return (f"[{self.index}] {self.name}  {self.vram_gb:.0f} GB  sm_{self.compute_cap.replace('.','')}  "
                f"{_n(self.power_default_w)} W default ({_n(self.power_min_w)}-{_n(self.power_max_w)})  "
                f"mem {_n(self.clocks_max_memory_mhz)} MHz")


def _f(x: str) -> float | None:
    try:
        return float(x)
    except ValueError:
        return None


_FIELDS = ("index,name,memory.total,compute_cap,driver_version,"
           "power.default_limit,power.min_limit,power.max_limit,"
           "clocks.max.memory,clocks.max.graphics")


def enumerate_devices() -> list[Device]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={_FIELDS}", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return []
    devs: list[Device] = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 10:
            print(f"[devices] ignoring an nvidia-smi row with {len(p)} of 10 fields: {line!r}",
                  flush=True)
            continue
        # Coercing all ten through one float() and swallowing ValueError meant a single [N/A] -
        # which is what a card without power-limit control reports - dropped the whole device.
        # The caller then saw "no GPU at index 0", not "one field is unsupported".
        try:
            index, name, vram = int(p[0]), p[1], float(p[2])
        except ValueError:
            print(f"[devices] ignoring a row whose index or memory did not parse: {line!r}",
                  flush=True)
            continue
        devs.append(Device(
            index=index, name=name, vram_total_mib=vram,
            compute_cap=p[3], driver=p[4],
            power_default_w=_f(p[5]), power_min_w=_f(p[6]), power_max_w=_f(p[7]),
            clocks_max_memory_mhz=_f(p[8]), clocks_max_graphics_mhz=_f(p[9]),
        ))
    return devs


def ecc_mode(index: int = 0) -> str:
    """A6000-class cards support ECC and a 3090 does not; ECC also reserves some VRAM, so the
    state belongs in the environment record rather than being discovered later."""
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=ecc.mode.current", "--format=csv,noheader",
             "-i", str(index)], text=True, stderr=subprocess.DEVNULL).strip() or "unknown"
    except Exception:
        return "unknown"


def other_devices_state(index: int = 0) -> list[dict]:
    """Every OTHER visible GPU's load and temperature.

    Exclusivity is asserted only for the device under test, but a second card in the same
    chassis shares the power supply, the case airflow and the PCIe root complex. A busy
    neighbour will not trip the exclusivity check and will still move the numbers, so its state
    is recorded.
    """
    out = []
    for d in enumerate_devices():
        if d.index == index:
            continue
        try:
            csv = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits", "-i", str(d.index)],
                text=True, stderr=subprocess.DEVNULL).strip()
            mem, util, temp, pw = [float(x) for x in csv.split(",")]
            out.append({"index": d.index, "name": d.name, "memory_used_mib": mem,
                        "utilization_pct": util, "temperature_c": temp, "power_w": pw,
                        "looks_busy": util > 5 or mem > 500})
        except Exception:
            out.append({"index": d.index, "name": d.name, "error": "query failed"})
    return out


def get_device(index: int = 0) -> Device:
    for d in enumerate_devices():
        if d.index == index:
            return d
    raise RuntimeError(f"no GPU at index {index}; visible: "
                       f"{[d.describe() for d in enumerate_devices()]}")


def find_device(name_contains: str) -> Device:
    """Select by a substring of the product name, so a matrix can ask for 'A6000' by name
    rather than by an index that changes with cabling or CUDA_VISIBLE_DEVICES."""
    want = name_contains.lower().replace(" ", "")
    matches = [d for d in enumerate_devices() if want in d.name.lower().replace(" ", "")]
    if not matches:
        raise RuntimeError(
            f"no GPU matching {name_contains!r}. Visible devices:\n  " +
            "\n  ".join(d.describe() for d in enumerate_devices()))
    if len(matches) > 1:
        raise RuntimeError(f"{name_contains!r} matches {len(matches)} devices; use an index")
    return matches[0]


def assert_capacity(dev: Device, required_gb: float, what: str) -> None:
    """Refuse before loading rather than OOM halfway through a matrix."""
    if dev.vram_gb + 0.01 < required_gb:
        raise RuntimeError(
            f"{what} needs about {required_gb:.1f} GB of VRAM; device {dev.index} "
            f"({dev.name}) has {dev.vram_gb:.1f} GB. Refusing to start - an OOM partway through "
            f"a matrix wastes the arms that already ran and leaves a half-populated result file.")


def _temp(index: int) -> float | None:
    try:
        return float(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits",
             "-i", str(index)], text=True, stderr=subprocess.DEVNULL).strip())
    except Exception:
        return None


def idle_floor_c(index: int = 0, *, interval_s: float = 5.0, max_wait_s: float = 300.0,
                 stable_needed: int = 3, tol_c: float = 1.0, verbose: bool = True) -> float:
    """The temperature this device settles at with no load - waited for, not merely sampled.

    An earlier version took the minimum of six readings two seconds apart. Called at the start of
    a run that follows another run, that returns a number taken while the card is still shedding
    heat: a "floor" of, say, 78 C, and a gate target of 86 C that every arm then meets instantly.
    The gate silently becomes a no-op precisely when it is most needed.

    Stability is the span of a window, not the gap between neighbours. Comparing each sample only
    with the one before it accepts a steady fall: at nvidia-smi's integer resolution a card
    shedding 1 C per sample clears `abs(v - prev) <= 1.0` every time, so 78, 77, 76, 75 was
    declared stable and returned a 75 C floor while the card was still cooling - reintroducing the
    no-op gate this function was written to prevent. Requiring the last `stable_needed + 1`
    readings to span no more than `tol_c` separates the two: that fall spans 3 C and keeps waiting.

    If it never stabilises within `max_wait_s` it returns the minimum anyway and says so, so a
    degraded measurement is visible rather than assumed.
    """
    t0 = time.perf_counter()
    seen: list[float] = []
    while time.perf_counter() - t0 < max_wait_s:
        v = _temp(index)
        if v is None:
            time.sleep(interval_s)
            continue
        seen.append(v)
        window = seen[-(stable_needed + 1):]
        if len(window) > stable_needed and max(window) - min(window) <= tol_c:
            floor = min(seen)
            if verbose:
                print(f"[devices] idle floor {floor:.0f} C on GPU {index} "
                      f"(stabilised after {time.perf_counter()-t0:.0f}s, "
                      f"{len(seen)} samples, {seen[0]:.0f} -> {v:.0f} C, "
                      f"last {len(window)} span {max(window)-min(window):.0f} C)", flush=True)
            return floor
        time.sleep(interval_s)
    floor = min(seen) if seen else 60.0
    print(f"[devices] WARNING: idle temperature on GPU {index} never stabilised within "
          f"{max_wait_s:.0f}s; using {floor:.0f} C. The thermal gate derived from this may be "
          f"too permissive - check for another tenant or poor case airflow.", flush=True)
    return floor


def stock_state_for(dev: Device):
    """This device's own stock condition, not another card's."""
    from gpustate import GpuState
    if dev.power_default_w is None:
        raise RuntimeError(
            f"device {dev.index} ({dev.name}) does not report power.default_limit, so its stock "
            f"power limit is not known. Refusing to guess - restoring the wrong limit is the "
            f"thing this function exists to prevent.")
    return GpuState(f"stock@{dev.short}", mem_transfer_offset=0, core_offset=0,
                    power_limit_w=int(round(dev.power_default_w)))
