"""Apply, verify and restore GPU clock/power state - with a guard against doing it mid-run.

Two things this module exists to prevent, both of which happened during development:

1. **An undisclosed overclock.** The card arrived with memory +400 MHz and core +100 MHz while
   the write-up said "stock". Because batch-1 decode is bandwidth-bound and speculative
   verification is comparatively compute-dense, a memory overclock moves the two arms by
   different amounts - the exact differential a paired design is supposed to exclude.

2. **Changing clocks while a benchmark was live.** Probing whether the memory clock could be
   underclocked was done while a run was measuring, briefly putting the card 200 MHz below its
   own baseline condition. No dip was visible in the trace, but "no dip visible" is not
   evidence of no effect, and that run was discarded. A benchmark run now holds a lockfile and
   `apply()` refuses while it exists.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

LOCKFILE = Path(__file__).resolve().parent.parent / ".gpu-in-use.lock"


class GpuStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuState:
    """A resource condition.

    `mem_transfer_offset` is in transfer-rate units, which are TWICE the memory clock delta:
    +800 raises the memory clock by 400 MHz.

    `lock_sm_mhz` pins the graphics clock with `nvidia-smi -lgc`. Prefer it over squeezing the
    power limit when the point is to vary compute. Measured on this card, a power cap does not
    produce a stable operating point: at 175 W the achieved SM clock oscillates with a
    within-request spread of 17 % for the no-spec baseline and 44 % for the deepest speculative
    arm, so the mean is a poor description of where the card actually ran. A power cap also lets
    each method settle at a DIFFERENT clock (906, 1081 and 1178 MHz for baseline, mtp-n3 and
    mtp-n7 at 250 W), because a bandwidth-heavy workload spends more of the budget on memory, so
    an interval labelled the same for every method spans a different clock range for each.
    Pinning removes both problems, and it also removes a third: with the core pinned, raising the
    memory clock can no longer steal power from it, which was contaminating the bandwidth lever.
    """
    name: str
    mem_transfer_offset: int = 0
    core_offset: int = 0
    power_limit_w: int = 420
    lock_sm_mhz: int | None = None

    @property
    def mem_clock_delta_mhz(self) -> float:
        return self.mem_transfer_offset / 2.0


def _settings(args: list[str]) -> str:
    env = dict(os.environ)
    env.setdefault("DISPLAY", ":0")
    return subprocess.check_output(["nvidia-settings", *args], text=True,
                                   stderr=subprocess.DEVNULL, env=env)


def _smi_uuid(index: int) -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", str(index)],
            text=True, stderr=subprocess.DEVNULL).strip() or None
    except Exception:
        return None


_SETTINGS_INDEX_CACHE: dict[int, int] = {}


def settings_index_for(smi_index: int) -> int:
    """Map an nvidia-smi GPU index to the nvidia-settings `[gpu:N]` index, verified by UUID.

    These two tools maintain independent enumerations. On a single-GPU host they trivially agree,
    which is exactly why the mismatch is easy to miss until a second card is installed - at which
    point clock offsets could be applied to one card while another is being measured, silently
    and with no error. Both tools report the same GPU UUID, so the mapping is resolved by
    matching on that rather than assumed.
    """
    if smi_index in _SETTINGS_INDEX_CACHE:
        return _SETTINGS_INDEX_CACHE[smi_index]

    want = _smi_uuid(smi_index)
    if not want:
        raise GpuStateError(
            f"could not read the UUID of nvidia-smi GPU {smi_index}; refusing to guess which "
            f"nvidia-settings GPU it corresponds to")

    for cand in range(8):
        try:
            out = _settings(["-q", f"[gpu:{cand}]/GPUUUID"])
        except Exception:
            continue
        for line in out.splitlines():
            if "):" in line and want in line:
                _SETTINGS_INDEX_CACHE[smi_index] = cand
                return cand

    raise GpuStateError(
        f"no nvidia-settings GPU reports UUID {want} (nvidia-smi index {smi_index}). "
        f"Clock offsets cannot be applied safely without a verified mapping. Is an X server "
        f"running with Coolbits enabled for this card?")


_FAN_ATTR = re.compile(r"Attribute '(\w+)' \([^)]*\[(gpu|fan):(\d+)\]\):\s*([-\d.]+)")


def parse_fan_query(stdout: str) -> dict:
    """Pull the fan attributes out of nvidia-settings' output.

    Split from the subprocess call so the parsing rule can be tested without an X server, a
    driver, or a card -- the same reason `telemetry.host_load` grew a `_ps_output` seam.
    """
    out: dict = {"fan_targets_pct": {}, "fan_current_pct": {}}
    for line in stdout.splitlines():
        m = _FAN_ATTR.search(line)
        if not m:
            continue
        attr, kind, idx, val = m.group(1), m.group(2), m.group(3), m.group(4)
        try:
            num = float(val)
        except ValueError:
            continue
        if attr == "GPUFanControlState" and kind == "gpu":
            out["fan_control_state"] = int(num)
        elif attr == "GPUTargetFanSpeed" and kind == "fan":
            out["fan_targets_pct"][idx] = num
        elif attr == "GPUCurrentFanSpeed" and kind == "fan":
            out["fan_current_pct"][idx] = num
    if "fan_control_state" not in out:
        out["fan_error"] = "GPUFanControlState did not resolve"
    out["fan_count"] = len(out["fan_targets_pct"])
    # 0 = driver-controlled, 1 = a human or a tool has taken the fans over. Spelled out because a
    # bare 0/1 in a result file read a year from now is a coin flip for whoever reads it.
    out["fan_control"] = {0: "auto", 1: "manual"}.get(out.get("fan_control_state"), "unknown")
    return out


def fan_state(sidx: int, *, max_fans: int = 4) -> dict:
    """Fan control mode and per-fan target/current speed, in one nvidia-settings call.

    Nothing in this harness recorded a fan anywhere until 2026-08-29. `overclock_state` captures
    clock offsets and power limits precisely so an overclock can never be silently in effect, and
    a fan curve is the same intervention by another route: a card held cooler holds a higher
    sustained clock, and Phase B's own telemetry has every arm losing 8-12 % of its clock inside a
    single arm-pass. Someone who changed the curve between two runs would have left no field
    saying so -- only temperatures that were quietly lower, which reads as a cooler room.

    Two values per fan, because they answer different questions. `GPUCurrentFanSpeed` is what the
    fan is doing; `GPUTargetFanSpeed` is what the curve is asking for. On this card at idle they
    disagree -- target 30 %, current 0 % at 41 C -- because a 3090 stops its fans below a
    threshold. So *current* alone cannot separate "the curve asks for nothing" from "the fans are
    dead", and it is *target*, read against the temperature already in `arm_pass_gpu`, that a
    changed curve would move.

    Deliberately not `check_output`. Asking for a fan index the card does not have makes
    nvidia-settings exit non-zero even though every attribute that does exist was printed on
    stdout; this card has two fans, so a four-fan query is a complete, successful call that
    returns 1. Trusting the return code here would throw all five good values away.
    """
    args = ["-q", f"[gpu:{sidx}]/GPUFanControlState"]
    for i in range(max_fans):
        args += ["-q", f"[fan:{i}]/GPUTargetFanSpeed",
                 "-q", f"[fan:{i}]/GPUCurrentFanSpeed"]
    env = dict(os.environ)
    env.setdefault("DISPLAY", ":0")
    try:
        r = subprocess.run(["nvidia-settings", *args], capture_output=True, text=True,
                           env=env, timeout=20)
    except Exception as e:                                        # noqa: BLE001
        return {"fan_error": repr(e)[:200], "fan_control": "unknown"}
    return parse_fan_query(r.stdout)


def read_state(index: int = 0) -> dict:
    out: dict = {}
    # No silent fallback to the raw index. A wrong read here feeds the `is_stock` gate, so
    # guessing could either refuse a clean card or admit an overclocked one. Unverifiable means
    # unknown, and unknown fails the gate.
    try:
        sidx = settings_index_for(index)
    except GpuStateError as e:
        out["mapping_error"] = str(e).splitlines()[0]
        out["mem_transfer_offset"] = None
        out["core_offset"] = None
        sidx = None
    for attr, key in (("GPUMemoryTransferRateOffset", "mem_transfer_offset"),
                      ("GPUGraphicsClockOffset", "core_offset")):
        if sidx is None:
            continue
        try:
            txt = _settings(["-q", f"[gpu:{sidx}]/{attr}[4]"])
            val = None
            for line in txt.splitlines():
                if "):" in line:
                    val = line.rsplit("):", 1)[1].strip().rstrip(".")
            out[key] = int(float(val)) if val is not None else None
        except Exception:
            out[key] = None
    try:
        csv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.limit,clocks.max.memory,clocks.max.graphics,"
             "clocks.current.graphics,fan.speed",
             "--format=csv,noheader,nounits", "-i", str(index)],
            text=True, stderr=subprocess.DEVNULL).strip()
        # fan.speed rides along on the query that was already being made, so the actual speed
        # costs nothing. The control mode needs nvidia-settings and is fetched separately below.
        pl, mem, gr, cur, fan = [float(x) for x in csv.split(",")]
        out.update(power_limit_w=pl, clocks_max_memory_mhz=mem, clocks_max_graphics_mhz=gr,
                   clocks_current_graphics_mhz=cur, fan_speed_pct=fan)
    except Exception as e:                                       # noqa: BLE001
        out["nvidia_smi_error"] = repr(e)
    # One batched nvidia-settings call, measured at 37 ms against 26 ms for a single -q, so
    # asking for nine attributes costs 11 ms more than asking for one. Over 21 arm-passes that is
    # under a second on a 109-minute run.
    if sidx is not None:
        out.update(fan_state(sidx))
    else:
        out["fan_control"] = "unknown"
    return out


def lock_held() -> bool:
    return LOCKFILE.exists()


def lock_owner_pid() -> int | None:
    """The pid recorded in the lockfile, if the lock exists and that process is alive."""
    if not LOCKFILE.exists():
        return None
    try:
        for line in LOCKFILE.read_text().splitlines():
            if line.startswith("pid="):
                pid = int(line[4:])
                os.kill(pid, 0)          # signal 0 = liveness probe only
                return pid
    except (ProcessLookupError, ValueError):
        return None
    except PermissionError:
        return None                      # exists but owned by another user; treat as not ours
    return None


def acquire_lock(owner: str, *, force: bool = False) -> None:
    """Take the run lock, refusing if a live run already holds it.

    The lock is deliberately global rather than per-device. Two benchmarks running at once in
    one chassis share a power supply, case airflow and PCIe root complex, so they contaminate
    each other's timing and energy figures even on different cards - and if either one varies
    clocks, it varies them for a card the other is measuring.

    The previous version simply overwrote the file, so two concurrent runs would each believe
    they held it. A stale lock (recorded pid no longer alive) is taken over and reported.
    """
    live = lock_owner_pid()
    if live is not None and live != os.getpid() and not force:
        raise GpuStateError(
            f"another benchmark run is already active (pid {live}) and holds {LOCKFILE.name}.\n"
            f"{LOCKFILE.read_text()}"
            "Concurrent runs in one chassis contaminate each other through the power supply, "
            "case airflow and PCIe, even on different GPUs. Wait for it to finish, or pass "
            "force=True if you have established that this is a stale lock.")
    if LOCKFILE.exists() and live is None:
        try:
            print(f"[gpustate] taking over a stale lock: {LOCKFILE.read_text().strip()!r}",
                  flush=True)
        except OSError:
            pass
    LOCKFILE.write_text(f"{owner}\npid={os.getpid()}\nsince={time.strftime('%FT%T%z')}\n")


def release_lock() -> None:
    LOCKFILE.unlink(missing_ok=True)


def apply(state, index: int = 0, *, force: bool = False,
          settle_s: float = 3.0) -> dict:
    """Apply a resource condition and verify it took effect. Refuses while a run holds the lock."""
    if isinstance(state, _StockProxy):
        state = stock_for(index)
    if lock_held() and not force:
        raise GpuStateError(
            f"a benchmark run holds {LOCKFILE.name}; refusing to change GPU clocks while it is "
            f"measuring.\n{LOCKFILE.read_text()}")
    sidx = settings_index_for(index)   # raises instead of writing to an unverified card
    _settings(["-a", f"[gpu:{sidx}]/GPUMemoryTransferRateOffset[4]={state.mem_transfer_offset}"])
    _settings(["-a", f"[gpu:{sidx}]/GPUGraphicsClockOffset[4]={state.core_offset}"])
    subprocess.check_output(["sudo", "nvidia-smi", "-i", str(index),
                             "-pl", str(state.power_limit_w)],
                            text=True, stderr=subprocess.DEVNULL)
    if state.lock_sm_mhz:
        subprocess.check_output(["sudo", "nvidia-smi", "-i", str(index),
                                 "-lgc", f"{state.lock_sm_mhz},{state.lock_sm_mhz}"],
                                text=True, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(["sudo", "nvidia-smi", "-i", str(index), "-rgc"],
                       capture_output=True, text=True)
    time.sleep(settle_s)

    got = read_state(index)
    problems = []
    if got.get("mem_transfer_offset") != state.mem_transfer_offset:
        problems.append(f"mem offset requested {state.mem_transfer_offset}, "
                        f"read back {got.get('mem_transfer_offset')}")
    if got.get("core_offset") != state.core_offset:
        problems.append(f"core offset requested {state.core_offset}, "
                        f"read back {got.get('core_offset')}")
    if abs((got.get("power_limit_w") or -1) - state.power_limit_w) > 0.5:
        problems.append(f"power limit requested {state.power_limit_w} W, "
                        f"read back {got.get('power_limit_w')} W")
    if state.lock_sm_mhz:
        cur = got.get("clocks_current_graphics_mhz")
        if cur is None or abs(cur - state.lock_sm_mhz) > 30:
            problems.append(f"SM clock locked to {state.lock_sm_mhz} MHz, "
                            f"reads {cur} MHz")
    if problems:
        raise GpuStateError(f"GPU state '{state.name}' did not take effect: " + "; ".join(problems))
    got["condition"] = state.name
    return got


# Deliberately NOT a module constant any more. "Stock" is a property of a specific card: this
# RTX 3090 defaults to 420 W, an RTX A6000 to 300 W. A hard-coded constant would ask the wrong
# card for the wrong limit and would make "restore stock" restore something that was never stock
# for that device.
def stock_for(index: int = 0) -> GpuState:
    """This device's own stock condition, read from its reported default power limit."""
    import devices as _dev
    d = _dev.get_device(index)
    return GpuState(f"stock@{d.short}", mem_transfer_offset=0, core_offset=0,
                    power_limit_w=int(round(d.power_default_w)), lock_sm_mhz=None)


def restore_stock(index: int = 0, *, force: bool = False) -> dict:
    return apply(stock_for(index), index, force=force)


class _StockProxy:
    """Backwards-compatible `gpustate.STOCK` that resolves per device at use time.

    Kept so existing call sites keep working, but it resolves against device 0 only. New code
    should call `stock_for(index)` explicitly.
    """
    def _resolve(self) -> GpuState:
        return stock_for(0)

    def __getattr__(self, item):
        return getattr(self._resolve(), item)

    def __repr__(self) -> str:
        return repr(self._resolve())


STOCK = _StockProxy()


def install_restore_guard(index: int = 0) -> None:
    """Restore stock clocks on normal exit and on SIGINT/SIGTERM.

    Without this, killing a resource-varying run leaves the card overclocked, and the next study
    on this machine silently inherits it -- which is exactly how this repo's own first Phase A
    run ended up being discarded.
    """
    import atexit
    import signal

    try:
        _stock_snapshot = stock_for(index)       # resolve now, while the tools are known good
    except Exception:
        _stock_snapshot = None

    def _restore(*_a):
        try:
            release_lock()
            target = _stock_snapshot if _stock_snapshot is not None else stock_for(index)
            apply(target, index, force=True)
            print("[gpustate] restored stock clocks", flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"[gpustate] FAILED to restore stock: {e!r}", flush=True)

    atexit.register(_restore)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            prev = signal.getsignal(sig)

            def _handler(s, f, _prev=prev):
                _restore()
                if callable(_prev):
                    _prev(s, f)
                raise SystemExit(128 + s)

            signal.signal(sig, _handler)
        except Exception:
            pass
