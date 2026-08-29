"""GPU telemetry and process-safety guards.

The port guard exists because of a documented failure in prior community work on this exact
model: a llama-server that was killed but not reaped kept answering /health, so the next arm in
a sweep failed to bind the port, died immediately, and its health check was answered by the OLD
server. Three published rows were produced that way before anyone noticed. The tell was a VRAM
figure that did not change between arms that should have differed.

We therefore refuse to start unless the port is free, and after starting we assert that the
process listening on the port is the child we just spawned.
"""
from __future__ import annotations

import contextlib
import os
import socket
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- port safety

def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def pid_owning_port(port: int) -> int | None:
    """PID listening on `port`, via `ss`. None if nothing is listening."""
    try:
        out = subprocess.check_output(
            ["ss", "-ltnpH", f"sport = :{port}"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    for line in out.splitlines():
        if "pid=" not in line:
            continue
        frag = line.split("pid=", 1)[1]
        digits = ""
        for ch in frag:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            return int(digits)
    return None


def assert_port_owned_by(port: int, pid: int) -> None:
    """Raise unless `pid` (or a descendant of it) owns `port`."""
    # Every process reaches pid 1, so the ancestor walk below would accept anything against it
    # and the assertion would hold vacuously. A real server pid is never 1; refuse rather than
    # pass.
    if pid <= 1:
        raise RuntimeError(f"refusing to check port {port} against pid {pid}: not a server pid")
    owner = pid_owning_port(port)
    if owner is None:
        raise RuntimeError(f"port {port}: nothing listening, but server was expected")
    if owner == pid:
        return
    # llama-server may fork; walk up the parent chain from the observed owner.
    cur = owner
    for _ in range(8):
        try:
            ppid = int(subprocess.check_output(
                ["ps", "-o", "ppid=", "-p", str(cur)], text=True).strip())
        except Exception:
            break
        if ppid == pid:
            return
        if ppid <= 1:
            break
        cur = ppid
    raise RuntimeError(
        f"port {port} is owned by pid {owner}, which is not our server (pid {pid}). "
        "A stale llama-server is almost certainly still answering /health - refusing to "
        "measure against it.")


# --------------------------------------------------------------------------- GPU sampling

_NVSMI_FIELDS = (
    # power.draw is a time-averaged field on Ampere - querying it beside power.draw.average
    # returns the same number on every sample - so integrating it smears a request's power over
    # the second around it. power.draw.instant is a separate, less-smoothed reading: over 20
    # samples under load the two were never equal and instant had 58 % more spread. Both are
    # sampled and both are integrated, so a file carries the averaged figure the earlier phases
    # used as well as the sharper one, and neither becomes incomparable.
    "power.draw", "power.draw.instant", "temperature.gpu",
    "clocks.current.graphics", "clocks.current.memory",
    "memory.used", "utilization.gpu",
)


def gpu_snapshot(index: int = 0) -> dict[str, float]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={','.join(_NVSMI_FIELDS)}",
             "--format=csv,noheader,nounits", "-i", str(index)],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return {}
    parts = [p.strip() for p in out.split(",")]
    snap: dict[str, float] = {}
    for name, raw in zip(_NVSMI_FIELDS, parts):
        try:
            snap[name] = float(raw)
        except ValueError:
            pass
    return snap


class NvmlEnergy:
    """The driver's own cumulative energy counter, read through NVML.

    `PowerSampler` integrates `power.draw` sampled by an nvidia-smi subprocess every 100 ms. That
    is one instrument, and until 2026-08-29 it was the only one, while the README asserted a
    -37.1 % decode-energy figure and separately argued that NVIDIA's cumulative counter "would not
    settle the magnitude either" -- on the strength of one developer-forum report about a different
    card. That is dismissing an instrument in advance instead of reading it.

    Probed on this RTX 3090 it returns NVML_SUCCESS. `nvmlDeviceGetTotalEnergyConsumption` is
    defined as millijoules accumulated since the last driver reload, so a delta across a request is
    that request's energy as the driver counts it. It is device-wide, which is exactly why the run
    holds a lock and refuses to start beside another tenant.

    Why this is worth having beside the integral rather than instead of it: the two disagree in
    ways that are diagnostic. A common multiplicative calibration error cancels in a same-board
    ratio and neither instrument would show it. What does NOT cancel is gain that varies with load,
    and the arms in this study do not draw the same power -- so two independent readings of the same
    requests are the only thing here that can see it.

    ctypes rather than pynvml: no new dependency, and the harness's own README says matplotlib and
    PyYAML are the only third-party packages. Fail-open throughout. An unsupported device, a driver
    without the symbol, or any error at all leaves `available` False and the run unaffected; a
    measurement must not stop because a second opinion was unavailable.
    """

    _LIB = None
    _ERR = None

    def __init__(self, index: int = 0):
        self.index = index
        self.available = False
        self.error = None
        self._handle = None
        self._init()

    def _init(self):
        import ctypes
        cls = type(self)
        try:
            if cls._LIB is None:
                cls._LIB = ctypes.CDLL("libnvidia-ml.so.1")
                if cls._LIB.nvmlInit_v2() != 0:
                    cls._LIB, cls._ERR = None, "nvmlInit_v2 failed"
            if cls._LIB is None:
                self.error = cls._ERR
                return
            h = ctypes.c_void_p()
            if cls._LIB.nvmlDeviceGetHandleByIndex_v2(self.index, ctypes.byref(h)) != 0:
                self.error = f"no NVML handle for device {self.index}"
                return
            self._handle = h
            # Probe rather than assume. GeForce support for this counter is not guaranteed, and
            # assuming either way is what the earlier text did.
            e = ctypes.c_ulonglong()
            rc = cls._LIB.nvmlDeviceGetTotalEnergyConsumption(h, ctypes.byref(e))
            if rc != 0:
                self.error = f"nvmlDeviceGetTotalEnergyConsumption rc={rc}"
                return
            self.available = True
        except Exception as exc:                                   # noqa: BLE001
            self.error = repr(exc)[:160]

    def read_mj(self):
        """Cumulative millijoules since driver reload, or None."""
        if not self.available:
            return None
        import ctypes
        e = ctypes.c_ulonglong()
        try:
            if type(self)._LIB.nvmlDeviceGetTotalEnergyConsumption(self._handle,
                                                                   ctypes.byref(e)) != 0:
                return None
        except Exception:                                          # noqa: BLE001
            return None
        return int(e.value)


@dataclass
class PowerSampler:
    """Background nvidia-smi poller. Integrates power over the sampled window -> joules.

    Energy is trapezoidal over the samples actually collected. If sampling produced fewer than
    two points the window is reported as `None` rather than guessed, so a too-short generation
    never silently becomes a fabricated tok/J number.
    """
    index: int = 0
    interval_s: float = 0.10

    _samples: list[tuple[float, float]] = field(default_factory=list)  # (t, watts), averaged
    _samples_instant: list[tuple[float, float]] = field(default_factory=list)  # (t, watts)
    _temps: list[float] = field(default_factory=list)
    _sm_clocks: list[float] = field(default_factory=list)
    _mem_clocks: list[float] = field(default_factory=list)
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)

    def __enter__(self) -> "PowerSampler":
        self._samples.clear(); self._samples_instant.clear()
        self._nvml_samples = []
        self._nvml_last_at = None
        self._temps.clear(); self._sm_clocks.clear()
        self._mem_clocks.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        nvml = NvmlEnergy(self.index)
        self._nvml_reader = nvml
        while not self._stop.is_set():
            snap = gpu_snapshot(self.index)
            now = time.perf_counter()
            if "power.draw" in snap:
                self._samples.append((now, snap["power.draw"]))
                # Read the counter at the SAME instant as the power sample it will be compared
                # against. The first version read it outside the `with` block, so its window was
                # the whole wall-clock span plus the thread join while the integral's window is
                # first-sample-to-last -- a mismatch the README already puts at about 4 % of a
                # request. NVML came out 1.08 to 2.59 % high on every record of a trial run, which
                # at 415 W is about 0.1 s, the poll period. That was the instrumentation, not the
                # instruments. Aligned here, a remaining difference is the thing being looked for.
                # EXACTLY TWICE per window: beside the first power sample and beside the last.
                # Never in between, and this is measured rather than assumed. Reading the counter
                # makes it lose energy, and the loss scales with how often it is read.
                #
                # The measurement is `harness/nvml_polling.py` and the numbers are in
                # `analysis/nvml_polling.txt`. They are NOT repeated here. This comment used to
                # carry its own table of watts, which is the defect Correction 43 section 3 raised
                # against the coverage figures -- quotable, not recheckable -- and it had drifted
                # against itself, giving the undisturbed window as 31.82 W in the table and 31.74 W
                # in the sentence below it. A number with no generator does that.
                #
                # A first version sampled it beside every power sample, which is 10 Hz, and it read
                # about a third low. That is the counter being disturbed by the reading, not the
                # two instruments disagreeing, and it reproduces with a dose-response curve the
                # developer-forum report this repository had cited as a reason to dismiss the
                # counter unread -- "the gap widens the more often power is polled".
                if not self._nvml_samples:
                    mj = nvml.read_mj()
                    if mj is not None:
                        self._nvml_samples.append((now, mj))
            if "power.draw.instant" in snap:
                self._samples_instant.append((now, snap["power.draw.instant"]))
            if "temperature.gpu" in snap:
                self._temps.append(snap["temperature.gpu"])
            if "clocks.current.graphics" in snap:
                self._sm_clocks.append(snap["clocks.current.graphics"])
            if "clocks.current.memory" in snap:
                self._mem_clocks.append(snap["clocks.current.memory"])
            self._stop.wait(self.interval_s)

        # The closing pair, taken together once the loop has stopped. Both instruments' windows
        # have to END at the same instant as well as start there. An earlier version took this
        # read after `join()` and then timestamped it with the LAST power sample, which is a
        # value from one moment wearing another moment's label: the counter had gone on
        # accumulating through the remaining sleep and the join. At idle that hid inside noise;
        # at 415 W one poll interval is about 41 J against a 4000 J request, which is one per
        # cent -- the same size as the disagreement this is built to detect.
        now = time.perf_counter()
        mj = nvml.read_mj()
        snap = gpu_snapshot(self.index)
        if mj is not None and self._nvml_samples:
            self._nvml_samples.append((now, mj))
        if "power.draw" in snap:
            self._samples.append((now, snap["power.draw"]))
        if "power.draw.instant" in snap:
            self._samples_instant.append((now, snap["power.draw.instant"]))

    # -- results -----------------------------------------------------------
    @property

    def n_samples(self) -> int:
        return len(self._samples)

    @staticmethod
    def _integrate(samples) -> float | None:
        if len(samples) < 2:
            return None
        total = 0.0
        for (t0, w0), (t1, w1) in zip(samples, samples[1:]):
            total += (w0 + w1) / 2.0 * (t1 - t0)
        return total

    def energy_j(self) -> float | None:
        return self._integrate(self._samples)

    def energy_j_instant(self) -> float | None:
        """The same trapezoid over power.draw.instant rather than the averaged field."""
        return self._integrate(self._samples_instant)

    def sample_span_s(self) -> float | None:
        """Seconds between the first and last sample, which is what energy_j integrates over.

        The loop queries nvidia-smi and then waits interval_s, so the period is the sum of the
        two and not interval_s alone. Without this the record carries only a sample count, and
        the fraction of the request the integral covers cannot be recovered from it.
        """
        if len(self._samples) < 2:
            return None
        return self._samples[-1][0] - self._samples[0][0]

    def summary(self) -> dict[str, float | int | None]:
        watts = [w for _, w in self._samples]
        ei = self.energy_j_instant()
        e = self.energy_j()
        # A second, independent reading of the same window. `energy_j` is an integral of a field
        # nvidia-smi reports as a one-second rolling average, sampled every 100 ms by a subprocess;
        # this is the driver's own accumulator. Both are on the same board at the same instant, so a
        # common multiplicative calibration error cancels in either and neither can see it -- what
        # a disagreement between them CAN show is error that varies with load, which is the one
        # kind that does not cancel in a ratio between two arms drawing different power.
        # First NVML reading to last, which are taken beside the first and last power samples, so
        # both instruments describe the identical window.
        ns = getattr(self, "_nvml_samples", [])
        nvml_j = ((ns[-1][1] - ns[0][1]) / 1000.0) if len(ns) >= 2 and ns[-1][1] >= ns[0][1] else None
        nerr = None if ns else "no NVML samples"
        return {
            "energy_j": e,
            "energy_j_nvml": nvml_j,
            "nvml_vs_integral_pct": (100.0 * (nvml_j - e) / e) if (e and nvml_j) else None,
            "nvml_error": nerr,
            "n_nvml_samples": len(ns),
            # The two spans must match. If they ever do not, the comparison above is measuring
            # window mismatch again and this is how a reader finds out.
            "nvml_span_s": (ns[-1][0] - ns[0][0]) if len(ns) >= 2 else None,
            "sample_span_s": self.sample_span_s(),
            "energy_j_instant": ei,
            "energy_instant_vs_average_pct": (100.0 * (ei - e) / e) if (e and ei) else None,
            "n_power_samples_instant": len(self._samples_instant),
            "power_mean_w": statistics.fmean(watts) if watts else None,
            "power_max_w": max(watts) if watts else None,
            "temp_max_c": max(self._temps) if self._temps else None,
            "temp_mean_c": statistics.fmean(self._temps) if self._temps else None,
            "sm_clock_mean_mhz": statistics.fmean(self._sm_clocks) if self._sm_clocks else None,
            "sm_clock_min_mhz": min(self._sm_clocks) if self._sm_clocks else None,
            # Memory clock is required, not optional: the resource-response phase estimates a
            # bandwidth elasticity, and that needs the clock the card ACHIEVED, not the one that
            # was requested. It also detects the design-breaking case where lowering the power
            # limit drags the memory P-state down with it, which would mean the "compute-only"
            # conditions were quietly varying bandwidth too.
            "mem_clock_mean_mhz": statistics.fmean(self._mem_clocks) if self._mem_clocks else None,
            "mem_clock_min_mhz": min(self._mem_clocks) if self._mem_clocks else None,
            "mem_clock_max_mhz": max(self._mem_clocks) if self._mem_clocks else None,
            "n_power_samples": len(self._samples),
        }


@contextlib.contextmanager
def sampling(index: int = 0, interval_s: float = 0.10):
    """Integrate power over a window, and read the driver's own energy counter across the same one.

    The NVML delta is taken here rather than at each call site so that both the prefill calibration
    and the measured request get it without either remembering to. It is a single library call at
    each end -- microseconds, no subprocess -- against the 100 ms nvidia-smi poll the integral
    already runs, so it cannot be what perturbs a measurement.
    """
    s = PowerSampler(index=index, interval_s=interval_s)
    with s:
        yield s


def gpu_compute_processes(index: int = 0) -> list[tuple[int, str, int]]:
    """(pid, name, used_MiB) for every process holding GPU memory."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits", "-i", str(index)],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return []
    procs = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                procs.append((int(parts[0]), parts[1], int(parts[2])))
            except ValueError:
                pass
    return procs


def assert_gpu_exclusive(index: int = 0, allow_pids: tuple[int, ...] = ()) -> None:
    """Raise if anything other than our own server holds GPU memory.

    Power is sampled for the whole device, so a second tenant makes every energy figure wrong,
    and a competing workload makes every timing figure wrong. Prior community work on this model
    documents a case where a desktop compositor silently halved decode throughput while the
    health endpoint stayed green -- the failure is invisible unless it is checked for.
    """
    intruders = [p for p in gpu_compute_processes(index) if p[0] not in allow_pids]
    if intruders:
        listing = ", ".join(f"pid {p} ({n}, {m} MiB)" for p, n, m in intruders)
        raise RuntimeError(
            f"GPU {index} is not exclusive to this benchmark: {listing}. "
            "Timing and energy would both be contaminated. Stop the other workload and retry.")


def _descendants_of(pid: int, ppid_of: dict[int, int]) -> set[int]:
    """Every pid whose parent chain reaches `pid`, plus `pid` itself."""
    kids: dict[int, list[int]] = {}
    for child, parent in ppid_of.items():
        kids.setdefault(parent, []).append(child)
    seen, stack = {pid}, [pid]
    while stack:
        for c in kids.get(stack.pop(), ()):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def host_load(own_names=("llama-server",), *, _ps_output=None, _self_pid=None):
    """What else is competing for the CPU at arm entry.

    The GPU is gated on temperature and clock; the host was not gated on anything. On 2026-08-26 a
    compiler ran on this machine during Phase M pass 2 and the only way to find out afterwards was
    to compare object-file timestamps against server-log timestamps by hand. Recording it per
    arm-pass turns that into a field.

    OWNERSHIP IS BY DESCENT, and names are only the fallback. Matching on names alone is wrong in
    both directions and this repository has now been bitten by each:

      * A false positive. The run polls nvidia-smi continuously to integrate power into joules,
        `nvidia-smi` is not in `own_names`, and a poll that happens to be alive when `ps` runs is
        recorded as 50 % of a core of outside competition. Phase B pass 2 has exactly that
        incident.
      * A false negative, which is worse. `python3` WAS in `own_names`, so any python on this
        host -- an analysis script, a plotting run, anything started in another terminal --
        competed invisibly. "The incident log is clean" was therefore only ever evidence about
        processes that are not python. python3, python and bench.py have been removed from the
        list for exactly that reason: descent already covers the run's own interpreter, and
        keeping the names would have kept the hole open on everyone else's.

    Descent settles both: the run's own poller is a child of this process and is not counted;
    someone else's python is not, and is. `own_names` survives as a fallback for a process that
    has been reparented away (a session leader whose parent has exited), which is why
    llama-server stays on the list.
    """
    out = {"loadavg_1m": None, "competing_pct": 0.0, "competing": [], "contended": False,
           "attribution": "descent"}
    try:
        out["loadavg_1m"] = os.getloadavg()[0]
    except OSError:
        pass
    # `_ps_output` and `_self_pid` are a test seam and nothing else. Before descent attribution
    # a test could exercise the 25 % threshold by passing own_names=() so the harness's own
    # python counted against it; descent makes that impossible by construction, because the
    # harness is always its own descendant. Without a seam the threshold and the descent rule
    # could only be checked by burning CPU on the machine running the suite, which is the one
    # thing this module exists to keep off a measuring host.
    if _ps_output is None:
        try:
            ps = subprocess.run(["ps", "-eo", "pid,ppid,pcpu,comm", "--no-headers"],
                                capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            out["note"] = "ps unavailable; contention not checked"
            return out
        ps_text = ps.stdout
    else:
        ps_text = _ps_output

    rows = []
    ppid_of: dict[int, int] = {}
    for line in ps_text.splitlines():
        parts = line.split(None, 3)
        if len(parts) != 4:
            continue
        try:
            pid, ppid, pct = int(parts[0]), int(parts[1]), float(parts[2])
        except ValueError:
            continue
        ppid_of[pid] = ppid
        rows.append((pid, pct, parts[3].strip()))

    mine = _descendants_of(os.getpid() if _self_pid is None else _self_pid, ppid_of)
    for pid, pct, name in rows:
        if pct < 5.0 or pid in mine or any(o in name for o in own_names):
            continue
        out["competing_pct"] += pct
        out["competing"].append({"comm": name, "pcpu": pct, "pid": pid})
    out["competing"].sort(key=lambda c: -c["pcpu"])
    out["competing"] = out["competing"][:5]
    # a quarter of one core. a build shows up as several hundred percent, so this is nowhere near
    # the noise and does not need tuning.
    out["contended"] = out["competing_pct"] >= 25.0
    return out


def settle_gpu(
    index: int = 0,
    *,
    target_temp_c: float | None = 60.0,
    idle_floor_c: float | None = None,
    margin_c: float = 8.0,
    max_wait_s: float = 240.0,
    poll_s: float = 5.0,
) -> dict:
    """Wait until the GPU has cooled to `target_temp_c` before an arm starts.

    Measured on this host: across one pass the card climbs 62 C -> 84 C while sitting on its
    450 W power cap, and the SM clock falls from ~1950 MHz to ~1769 MHz -- a 9.3 % spread. That
    is larger than several of the effects this study is trying to resolve, and because arms run
    at different positions within a pass, it lands directly inside every paired comparison.
    Rotation spreads the position effect but does not remove it.

    Gating on entry temperature makes every arm start from the same thermal state. If the target
    cannot be reached within `max_wait_s` the actual state is returned anyway and recorded, so a
    run on a hot day is degraded and disclosed rather than silently biased.
    """
    # A fixed absolute target is device-specific: 60 C is a meaningful, reachable entry
    # condition for this open-air RTX 3090, but a blower-cooled workstation card idles and cools
    # differently, so the same number could be unreachable (a timeout on every arm) or trivially
    # met (a no-op). When an idle floor is supplied, the gate targets floor + margin instead,
    # which means the same thing on any card.
    if target_temp_c is None:
        if idle_floor_c is None:
            raise ValueError("settle_gpu needs either target_temp_c or idle_floor_c")
        target_temp_c = idle_floor_c + margin_c

    t0 = time.perf_counter()
    start = gpu_snapshot(index)
    while time.perf_counter() - t0 < max_wait_s:
        snap = gpu_snapshot(index)
        temp = snap.get("temperature.gpu")
        if temp is None or temp <= target_temp_c:
            return {"waited_s": round(time.perf_counter() - t0, 1),
                    "entry_temp_c": temp, "target_c": target_temp_c,
                    "idle_floor_c": idle_floor_c, "margin_c": margin_c,
                    "reached_target": True,
                    "start_temp_c": start.get("temperature.gpu"),
                    "entry_sm_clock_mhz": snap.get("clocks.current.graphics")}
        time.sleep(poll_s)
    snap = gpu_snapshot(index)
    return {"waited_s": round(time.perf_counter() - t0, 1),
            "entry_temp_c": snap.get("temperature.gpu"), "target_c": target_temp_c,
            "idle_floor_c": idle_floor_c, "margin_c": margin_c,
            "reached_target": False,
            "start_temp_c": start.get("temperature.gpu"),
            "entry_sm_clock_mhz": snap.get("clocks.current.graphics")}


def overclock_state(index: int = 0) -> dict:
    """Capture clock offsets and power limits so an overclock can never be silently in effect.

    This exists because it happened. Ten minutes into the first full run of this study the card
    was found carrying `GPUMemoryTransferRateOffset=800` (+400 MHz memory) and
    `GPUGraphicsClockOffset=100`, with a 450 W limit against a 420 W default -- while the
    README described it as stock. Memory-bandwidth overclock is not a neutral variable here:
    batch-1 decode is bandwidth-bound and speculative verification is comparatively
    compute-dense, so an undisclosed memory overclock moves the two arms by different amounts.
    That run was discarded. This function makes the state part of every result file.
    """
    state: dict = {}
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=power.limit,power.default_limit,power.min_limit,power.max_limit,"
             "clocks.max.graphics,clocks.max.memory,clocks.max.sm",
             "--format=csv,noheader,nounits", "-i", str(index)],
            text=True, stderr=subprocess.DEVNULL).strip()
        keys = ("power_limit_w", "power_default_limit_w", "power_min_limit_w",
                "power_max_limit_w", "clocks_max_graphics_mhz", "clocks_max_memory_mhz",
                "clocks_max_sm_mhz")
        for k, v in zip(keys, [x.strip() for x in out.split(",")]):
            try:
                state[k] = float(v)
            except ValueError:
                state[k] = v
    except Exception as e:                                    # noqa: BLE001
        state["nvidia_smi_error"] = repr(e)

    # This used the raw nvidia-smi index against nvidia-settings, which maintains an
    # independent enumeration. Harmless on one GPU and wrong on two. Route it through the
    # UUID-verified mapping, and treat an unverifiable mapping as unknown rather than guessing.
    env = dict(os.environ)
    env.setdefault("DISPLAY", ":0")
    import gpustate as _gs
    try:
        sidx = _gs.settings_index_for(index)
    except Exception as e:                                        # noqa: BLE001
        state["settings_mapping_error"] = repr(e)[:200]
        sidx = None
    for attr, key in (("GPUMemoryTransferRateOffset", "mem_transfer_rate_offset"),
                      ("GPUGraphicsClockOffset", "graphics_clock_offset")):
        if sidx is None:
            state[key] = "unverifiable"
            continue
        try:
            out = subprocess.check_output(
                ["nvidia-settings", "-q", f"[gpu:{sidx}]/{attr}[4]"],
                text=True, stderr=subprocess.DEVNULL, env=env)
            val = None
            for line in out.splitlines():
                if "):" in line:
                    val = line.rsplit("):", 1)[1].strip().rstrip(".")
            state[key] = float(val) if val is not None else None
        except Exception:
            state[key] = "unavailable"

    # Fans, from 2026-08-29. The gate covered clock offsets and the power limit and said nothing
    # about cooling, so holding the card cooler -- which raises the sustained clock, and Phase B
    # shows every arm shedding 8-12 % of its clock inside one arm-pass -- was the one lever that
    # could be pulled without any field recording it.
    if sidx is not None:
        try:
            state.update(_gs.fan_state(sidx))
        except Exception as e:                                    # noqa: BLE001
            state["fan_error"] = repr(e)[:200]
            state["fan_control"] = "unknown"
    else:
        state["fan_control"] = "unknown"

    state["is_stock"] = is_stock(state)
    return state


def is_stock(state: dict) -> bool:
    """Whether a card is in the condition this study calls stock.

    A separate function from `overclock_state` because the rule is the interesting part and it
    could not be tested while it was four lines inside a function that shells out to two binaries
    and an X server. Feed it a dict.

    UNKNOWN FAILS, in every clause. `len(numeric) == 2` is what enforces that for the offsets: a
    card whose offsets could not be read is not thereby stock, it is unverified, and unverified
    has to stop a run for the same reason a +400 MHz memory overclock does -- the first Phase A
    run was discarded over exactly that, found ten minutes in while the write-up said "stock".
    `fan_control != "auto"` extends the same treatment to cooling, and covers "manual" and
    "unknown" with one comparison.
    """
    offsets = [state.get("mem_transfer_rate_offset"), state.get("graphics_clock_offset")]
    numeric = [o for o in offsets if isinstance(o, (int, float))]
    if len(numeric) != 2 or not all(o == 0 for o in numeric):
        return False
    pl, dpl = state.get("power_limit_w"), state.get("power_default_limit_w")
    if not (isinstance(pl, float) and isinstance(dpl, float) and pl == dpl):
        return False
    if state.get("fan_control") != "auto":
        return False
    return True
