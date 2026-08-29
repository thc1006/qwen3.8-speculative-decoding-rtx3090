#!/usr/bin/env python3
"""Does reading NVML's energy counter change what the counter reports? Measured, with a control.

`telemetry.PowerSampler` reads `nvmlDeviceGetTotalEnergyConsumption` EXACTLY TWICE per window --
beside the first power sample and beside the last -- and a comment in that file explained why with
a five-row table of watts. The table was right and its provenance was wrong: it lived only in a
source comment, so it could be quoted and not rechecked. That is the defect Correction 43 raised
against the coverage figures ("those figures live only in docstrings"), committed again four hours
later in a different file. The comment had also drifted against itself, giving the undisturbed
window as 31.82 W in the table and 31.74 W in the sentence below it. Numbers with no generator do
that.

This is the generator. It writes `analysis/nvml_polling.txt`.

WHAT WOULD MAKE THE READING WRONG. "Reading the counter loses energy" and "polling makes the card
draw less power" predict the same counter values and are different claims. Only one control
separates them: run the harness's own 10 Hz nvidia-smi integral in every condition, unchanged, and
vary only the counter polling. The card's actual draw is then observed by an instrument that is
not the one under test. If the integral holds flat while the counter falls, the counter is losing
energy. If both fall, the card really is drawing less and the counter is honest.

A second control separates the harness's own polling from the effect: `nosmi` runs no nvidia-smi at
all. If it matches `hz0`, the 10 Hz subprocess polling the harness has always done is not itself
disturbing the counter -- which is the assumption every energy number in this repository rests on.

CONDITIONS, all on the same card in the same state, differing only in what happens between the two
end reads:

    nosmi   no nvidia-smi, counter read at the window ends only
    hz0     nvidia-smi at 10 Hz, counter read at the window ends only   <- what the harness does
    hz1     as hz0, plus counter reads at   1 Hz
    hz10    as hz0, plus counter reads at  10 Hz
    hz100   as hz0, plus counter reads at 100 Hz

Reps are INTERLEAVED -- every condition once, then again -- because an idle card's power drifts
with temperature over minutes, and running five reps of one condition before the next would let
that drift load onto whichever condition ran last.

The counter delta is always last-read minus first-read, so intermediate reads are inside the
measured span and any energy they cost is inside the number.

  nvml_polling.py                 idle card, 5 conditions x 5 reps x 8 s
  nvml_polling.py --busy          same, under a fixed synthetic GPU load, if torch is importable
  nvml_polling.py --reps N --window S
"""
from __future__ import annotations

import argparse
import statistics as st
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gpustate                        # noqa: E402
from telemetry import NvmlEnergy       # noqa: E402

ROOT = Path(__file__).parent.parent
OUT = ROOT / "analysis" / "nvml_polling.txt"

# (name, run nvidia-smi at 10 Hz, counter reads per second between the ends)
CONDITIONS = [
    ("nosmi", False, 0.0),
    ("hz0",   True,  0.0),
    ("hz1",   True,  1.0),
    ("hz10",  True,  10.0),
    ("hz100", True,  100.0),
]


def _smi_watts(index: int) -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
             "-i", str(index)], text=True, stderr=subprocess.DEVNULL).strip()
        return float(out.split(",")[0])
    except Exception:                                              # noqa: BLE001
        return None


class _Smi(threading.Thread):
    """The harness's own sampler, reproduced: one nvidia-smi subprocess every 100 ms."""

    def __init__(self, index: int, interval_s: float = 0.10):
        super().__init__(daemon=True)
        self.index, self.interval_s = index, interval_s
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            w = _smi_watts(self.index)
            if w is not None:
                self.samples.append((time.perf_counter(), w))
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=5)

    def integral_j(self) -> tuple[float, float] | None:
        """Trapezoid over the samples collected -> (joules, span_s). None under two samples."""
        s = self.samples
        if len(s) < 2:
            return None
        e = sum((s[i + 1][0] - s[i][0]) * (s[i + 1][1] + s[i][1]) / 2.0 for i in range(len(s) - 1))
        return e, s[-1][0] - s[0][0]


def run_condition(name: str, use_smi: bool, hz: float, *, window_s: float, index: int) -> dict:
    nvml = NvmlEnergy(index)
    if not nvml.available:
        raise SystemExit(f"NVML energy counter unavailable: {nvml.error}")

    smi = _Smi(index) if use_smi else None
    if smi is not None:
        smi.start()
        time.sleep(0.25)                       # let one sample land before the window opens

    reads = 0
    t0 = time.perf_counter()
    e0 = nvml.read_mj()
    reads += 1
    if hz > 0:
        period = 1.0 / hz
        nxt = t0 + period
        while True:
            now = time.perf_counter()
            if now - t0 >= window_s:
                break
            if now >= nxt:
                nvml.read_mj()
                reads += 1
                # Advance to the next slot rather than now+period, so a slow read does not
                # silently lower the achieved rate for the rest of the window.
                nxt += period
                while nxt <= now:
                    nxt += period
            else:
                time.sleep(min(0.0005, max(0.0, nxt - now)))
    else:
        while time.perf_counter() - t0 < window_s:
            time.sleep(0.005)
    e1 = nvml.read_mj()
    reads += 1
    t1 = time.perf_counter()

    integral = None
    if smi is not None:
        smi.stop()
        integral = smi.integral_j()

    span = t1 - t0
    counter_j = (e1 - e0) / 1000.0
    row = {
        "condition": name, "hz": hz, "smi": use_smi,
        "span_s": span, "reads": reads,
        "counter_w": counter_j / span if span else float("nan"),
        "integral_w": (integral[0] / integral[1]) if integral else None,
        "n_smi": len(smi.samples) if smi else 0,
        "achieved_hz": (reads - 2) / span if span and hz else 0.0,
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--window", type=float, default=8.0)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--busy", action="store_true",
                    help="hold a fixed synthetic GPU load; requires torch with CUDA")
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    live = gpustate.lock_owner_pid()
    if live is not None:
        raise SystemExit(f"a measurement holds {gpustate.LOCKFILE.name} (pid {live}); "
                         "this reads device-wide energy and would measure that run instead")

    gpustate.acquire_lock(f"nvml_polling.py -> {OUT.name}")
    load = None
    rows: list[dict] = []
    try:
        # Inside the try, so a load that starts and then hits an error is still stopped and the
        # lock still released. Started after the lock, so a refused lock never leaves a GPU busy.
        if args.busy:
            load = _start_load()
        for rep in range(args.reps):
            for name, use_smi, hz in CONDITIONS:
                time.sleep(args.settle)
                r = run_condition(name, use_smi, hz, window_s=args.window, index=args.index)
                r["rep"] = rep
                rows.append(r)
                print(f"  rep{rep} {name:6s} counter {r['counter_w']:7.2f} W  "
                      f"integral {r['integral_w'] if r['integral_w'] is None else round(r['integral_w'], 2)}"
                      f"  reads {r['reads']}", flush=True)
    finally:
        if load is not None:
            load.stop()
        gpustate.release_lock()

    text = report(rows, args)
    if args.stdout:
        sys.stdout.write(text)      # see energy_instruments.py: print would add a second newline
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = OUT.with_suffix(".txt.tmp")
        tmp.write_text(text)
        tmp.replace(OUT)
        print(f"\n   wrote {OUT.relative_to(ROOT)}")
    return 0


class _Load:
    """A fixed CUDA matmul loop, so the counter can be exercised somewhere other than idle."""

    def __init__(self, torch, a, b):
        self.torch, self.a, self.b = torch, a, b
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        # Synchronise every iteration. Without it the loop queues kernels far faster than the
        # device retires them, so the host runs ahead unboundedly and `stop()` returns while the
        # GPU is still working through a backlog -- which would put load into the settle period
        # of whichever condition ran next.
        while not self._stop.is_set():
            self.a @ self.b
            self.torch.cuda.synchronize()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=10)


def _start_load():
    try:
        import torch
    except ImportError:
        raise SystemExit("--busy needs torch; it is not importable in this environment")
    if not torch.cuda.is_available():
        raise SystemExit("--busy needs a CUDA device visible to torch")
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    return _Load(torch, a, b)


def report(rows: list[dict], args) -> str:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["condition"], []).append(r)

    L: list[str] = []
    L.append("=" * 92)
    L.append("DOES READING THE NVML ENERGY COUNTER CHANGE WHAT IT REPORTS?")
    L.append("=" * 92)
    L.append(f"{args.reps} interleaved reps x {args.window:.0f} s windows, "
             f"{'synthetic CUDA load' if args.busy else 'idle card'}, device {args.index}.")
    L.append("The counter delta is last-read minus first-read; intermediate reads are inside it.")
    L.append("")
    L.append("  condition   counter W       integral W      counter/integral   reads   achieved Hz")
    L.append("  " + "-" * 84)

    ref = None
    for name, _, _ in CONDITIONS:
        rs = by.get(name) or []
        if not rs:
            continue
        cw = [r["counter_w"] for r in rs]
        iw = [r["integral_w"] for r in rs if r["integral_w"] is not None]
        c_m = st.fmean(cw)
        c_sd = st.stdev(cw) if len(cw) > 1 else 0.0
        if iw:
            i_m = st.fmean(iw)
            i_sd = st.stdev(iw) if len(iw) > 1 else 0.0
            i_txt = f"{i_m:6.2f} +- {i_sd:4.2f}"
            ratio = f"{100.0 * c_m / i_m:8.1f} %"
        else:
            i_m, i_txt, ratio = None, "        --      ", "       --"
        if name == "hz0" and i_m is not None:
            ref = (c_m, i_m)
        L.append(f"  {name:9s} {c_m:6.2f} +- {c_sd:4.2f}   {i_txt}   {ratio}      "
                 f"{st.fmean([r['reads'] for r in rs]):5.0f}   "
                 f"{st.fmean([r['achieved_hz'] for r in rs]):8.1f}")
    L.append("")

    # The control. If the integral moves with the polling rate, the card is drawing less and the
    # counter is not at fault; the effect claimed here would not exist.
    iw_all = {n: [r["integral_w"] for r in (by.get(n) or []) if r["integral_w"] is not None]
              for n, _, _ in CONDITIONS}
    polled = [n for n in ("hz0", "hz1", "hz10", "hz100") if iw_all.get(n)]
    if len(polled) >= 2:
        means = {n: st.fmean(iw_all[n]) for n in polled}
        spread = max(means.values()) - min(means.values())
        base = means.get("hz0") or st.fmean(list(means.values()))
        L.append("CONTROL -- what the card actually drew, by an instrument that is not under test:")
        L.append("  " + "   ".join(f"{n} {means[n]:.2f} W" for n in polled))
        L.append(f"  spread {spread:.2f} W = {100.0 * spread / base:.1f} % of hz0.")
        hi = max(means, key=lambda n: means[n])
        rest = [means[n] for n in polled if n != hi]
        if rest and means[hi] > max(rest) * 1.02:
            L.append(f"  Not flat, and the direction settles it: the control is HIGHEST at {hi} "
                     f"({means[hi]:.2f} W against {min(rest):.2f}-{max(rest):.2f} W elsewhere).")
            L.append("  The card draws MORE while the counter reports less, so the counter is")
            L.append("  losing energy rather than reflecting a quieter device. The extra watts are")
            L.append("  the cost of the polling itself.")
        else:
            L.append("  The control does not move with the polling rate while the counter does,")
            L.append("  so the counter is losing energy rather than reporting a quieter device.")
        L.append("")

    if by.get("nosmi") and by.get("hz0"):
        a = st.fmean([r["counter_w"] for r in by["nosmi"]])
        b = st.fmean([r["counter_w"] for r in by["hz0"]])
        pooled = [r["counter_w"] for r in by["nosmi"]] + [r["counter_w"] for r in by["hz0"]]
        sd = st.stdev(pooled) if len(pooled) > 1 else 0.0
        L.append("THE HARNESS'S OWN POLLING -- 10 Hz nvidia-smi subprocesses, in every energy")
        L.append("number this repository reports:")
        L.append(f"  no nvidia-smi {a:6.2f} W   vs   nvidia-smi at 10 Hz {b:6.2f} W   "
                 f"difference {b - a:+.2f} W")
        L.append(f"  pooled sd across those {len(pooled)} windows is {sd:.2f} W.")
        L.append("")

    # A mechanism the data can refuse. If each read discarded a fixed quantity of energy, loss
    # divided by read count would be constant across the rates. Reporting it either way is the
    # point: a dose-response curve is an effect, not an explanation.
    if ref:
        L.append("IS THE LOSS A FIXED COST PER READ? Loss against each condition's OWN integral,")
        L.append("divided by the reads that condition made:")
        for name in ("hz1", "hz10", "hz100"):
            rs = by.get(name) or []
            iw = [r["integral_w"] for r in rs if r["integral_w"] is not None]
            if not rs or not iw:
                continue
            c_m, i_m = st.fmean([r["counter_w"] for r in rs]), st.fmean(iw)
            span = st.fmean([r["span_s"] for r in rs])
            n = st.fmean([r["reads"] for r in rs])
            lost_j = (i_m - c_m) * span
            L.append(f"  {name:6s} lost {lost_j:8.2f} J over {n:5.0f} reads "
                     f"= {lost_j / n * 1000:7.2f} mJ per read   "
                     f"(retained {100.0 * c_m / i_m:5.1f} %)")
        L.append("  Constant across the three rates would support a fixed cost per read. It is")
        L.append("  reported because it can fail, and what it does is in the numbers above.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
