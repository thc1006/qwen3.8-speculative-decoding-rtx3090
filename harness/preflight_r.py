"""Pre-flight for Phase R: prove every resource condition can actually be applied, and that the
model still runs and produces sane output under it, BEFORE committing hours to the matrix.

Checks per condition:
  1. the setting is accepted and reads back exactly (a clamped offset is a silent lie)
  2. the resulting max memory / graphics clock is what the arithmetic predicts
  3. llama-server loads and generates
  4. the generation is not degenerate (a starved card that produces garbage is not a data point)

Run with no benchmark holding the lock. Restores stock on exit, including on failure.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "matrices"))

import gpustate as G       # noqa: E402
import quality             # noqa: E402
import server as S         # noqa: E402
import telemetry as T      # noqa: E402
import importlib           # noqa: E402

# Which matrix to check is chosen at run time, so the same pre-flight serves
# phase_r (power-capped conditions) and phase_r2 (pinned-clock conditions).
_MATRIX = os.environ.get("PREFLIGHT_MATRIX", "phase_r")
M = importlib.import_module(_MATRIX)

PORT = int(os.environ.get("PREFLIGHT_PORT", "18190"))
STOCK_MEM_CLK = 9751.0


def check(cond: G.GpuState) -> dict:
    out = {"condition": cond.name}
    applied = G.apply(cond, force=True)
    out["applied"] = applied

    want_mem = STOCK_MEM_CLK + cond.mem_clock_delta_mhz
    got_mem = applied.get("clocks_max_memory_mhz")
    out["mem_clock_ok"] = (got_mem == want_mem)
    out["mem_clock_want"] = want_mem
    out["mem_clock_got"] = got_mem
    out["lock_sm_mhz"] = getattr(cond, "lock_sm_mhz", None)

    log = HERE.parent / f"logs/preflight_{cond.name}.log"
    t0 = time.perf_counter()
    h = S.start(M.BINARIES["master"], M.MODEL, [], port=PORT, log_path=log,
                common_args=M.COMMON_ARGS, gpu_index=0)
    try:
        # Sample the clocks DURING generation. This is the check that decides whether Phase R
        # is even valid: the design assumes a power-limit reduction constrains COMPUTE while
        # leaving memory bandwidth alone. If the card drags its memory P-state down under a low
        # power cap, the "compute-only" conditions are quietly varying bandwidth as well and the
        # elasticity decomposition collapses.
        with T.sampling(index=0, interval_s=0.10) as ps:
            r = S.chat(PORT, "You write production Python.",
                       "Write a Python function that merges two sorted lists, with type hints, a "
                       "docstring, and three doctest examples.",
                       max_tokens=200, temperature=0.0)
        power = ps.summary()
        text = (r.get("reasoning_content") or "") + (r.get("content") or "")
        deg = quality.assess_degeneracy(text)
        out.update(
            load_s=round(h.ready_s, 1),
            decode_tok_s=r.get("t_predicted_per_second"),
            predicted_n=r.get("t_predicted_n"),
            degenerate=deg.is_degenerate,
            degeneracy_reason=deg.reason,
            gpu=T.gpu_snapshot(),
            power=power,
            mem_clock_under_load_mhz=power.get("mem_clock_mean_mhz"),
            sm_clock_under_load_mhz=power.get("sm_clock_mean_mhz"),
            power_mean_w=power.get("power_mean_w"),
            total_s=round(time.perf_counter() - t0, 1),
        )
    finally:
        S.stop(h.proc, port=PORT)
    return out


def main() -> int:
    if G.lock_held():
        print("A benchmark run holds the GPU lock. Wait for it to finish.")
        print(G.LOCKFILE.read_text())
        return 2

    results, failures = [], []
    try:
        for cond in M.CONDITIONS:
            print(f"\n=== {cond.name}: mem{cond.mem_transfer_offset:+d} "
                  f"({cond.mem_clock_delta_mhz:+.0f} MHz), {cond.power_limit_w} W ===", flush=True)
            try:
                res = check(cond)
                results.append(res)
                flag = ""
                if not res["mem_clock_ok"]:
                    flag += f"  !! mem clock {res['mem_clock_got']} != expected {res['mem_clock_want']}"
                if res.get("degenerate"):
                    flag += f"  !! degenerate: {res['degeneracy_reason']}"
                print(f"  load {res['load_s']}s  decode {res['decode_tok_s']:.2f} tok/s  "
                      f"n={res['predicted_n']}  "
                      f"sm={res['sm_clock_under_load_mhz']:.0f}MHz  "
                      f"mem={res['mem_clock_under_load_mhz']:.0f}MHz  "
                      f"P={res['power_mean_w']:.0f}W{flag}", flush=True)
                if flag:
                    failures.append((cond.name, flag.strip()))
            except Exception as e:                                # noqa: BLE001
                print(f"  FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
                traceback.print_exc(limit=2)
                failures.append((cond.name, f"{type(e).__name__}: {str(e)[:200]}"))
    finally:
        print("\n=== restoring stock ===")
        print(G.apply(G.STOCK, force=True))

    print("\n=== summary ===")
    print(f"  {'cond':13s} {'pin':>5s} {'mem_max':>8s} {'mem_load':>9s} {'sm_load':>8s} "
          f"{'W':>6s} {'tok/s':>7s}  degen")
    for r in results:
        pin = r.get("lock_sm_mhz")
        sm = r.get("sm_clock_under_load_mhz") or 0
        held = "" if pin is None else ("  ok" if abs(sm - pin) <= 30 else f"  DRIFT {sm-pin:+.0f}")
        print(f"  {r['condition']:13s} {str(pin or '-'):>5s} {r['mem_clock_got']:8.0f} "
              f"{(r.get('mem_clock_under_load_mhz') or 0):9.0f} {sm:8.0f} "
              f"{(r.get('power_mean_w') or 0):6.0f} {(r.get('decode_tok_s') or 0):7.2f}  "
              f"{r.get('degenerate')}{held}")
        if pin is not None and abs(sm - pin) > 30:
            failures.append((r["condition"],
                             f"pinned to {pin} MHz but ran at {sm:.0f} MHz under load"))

    # ---- the check that validates or kills the Phase R design ----
    ref = next((r for r in results if r["condition"] == "stock"), None)
    if ref and ref.get("mem_clock_under_load_mhz"):
        base_mem = ref["mem_clock_under_load_mhz"]
        print(f"\n  memory clock under load, relative to stock ({base_mem:.0f} MHz):")
        for r in results:
            m = r.get("mem_clock_under_load_mhz")
            if not m:
                continue
            drift = (m - base_mem) / base_mem * 100
            note = ""
            pinned = r.get("lock_sm_mhz")
            # A condition is "compute only" if it was not asked to move the memory clock,
            # whether it varies compute by a power cap or by pinning the core. Either way its
            # memory clock must not move, or it is not isolating compute.
            compute_only = (abs(r["mem_clock_want"] - STOCK_MEM_CLK) < 1
                            and (pinned is not None or r["condition"].startswith("pw")))
            if compute_only and abs(drift) > 1.0:
                note = ("   <-- THIS COMPUTE CONDITION IS ALSO MOVING THE MEMORY CLOCK. "
                        "It does not isolate compute and the elasticity decomposition is "
                        "invalid for it.")
                failures.append((r["condition"], "compute condition drags the memory clock"))
            print(f"    {r['condition']:13s} {m:8.0f} MHz  ({drift:+.1f} %){note}")
    if failures:
        print("\nFAILURES -- do not launch Phase R until these are resolved or the condition "
              "is removed from the matrix:")
        for name, why in failures:
            print(f"  {name}: {why}")
        return 1
    print("\nall conditions usable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
