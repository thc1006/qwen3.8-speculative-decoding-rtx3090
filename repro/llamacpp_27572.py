"""CUDA reproduction attempt for llama.cpp #27572.

The report is on HIP/gfx1151 and traces draft acceptance collapsing to exactly 0.0 to an async
device-to-host copy of `t_h_nextn` racing a later graph that reuses the same extra buffer. It
needs concurrent requests whose decode batches span several ubatches, and it does not trigger at
`-np 1`, with short prompts, sequentially, or against a warm prefix cache. Whether the same race
exists on CUDA is the open question, and one 3090 answers it.

Two things here go beyond repeating the reported configuration.

The reported data has two points, single-ubatch prompts passing 12 of 12 and roughly 19k-token
prompts collapsing, so the threshold between them is unmeasured. This sweeps prompt length to
find the shortest one that fails, which is what a minimal reproducer needs.

Each concurrent request gets its own slice of the corpus. Sharing a prefix would let the prompt
cache serve most of the prefill, and the report says a warm cache is healthy, so identical
prompts would test the wrong thing.

Usage:  python3 repro/llamacpp_27572.py [--lengths 256,1024,4096,8192,16384] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))

import filler as FILLER  # noqa: E402
import gpustate as G  # noqa: E402
import server as S  # noqa: E402

REPO = HERE.parent
BINARY = REPO / "llamacpp-master/build/bin/llama-server"
MODEL = REPO / "models/target/Qwen3.8-27B-UD-Q4_K_XL.gguf"

# Matches the reported server flags except for context, which is sized for 24 GiB rather than
# their 1M, and the multimodal projector, which the text-only reproducer does not need.
COMMON = [
    "-ngl", "999", "--fit", "off",
    "-c", "81920", "-b", "1024", "-ub", "256",
    "-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0",
    "--no-webui", "--jinja", "--metrics",
    # their sampling, set server-side because harness.server.chat only pins the chain at
    # temperature 0 and this reproduction runs at their temperature 1.0
    "--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0",
]
SPEC = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]

# Their sampling, not this study's greedy default. top_p/top_k/min_p ride on the server flags
# above; only the temperature is per request.
TEMPERATURE = 1.0


def fire(port: int, prompts: list[str], max_tokens: int, concurrent: bool) -> list[dict]:
    """Send one request per prompt, together or one at a time."""
    out: list[dict] = [None] * len(prompts)  # type: ignore[list-item]

    def one(i: int) -> None:
        try:
            out[i] = S.chat(port, "You are concise.", prompts[i],
                            max_tokens=max_tokens, temperature=TEMPERATURE,
                            seed=1000 + i, think=False, cache_prompt=False)
        except Exception as e:  # noqa: BLE001
            out[i] = {"error": repr(e)[:200]}

    if concurrent:
        ts = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    else:
        for i in range(len(prompts)):
            one(i)
    return out


def run_case(n_slots: int, n_req: int, n_prompt_tokens: int, concurrent: bool,
             port: int, out_dir: Path) -> dict:
    tag = f"np{n_slots}_{n_prompt_tokens}tok_{'conc' if concurrent else 'seq'}"
    log_path = out_dir / f"{tag}.log"
    handle = S.start(BINARY, MODEL, [*SPEC, "-np", str(n_slots)],
                     port=port, log_path=log_path, common_args=COMMON, gpu_index=0)
    try:
        S.assert_drafter_loaded(handle, "draft-mtp")

        # A distinct slice per request, so no two share a prefix.
        prompts = []
        for i in range(n_req):
            text, got = FILLER.filler_of(port, n_prompt_tokens,
                                         offset_chars=i * 400_000)
            prompts.append(text + "\n\nSummarise the passage above in one sentence.")
            if abs(got - n_prompt_tokens) > 64:
                print(f"    filler for request {i} realised {got} of {n_prompt_tokens}")
        t0 = time.time()
        responses = fire(port, prompts, max_tokens=128, concurrent=concurrent)
        wall = time.time() - t0
        acc = S.parse_acceptance_from_log(handle)
    finally:
        S.stop(handle.proc, port=port)

    # The warmup-free path: every acceptance line here belongs to one of our requests.
    rates = [a["accept_rate"] for a in acc]
    zero = [r for r in rates if r == 0.0]
    empty = [r for r in responses if isinstance(r, dict) and not r.get("content")]
    return {
        "tag": tag, "n_slots": n_slots, "n_req": n_req,
        "prompt_tokens": n_prompt_tokens, "concurrent": concurrent,
        "wall_s": round(wall, 1),
        "acceptance": [round(r, 5) for r in rates],
        "acceptance_min": min(rates) if rates else None,
        "acceptance_mean": round(statistics.fmean(rates), 5) if rates else None,
        "exactly_zero": len(zero),
        "empty_completions": len(empty),
        "errors": [r.get("error") for r in responses if isinstance(r, dict) and r.get("error")],
    }


def probe_config(port: int, out_dir: Path) -> dict:
    """Capture what the server says about pipeline parallelism, verbosely, once.

    Kept out of the sweep on purpose. The sweep is testing a race, and verbose logging changes
    the timing it depends on, so the configuration question gets its own short server at `-lv 5`
    while the measured runs stay at the default.
    """
    log_path = out_dir / "config_probe.log"
    handle = S.start(BINARY, MODEL, [*SPEC, "-np", "4", "-lv", "5"],
                     port=port, log_path=log_path, common_args=COMMON, gpu_index=0)
    try:
        text = handle.log_text()
    finally:
        S.stop(handle.proc, port=port)
    want = ("pipeline parallel", "n_copies", "n_devices", "ggml_backend", "using device",
            "CUDA0", "offloaded")
    lines = [ln for ln in text.splitlines() if any(w.lower() in ln.lower() for w in want)]
    return {"probe_lines": lines[:60], "log": str(log_path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="256,1024,4096,8192,16384")
    ap.add_argument("--port", type=int, default=18300)
    ap.add_argument("--out", default="repro/results_27572.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and check the inputs without touching the GPU")
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]

    cases = [(4, 4, n, True) for n in lengths]
    cases.append((1, 4, lengths[-1], False))   # control: sequential, one slot
    cases.append((4, 1, lengths[-1], False))   # control: four slots, one request at a time

    print("plan")
    for slots, n_req, toks, conc in cases:
        print(f"  -np {slots}  {n_req} request(s) {'together' if conc else 'one at a time'}  "
              f"{toks} prompt tokens")
    print(f"\nmodel   {MODEL.name}  {'present' if MODEL.exists() else 'MISSING'}")
    print(f"binary  {'present' if BINARY.exists() else 'MISSING'}")
    print(f"flags   {' '.join(COMMON)} {' '.join(SPEC)}")
    if args.dry_run:
        print("\ndry run, nothing started")
        return 0

    out_dir = Path("repro/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    G.acquire_lock("repro-27572")
    results = []
    try:
        print("\n=== configuration probe (-lv 5, not part of the sweep) ===", flush=True)
        try:
            cfg = probe_config(args.port, out_dir)
            for ln in cfg["probe_lines"][:12]:
                print(f"    {ln[:120]}", flush=True)
            Path("repro/config_27572.json").write_text(json.dumps(cfg, indent=2))
        except Exception as e:  # noqa: BLE001
            print(f"    probe failed: {e!r}", flush=True)

        for slots, n_req, toks, conc in cases:
            print(f"\n=== -np {slots}, {n_req} req, {toks} tokens, "
                  f"{'concurrent' if conc else 'sequential'} ===", flush=True)
            # One case failing must not take the sweep with it: this runs unattended after a
            # multi-hour queue, and losing six good cases to one bad server start is not a
            # trade worth making.
            try:
                r = run_case(slots, n_req, toks, conc, args.port, out_dir)
            except Exception as e:  # noqa: BLE001
                r = {"tag": f"np{slots}_{toks}tok_{'conc' if conc else 'seq'}",
                     "n_slots": slots, "n_req": n_req, "prompt_tokens": toks,
                     "concurrent": conc, "case_failed": repr(e)[:300],
                     "acceptance": [], "exactly_zero": 0, "empty_completions": 0}
                print(f"    case failed: {r['case_failed']}", flush=True)
            else:
                print(f"    acceptance {r['acceptance']}  exactly zero: {r['exactly_zero']}  "
                      f"empty: {r['empty_completions']}", flush=True)
            results.append(r)
            Path(args.out).write_text(json.dumps(results, indent=2))
    finally:
        G.release_lock()

    print(f"\n{'case':34s} {'acceptance (per request)':32s} {'zero':>5s} {'empty':>6s}")
    for r in results:
        print(f"  {r['tag']:32s} {str(r['acceptance'])[:30]:32s} "
              f"{r['exactly_zero']:5d} {r['empty_completions']:6d}")
    broken = [r for r in results if r.get("case_failed")]
    if broken:
        print(f"\n  {len(broken)} case(s) did not run: "
              + ", ".join(r["tag"] for r in broken))
    failing = [r for r in results if r["exactly_zero"]]
    if failing:
        shortest = min(r["prompt_tokens"] for r in failing if r["concurrent"])
        print(f"\n  reproduces on CUDA. Shortest concurrent prompt that collapses: {shortest} tokens.")
    elif broken:
        print("\n  no collapse seen, but some cases did not run, so this is not yet a negative")
        print("  result. Rerun the failed cases before concluding anything.")
    else:
        print("\n  does not reproduce on CUDA at any tested length, which narrows #27572 to HIP")
        print("  or to something else in the reporter's configuration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
