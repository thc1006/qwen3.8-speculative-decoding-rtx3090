"""Find out what vLLM actually publishes, before a run loop is written against guesses.

Phase V needs three facts that no amount of reading settles, because they have moved between
vLLM versions and the documentation does not list them:

  1. What the speculative counters are CALLED on this build. `vllm_server.spec_delta` finds them
     by pattern for exactly this reason, but a pattern that matches a histogram bucket instead of
     a counter reads as a plausible number rather than as an error. This prints every candidate
     and shows which one the picker chose.

  2. Whether decode rate is separable from prefill. llama.cpp's server returns
     `timings.predicted_per_second`, which is decode alone. vLLM's OpenAI endpoint returns no
     per-token timing at all, so a rate computed as completion_tokens/wall_ms carries the
     prefill inside it. That does not cancel when each engine is divided by its own baseline: a
     speculative arm decodes faster, so prefill is a larger share of its wall time, and the
     speedup comes out too small. If `vllm:time_per_output_token_seconds` exists on this build,
     the run loop should use it; this says whether it does.

  3. Whether the model, the MTP head and `--speculative-config` work together at all on this
     card, before 18 GiB of weights and an afternoon go into a run loop.

Nothing here measures anything. It starts a server, sends two requests, prints what moved, and
stops. Run it once after installing vLLM and read the output; the run loop gets written against
that, not against this file's expectations.

    python3 harness/vllm_probe.py --binary .venv-vllm/bin/vllm \\
        --model RedHatAI/Qwen3.8-27B-INT4 --spec '{"method":"mtp","num_speculative_tokens":1}'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vllm_server as V  # noqa: E402


def dump_metric_names(port: int) -> dict[str, list[str]]:
    """Every metric family the server publishes, grouped by what it might be useful for."""
    body = V._get(port, "/metrics")
    names = set()
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        names.add(line.partition(" ")[0].split("{", 1)[0])
    groups = {"spec": [], "timing": [], "tokens": [], "other": []}
    for n in sorted(names):
        low = n.lower()
        if "spec_decode" in low:
            groups["spec"].append(n)
        elif any(w in low for w in ("time_to_first", "time_per_output", "latency", "seconds")):
            groups["timing"].append(n)
        elif "token" in low:
            groups["tokens"].append(n)
        else:
            groups["other"].append(n)
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=".venv-vllm/bin/vllm")
    ap.add_argument("--model", default="RedHatAI/Qwen3.8-27B-INT4")
    ap.add_argument("--port", type=int, default=18190)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--spec", default="", help="JSON for --speculative-config; empty = baseline")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--log", default="logs/vllm_probe.log")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from matrices import phase_v  # noqa: E402  -- for the exact flags the phase will use

    extra = list(phase_v.COMMON_ARGS)
    if args.spec:
        extra += ["--speculative-config", args.spec]
    print(f"launching: {args.binary} serve {args.model} " + " ".join(extra))
    print("(first start compiles CUDA graphs; minutes, not seconds)\n")

    proc = V.start(args.binary, args.model, args.port, extra, Path(args.log), gpu_index=args.gpu)
    try:
        groups = dump_metric_names(args.port)
        print("=" * 96)
        print("METRIC FAMILIES PUBLISHED")
        print("=" * 96)
        for g in ("spec", "timing", "tokens"):
            print(f"\n[{g}]  {len(groups[g])}")
            for n in groups[g]:
                print("   ", n)
        print(f"\n[other] {len(groups['other'])} families not printed")

        print("\n" + "=" * 96)
        print("FACT 2: is decode separable from prefill?")
        print("=" * 96)
        ttft = [n for n in groups["timing"] if "time_to_first" in n]
        tpot = [n for n in groups["timing"] if "time_per_output" in n]
        print(f"  time_to_first_token : {ttft or 'ABSENT'}")
        print(f"  time_per_output_token: {tpot or 'ABSENT'}")
        if tpot:
            print("  -> the run loop can report a decode rate that excludes prefill, matching")
            print("     what llama.cpp's timings.predicted_per_second means.")
        else:
            print("  -> NOT separable from /metrics. The run loop must either stream and time the")
            print("     first chunk itself, or report a wall-clock rate and state that it")
            print("     includes prefill -- which biases the speedup DOWNWARD for the faster arm.")

        # two requests: the first warms caches and is discarded, as every phase here does
        sys_prompt, user = "", "Write a short paragraph about tea."
        print("\n" + "=" * 96)
        print("SENDING 2 REQUESTS (first discarded as warmup)")
        print("=" * 96)
        V.chat(args.port, sys_prompt, user, max_tokens=args.max_tokens, model=args.model)
        before = V.spec_counters(args.port)
        r = V.chat(args.port, sys_prompt, user, max_tokens=args.max_tokens, model=args.model)
        after = V.spec_counters(args.port)
        print(f"  wall {r['wall_ms']:.0f} ms   completion_tokens {r['completion_tokens']}"
              f"   finish {r['finish_reason']!r}")
        print(f"  naive rate (includes prefill): "
              f"{1000.0*r['completion_tokens']/r['wall_ms']:.2f} tok/s")

        print("\n" + "=" * 96)
        print("FACT 1: what moved, and what the picker chose")
        print("=" * 96)
        delta = V.spec_delta(before, after)
        moved = {k: v for k, v in delta["counters"].items() if v}
        if not moved:
            print("  NOTHING MOVED. Either this is the baseline arm, or --speculative-config was")
            print("  accepted and ignored.")
        for k, v in sorted(moved.items()):
            print(f"    {v:+12.2f}  {k}")
        print(f"\n  picker chose:  drafted={delta.get('drafted')}  accepted={delta.get('accepted')}"
              f"  drafts={delta.get('drafts')}")
        print(f"                 accept_rate={delta.get('accept_rate')}  "
              f"mean_len={delta.get('mean_len')}")
        # The failure this is here to catch: a histogram bucket matching the same words as a
        # counter, so the picker returns a bucket count and it reads as a token count.
        suspicious = [k for k in (delta.get("counters") or {})
                      if any(s in k for s in ("_bucket", "_sum", "_count"))]
        if suspicious:
            print(f"\n  NOTE: {len(suspicious)} histogram series are in the same family and match")
            print("  the picker's words. Check that drafted/accepted above are counters and not")
            print("  bucket counts. Series seen:")
            for k in sorted(suspicious)[:8]:
                print("   ", k)
        if args.spec:
            try:
                V.assert_speculation_observed(delta, "probe")
                print("\n  assert_speculation_observed: PASSED")
            except V.VllmError as e:
                print(f"\n  assert_speculation_observed: FAILED -- {e}")
        return 0
    finally:
        V.stop(proc)
        print("\nserver stopped.")


if __name__ == "__main__":
    sys.exit(main())
