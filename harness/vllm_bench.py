"""Phase V run loop: the same question on vLLM that bench.py asks on llama.cpp.

`matrices/phase_v.py` has been reviewable data since it was written, and its own docstring says
the run loop "waits for an installed vLLM to be tested against". vLLM 0.27.1 is installed in
`.venv-vllm` and `vllm_probe.py` has been run against it, so this is that loop.

It is a separate file from bench.py rather than a mode inside it, because almost nothing is
shared: no gguf, no llama.cpp tree, no `timings.predicted_per_second`, and a different definition
of what "one arm" costs to start. What IS shared is deliberate and is imported rather than
reimplemented -- the prompt set, the degeneracy and divergence measures, the GPU lock, the settle
gate and the host-load telemetry -- so that a number from this file and a number from bench.py
mean the same thing.

Three decisions that are not obvious:

  * Decode rate comes from `/metrics`, not from completion_tokens over wall time. vLLM's OpenAI
    endpoint publishes no per-token timing, and a wall-clock rate carries prefill inside it. That
    does not cancel when each engine is divided by its own baseline: a speculative arm decodes
    faster, so prefill is a larger share of its wall time and the speedup comes out too small.
    Getting this wrong is what llama.cpp #27623 did before its author withdrew the 25x figure.

  * `own_names` for the host-load probe has to include vLLM's own processes. `setproctitle`
    renames them, and the rename reaches /proc/pid/comm truncated to 15 characters, so the engine
    appears in `ps` as `VLLM::EngineCor`. Without it every arm-pass would be flagged as contended
    by its own server, at several hundred percent CPU.

  * A speculative arm that publishes no drafted tokens is a failure, not a zero. vLLM accepts a
    `--speculative-config` it then ignores, and an arm recorded as speculative on no evidence is
    the error this whole study exists to avoid.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import devices as DEV
import gpustate as G
import prompts as P
import quality
import telemetry as T
import vllm_server as V

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# A fallback only: `telemetry.host_load` attributes by descent, and every vLLM process this
# driver starts is a descendant of it. The name matters when one has been reparented, and it is a
# prefix because setproctitle also sets /proc/pid/comm, which the kernel truncates to 15
# characters -- "VLLM::EngineCore" arrives in `ps` as "VLLM::EngineCor". Verified 2026-08-26
# against the setproctitle in .venv-vllm. python3 is deliberately NOT here: it would make every
# other python on the host invisible, which is the hole descent was introduced to close.
OWN_PROCESS_NAMES = ("VLLM::",)


def _cmd(*argv: str) -> str | None:
    try:
        return subprocess.check_output(argv, text=True, timeout=30).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment_snapshot(binary: str, model: str, matrix_provenance: dict) -> dict:
    """What would have to be equal for a rerun to be a rerun.

    Deliberately parallel to bench.py's snapshot of the same name, with the llama.cpp tree
    revisions replaced by the vLLM and torch versions. The model is named rather than hashed: it
    is a Hugging Face repo id resolved through the local cache, so the revision -- recorded
    beside it when the cache exposes one -- is what pins the weights.
    """
    ver = _cmd(binary, "--version")
    torch_ver = None
    try:
        torch_ver = subprocess.check_output(
            [str(Path(binary).parent / "python"), "-c", "import torch;print(torch.__version__)"],
            text=True, timeout=120).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "engine": "vllm",
        "vllm_version": ver,
        "torch_version": torch_ver,
        "vllm_binary": binary,
        "gpu": _cmd("nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap,"
                                 "power.limit,power.default_limit,power.max_limit,"
                                 "clocks.max.sm,clocks.max.mem",
                    "--format=csv,noheader"),
        "gpu_state_at_start": T.gpu_snapshot(),
        "overclock_state": T.overclock_state(),
        "model": model,
        "model_revision": _hf_revision(model),
        "matrix": matrix_provenance,
    }


def _hf_revision(repo_id: str) -> str | None:
    """The commit the local cache resolved this repo to, when it can be read.

    Returned as None rather than guessed when the cache layout does not match: a wrong revision
    recorded as fact is worse than an absent one.
    """
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    d = cache / ("models--" + repo_id.replace("/", "--")) / "refs" / "main"
    try:
        return d.read_text().strip()
    except OSError:
        return None


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def assert_arms_fit(dev, arms: list[dict], required_vram_gb: float, label: str) -> None:
    """Refuse before loading anything, on the matrix figure and on each arm's own.

    Both numbers were declared in `matrices/phase_v.py` and read by nothing: `REQUIRES_VRAM_GB`
    is consumed by bench.py, which that matrix says does not run it, and the per-arm
    `requires_vram_gb` on the DFlash2 arms had no consumer at all. Those arms carry 40.0 because
    a 3.58 GiB BF16 speculator does not fit beside an 18.1 GiB target and a KV cache on a 24 GiB
    card, so the gate is the difference between an early refusal and an OOM partway through.
    """
    if required_vram_gb:
        DEV.assert_capacity(dev, required_vram_gb, f"the {label} matrix")
    for arm in arms:
        need = float(arm.get("requires_vram_gb") or 0.0)
        if need:
            DEV.assert_capacity(dev, need, f"arm {arm['name']!r}")


def arm_order_for_pass(arms: list[dict], p_idx: int) -> list[dict]:
    """Rotate arm order by pass, as bench.py does.

    A fixed order confounds arm with position in the session: the card is coldest at the start of
    a pass and the first arm gets it every time. Rotation spreads that across arms instead of
    letting it land on one.
    """
    if not arms:
        return []
    k = (p_idx - 1) % len(arms)
    return arms[k:] + arms[:k]


def run(arms: list[dict], *, binary: str, model: str, common_args: list[str], baseline: str,
        passes: int, port: int, out_path: Path, max_tokens: int, gpu_index: int = 0,
        required_vram_gb: float = 0.0, matrix_provenance: dict | None = None,
        settle_temp_c: float | None = 60.0) -> dict:
    dev = DEV.get_device(gpu_index)
    assert_arms_fit(dev, arms, required_vram_gb, out_path.stem)

    G.acquire_lock(f"vllm_bench.py -> {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path.with_suffix(".records.jsonl")
    log_dir = out_path.parent / "server_logs"

    result = {
        "schema": "qwen38-specdec/1",
        "env": environment_snapshot(binary, model, matrix_provenance or {}),
        "matrix": matrix_provenance or {},
        "design": {
            "passes": passes,
            "interleaved": True,
            "fresh_server_per_arm_per_pass": True,
            "prompt_order": "fixed, identical across arms",
            "max_tokens": max_tokens,
            "common_args": list(common_args),
            "n_prompts": len(P.PROMPTS),
            "prompt_classes": {c: len(P.by_class(c)) for c in P.CLASSES},
            "full_prompt_set": len(P.PROMPTS) == 25,
            "prompt_tags": [p.tag for p in P.PROMPTS],
            "decode_rate_source": "vllm:request_decode_time_seconds_sum over "
                                  "vllm:generation_tokens_total, prefill and queueing excluded",
        },
        "arms": [a["name"] for a in arms],
        "arm_notes": {a["name"]: a.get("note", "") for a in arms},
        "baseline": baseline,
        "records": [],
        "incidents": [],
        "arm_pass_host_load": {},
        "arm_pass_gpu_state": {},
        "arm_pass_settle": {},
        "arm_pass_spec": {},
        "arm_pass_failed": {},
        "arm_order_by_pass": {},
    }
    _atomic_write_json(out_path, result)

    baseline_text: dict[tuple[str, int], str] = {}

    for p_idx in range(1, passes + 1):
        pass_arms = arm_order_for_pass(arms, p_idx)
        result["arm_order_by_pass"][str(p_idx)] = [a["name"] for a in pass_arms]
        for arm in pass_arms:
            tag = f"pass{p_idx:02d}_{arm['name']}"
            print(f"\n=== {tag} ===", flush=True)
            log_path = log_dir / f"{out_path.stem}_{tag}.log"

            if settle_temp_c is not None:
                result["arm_pass_settle"][tag] = T.settle_gpu(gpu_index,
                                                              target_temp_c=settle_temp_c)
            load = T.host_load(own_names=OWN_PROCESS_NAMES)
            result["arm_pass_host_load"][tag] = load
            if load.get("contended"):
                result["incidents"].append({
                    "kind": "host_contended_at_arm_entry", "arm_pass": tag,
                    "competing_pct": load["competing_pct"],
                    "competing": load["competing"],
                    "note": "another workload held the CPU when this arm started; its timings "
                            "are not comparable to an uncontended arm's",
                })
            result["arm_pass_gpu_state"][tag] = G.read_state(gpu_index)

            proc = None
            try:
                proc = V.start(binary, model, port, [*common_args, *arm["args"]], log_path,
                               gpu_index=gpu_index)
            except V.VllmError as e:
                # An arm the matrix marked may_fail is being tested, not assumed: the failure is
                # the result for that arm and the rest of the matrix still runs.
                result["incidents"].append({
                    "kind": "server_failed_to_start", "arm_pass": tag,
                    "expected": bool(arm.get("may_fail")), "error": str(e),
                    "log": str(log_path),
                })
                result["arm_pass_failed"][tag] = str(e)
                _atomic_write_json(out_path, result)
                print(f"  !! {arm['name']} did not start: {e}", flush=True)
                continue

            try:
                for ordinal, pr in enumerate(P.PROMPTS):
                    spec_before = V.spec_counters(port)
                    tim_before = V.timing_counters(port)
                    r = V.chat(port, pr.system, pr.user, max_tokens=max_tokens,
                               temperature=0.0, model=model)
                    tim_after = V.timing_counters(port)
                    spec_after = V.spec_counters(port)

                    rate = V.decode_rate(tim_before, tim_after)
                    delta = V.spec_delta(spec_before, spec_after)
                    text = r.get("content", "")
                    deg = quality.assess_degeneracy(text)

                    rec = {
                        "pass": p_idx,
                        "arm": arm["name"],
                        "prompt": pr.tag,
                        "class": pr.cls,
                        "think": pr.think,
                        "temperature": 0.0,
                        "decode_tok_s": rate.get("decode_tok_s"),
                        "predicted_n": r.get("completion_tokens"),
                        "hit_cap": (r.get("completion_tokens") or 0) >= max_tokens,
                        "finish_reason": r.get("finish_reason"),
                        "prompt_tokens": r.get("prompt_tokens"),
                        "wall_ms": r.get("wall_ms"),
                        "vllm_timing": rate,
                        "spec": {k: v for k, v in delta.items() if k != "counters"},
                        "degeneracy": asdict(deg),
                        "text_len": len(text),
                        "ordinal": ordinal,
                        "text": text,
                    }
                    if rate.get("decode_tok_s") is None:
                        # Never silently: a missing rate means the counters could not be read,
                        # and a record carrying None where every other record carries a number
                        # is the shape that gets averaged over without anyone noticing.
                        result["incidents"].append({
                            "kind": "decode_rate_unavailable", "arm_pass": tag,
                            "prompt": pr.tag, "timing_series_seen": sorted(tim_after)[:8],
                        })
                    result["records"].append(rec)
                    _append_jsonl(jsonl_path, rec)

                    if arm["name"] == baseline:
                        baseline_text[(pr.tag, p_idx)] = text

                    flags = []
                    if not rec["hit_cap"]:
                        flags.append(f"SHORT({rec['predicted_n']})")
                    if deg.is_degenerate:
                        flags.append("DEGEN")
                    if delta.get("accept_rate") is not None:
                        flags.append(f"acc={delta['accept_rate']:.2f}")
                    rstr = (f"{rec['decode_tok_s']:7.2f}" if rec["decode_tok_s"] else "   n/a ")
                    print(f"  [{pr.cls:6s}] {pr.tag:22s} {rstr} tok/s  "
                          f"n={rec['predicted_n']}  {'  '.join(flags)}", flush=True)

                # Acceptance over the whole arm-pass, which is what the llama.cpp side reports
                # from the server log, so the two engines are compared on the same aggregate.
                arm_spec = V.spec_delta({}, V.spec_counters(port))
                result["arm_pass_spec"][tag] = {k: v for k, v in arm_spec.items()
                                                if k != "counters"}
                if arm.get("expects_drafter"):
                    try:
                        V.assert_speculation_observed(arm_spec, arm["name"])
                    except V.VllmError as e:
                        result["incidents"].append({
                            "kind": "declared_speculative_but_never_drafted",
                            "arm_pass": tag, "error": str(e),
                        })
                        result["arm_pass_failed"][tag] = str(e)
                elif arm_spec.get("drafted"):
                    result["incidents"].append({
                        "kind": "baseline_drafted_tokens", "arm_pass": tag,
                        "drafted": arm_spec["drafted"],
                        "note": "the arm that defines the non-speculative reference ran a "
                                "drafter; every speedup measured against it is understated",
                    })
            finally:
                V.stop(proc)
                _atomic_write_json(out_path, result)

        # Deferred to the end of the pass because arm order rotates: the baseline may run after
        # some treatment arms within a pass.
        for rec in result["records"]:
            if rec["pass"] != p_idx or rec["arm"] == baseline or "divergence" in rec:
                continue
            ref = baseline_text.get((rec["prompt"], p_idx))
            if ref is None:
                rec["baseline_comparison_unavailable"] = True
                continue
            rec["compared_against"] = baseline
            rec["divergence"] = asdict(quality.compare_outputs(ref, rec.get("text", "")))
            rec["rel_degeneracy"] = asdict(
                quality.assess_against_baseline(ref, rec.get("text", "")))
        _atomic_write_json(out_path, result)

    result["gpu_state_at_end"] = G.read_state(gpu_index)
    _atomic_write_json(out_path, result)
    # Released here, not left for the next run to take over as stale. acquire_lock does handle a
    # dead owner, but a lock file outliving its process makes `test -f .gpu-in-use.lock` -- which
    # is how a human and every guard in this repository check whether a measurement is running --
    # answer yes when nothing is.
    G.release_lock()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="phase_v", help="python module in harness/matrices/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--port", type=int, default=18211)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=P.MAX_TOKENS)
    ap.add_argument("--binary", default=str(REPO / ".venv-vllm" / "bin" / "vllm"))
    ap.add_argument("--prompts-per-class", type=int, default=0,
                    help="dry-run a matrix cheaply; the result records that it was reduced")
    ap.add_argument("--include-a6000-arms", action="store_true",
                    help="also run the arms the matrix marks as needing a larger card; the "
                         "per-arm VRAM gate still has to pass")
    ap.add_argument("--settle-floor", action="store_true")
    args = ap.parse_args()

    import importlib
    sys.path.insert(0, str(HERE / "matrices"))
    mod = importlib.import_module(args.matrix)

    arms = list(mod.ARMS)
    if args.include_a6000_arms:
        arms += list(getattr(mod, "A6000_ONLY_ARMS", []))

    if args.prompts_per_class:
        kept = []
        for cls in P.CLASSES:
            kept.extend(P.by_class(cls)[:args.prompts_per_class])
        P.PROMPTS = tuple(kept)
        print(f"!! REDUCED PROMPT SET: {len(P.PROMPTS)} prompts "
              f"({args.prompts_per_class}/class). This is a dry run, not a result.")

    mfile = Path(mod.__file__)
    import hashlib
    provenance = {
        "module": args.matrix,
        "file": mfile.name,
        "file_sha256": hashlib.sha256(mfile.read_bytes()).hexdigest(),
        "knobs": {k: v for k, v in sorted(os.environ.items()) if k.startswith("QWEN_")},
        "argv": sys.argv[1:],
        "driver": "vllm_bench.py",
    }

    run(arms,
        binary=args.binary,
        model=mod.TARGET,
        common_args=list(mod.COMMON_ARGS),
        baseline=mod.BASELINE,
        passes=args.passes,
        port=args.port,
        out_path=Path(args.out),
        max_tokens=args.max_tokens,
        gpu_index=args.gpu,
        required_vram_gb=float(getattr(mod, "REQUIRES_VRAM_GB", 0.0)),
        matrix_provenance=provenance,
        settle_temp_c=(None if args.settle_floor else 60.0))


if __name__ == "__main__":
    main()
