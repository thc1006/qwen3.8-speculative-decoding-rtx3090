"""Paired, interleaved, class-stratified benchmark runner.

Design (fixed in PREREGISTRATION.md, justified in docs/METHODOLOGY_AUDIT.md):

    for pass in 1..N:                 # arms INTERLEAVED inside each pass, not blocked
        for arm in arms:              # fresh server per arm per pass
            start server -> assert port ownership -> assert drafter loaded
            warmup (discarded)
            for prompt in PROMPTS:    # fixed order, identical across arms
                generate, sampling GPU power for exactly that request
            stop server -> wait for port release

Running each arm to completion before the next confounds any session drift (thermal soak,
clock behaviour, background load) with arm identity. Interleaving spreads that drift across
all arms instead of loading it onto whichever ran last.

Every request records: decode rate, prompt/predicted token counts, finish reason, acceptance
counters, integrated GPU energy, temperature and clock, the full generated text, degeneracy,
and -- for speculative arms at greedy -- byte-level divergence from the baseline's own output
for that same prompt and pass.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import devices as DEV
import filler as FILLER
import gpustate as G
import prompts as P
import quality
import server as S
import telemetry as T

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


@dataclass
class Arm:
    """One server configuration under test."""
    name: str
    extra_args: list[str] = field(default_factory=list)
    tree: str = "master"            # which llama.cpp build tree
    expects_drafter: bool = False   # assert a drafter actually attached
    temperature: float = 0.0
    note: str = ""
    gpu_state: "G.GpuState | None" = None   # resource condition; None = leave the card alone


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Write the result file atomically.

    `Path.write_text` opens with mode 'w', which truncates before writing. This file is rewritten
    after every arm -- 35 times in a 5-pass matrix, at a few megabytes each -- so a process death
    or a full disk inside that window would leave a truncated file and destroy the entire run's
    data. Writing to a sibling temp file, fsyncing it, and renaming makes the replacement atomic
    on POSIX: the reader sees either the old complete file or the new complete file, never a
    half-written one.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, ensure_ascii=False, indent=1)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: Path, rec: dict) -> None:
    """Append one record to a parallel line-delimited stream, flushed immediately.

    Independent of the main JSON. If that file is ever lost or corrupted, every measured record
    is still recoverable from here in the order it was produced.
    """
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        pass


def _with_filler(filler_text: str, user: str) -> str:
    """Frame the filler as a document the question is about, so the model has a reason for it.

    Dropping tens of thousands of tokens of unexplained prose in front of a question makes the
    model try to continue the novel. Naming it as a document to be set aside keeps the answer
    on the question while the KV cache still carries the depth, which is the thing being varied.
    """
    if not filler_text:
        return user
    return (f"Reference document (background only, do not summarise it):\n\n{filler_text}\n\n"
            f"---\n\nIgnore the document above unless it is relevant. {user}")


def _pid_tree(pid: int) -> tuple[int, ...]:
    """The server pid plus its descendants (llama-server may fork)."""
    pids = {pid}
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,ppid", "--no-headers"], text=True)
        children: dict[int, list[int]] = {}
        for line in out.splitlines():
            a, b = line.split()
            children.setdefault(int(b), []).append(int(a))
        stack = [pid]
        while stack:
            cur = stack.pop()
            for c in children.get(cur, []):
                if c not in pids:
                    pids.add(c); stack.append(c)
    except Exception:
        pass
    return tuple(pids)


def _rev(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    """Hash the model once and cache it beside the file, keyed on size+mtime.

    Re-hashing a 17.5 GB GGUF on every launch costs about a minute and produces the same
    string every time. The cache is invalidated if size or mtime changes, so a swapped model
    file is still caught.
    """
    try:
        st = path.stat()
    except OSError:
        return "unknown"
    stamp = f"{st.st_size}:{int(st.st_mtime)}"
    cache = path.with_suffix(path.suffix + ".sha256")
    if cache.exists():
        try:
            cached_stamp, cached_hash = cache.read_text().strip().split(None, 1)
            if cached_stamp == stamp:
                return cached_hash
        except Exception:
            pass
    try:
        h = subprocess.check_output(["sha256sum", str(path)], text=True).split()[0]
    except Exception:
        return "unknown"
    try:
        cache.write_text(f"{stamp} {h}\n")
    except OSError:
        pass
    return h


def environment_snapshot(trees: dict[str, Path], model: Path) -> dict:
    def _cmd(*c) -> str:
        try:
            return subprocess.check_output(c, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "gpu": _cmd("nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version,compute_cap,power.limit,"
                    "power.default_limit,power.max_limit,clocks.max.graphics,clocks.max.memory",
                    "--format=csv,noheader"),
        "gpu_state_at_start": T.gpu_snapshot(),
        "overclock_state": T.overclock_state(),
        "llama_cpp_revisions": {k: _rev(v) for k, v in trees.items()},
        "model": str(model),
        "model_sha256": _sha256(model),
    }


# Set by main() when the matrix itself declares a reduced prompt set (Phase L), as opposed to
# --prompts-per-class being passed on the command line to dry-run a matrix.
PROMPT_SUBSET_IS_DELIBERATE = False


def run_matrix(
    arms: list[Arm],
    *,
    binaries: dict[str, Path],
    trees: dict[str, Path],
    model: Path,
    common_args: list[str],
    passes: int,
    port: int,
    out_path: Path,
    gpu_index: int = 0,
    warmup: int = 1,
    max_tokens: int = P.MAX_TOKENS,
    power_interval_s: float = 0.10,
    prefill_reps: int = 8,
    baseline_map: dict[str, str] | None = None,
    settle_temp_c: float | None = 60.0,
    settle_margin_c: float = 8.0,
    required_vram_gb: float = 0.0,
    context_filler_tokens: int = 0,
    cache_prompt: bool = False,
    settle_max_wait_s: float = 240.0,
    allow_non_stock: bool = False,
) -> dict:
    if required_vram_gb:
        DEV.assert_capacity(DEV.get_device(gpu_index), required_vram_gb,
                            f"matrix in {out_path.name}")
    varies_resources = any(a.gpu_state is not None for a in arms)
    oc = T.overclock_state(gpu_index)
    if not oc.get("is_stock") and not allow_non_stock and not varies_resources:
        raise RuntimeError(
            "GPU is not at stock settings and allow_non_stock is False. Refusing to run: an "
            "undisclosed overclock silently moves bandwidth-bound and compute-bound arms by "
            "different amounts.\n"
            f"  power limit      : {oc.get('power_limit_w')} W "
            f"(default {oc.get('power_default_limit_w')} W)\n"
            f"  memory offset    : {oc.get('mem_transfer_rate_offset')}\n"
            f"  core offset      : {oc.get('graphics_clock_offset')}\n"
            "Reset with:\n"
            "  DISPLAY=:0 nvidia-settings -a '[gpu:0]/GPUMemoryTransferRateOffset[4]=0'\n"
            "  DISPLAY=:0 nvidia-settings -a '[gpu:0]/GPUGraphicsClockOffset[4]=0'\n"
            "  sudo nvidia-smi -pl <default>\n"
            "or pass --allow-non-stock when the overclock is a deliberate experimental factor.")

    dev = DEV.get_device(gpu_index)
    print(f"device: {dev.describe()}", flush=True)
    neighbours = DEV.other_devices_state(gpu_index)
    for n in neighbours:
        flag = "  <-- BUSY" if n.get("looks_busy") else ""
        print(f"  neighbour GPU {n.get('index')}: {n.get('name')} "
              f"util={n.get('utilization_pct')}% mem={n.get('memory_used_mib')}MiB{flag}",
              flush=True)

    # The fixed 60 C gate is calibrated for this open-air RTX 3090 in this chassis. On any other
    # card it is a number with no justification: possibly unreachable (a timeout on every arm) or
    # trivially met (a gate that does nothing). Rather than let that pass silently, insist on the
    # measured-floor mode.
    _CALIBRATED_FOR = "3090"
    measured_floor = None
    if settle_temp_c is not None and _CALIBRATED_FOR not in dev.name.replace(" ", ""):
        raise RuntimeError(
            f"the fixed {settle_temp_c:.0f} C thermal gate is calibrated for an RTX 3090, but "
            f"device {gpu_index} is {dev.name!r}. Re-run with --settle-floor so the gate is "
            f"derived from this card's own measured idle temperature, or state an explicit "
            f"target you have justified for it.")
    if settle_temp_c is None:
        measured_floor = DEV.idle_floor_c(gpu_index)
        print(f"thermal gate targets {measured_floor + settle_margin_c:.0f} C "
              f"(measured floor {measured_floor:.0f} C + {settle_margin_c:.0f} C margin)",
              flush=True)

    declared_state = G.read_state(gpu_index)

    # Order matters. The lock is taken FIRST, and only then is the restore-on-exit guard armed.
    # The guard restores stock clocks with force=True, which bypasses the lock by design so that
    # a dying run can always clean up after itself — but armed before the lock is held, a run
    # that then fails to acquire the lock would fire that guard on exit and move the clocks of a
    # card another run is actively measuring.
    G.acquire_lock(f"bench.py -> {out_path}")
    if varies_resources:
        G.install_restore_guard(gpu_index)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path.with_suffix('.records.jsonl')
    log_dir = out_path.parent / "server_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "schema": "qwen38-specdec/1",
        "env": environment_snapshot(trees, model),
        "gpu_state_declared": declared_state,
        "design": {
            "passes": passes,
            "interleaved": True,
            "fresh_server_per_arm_per_pass": True,
            "prompt_order": "fixed, identical across arms",
            "max_tokens": max_tokens,
            "common_args": common_args,
            "warmup_requests_discarded": warmup,
            "prefill_calibration_reps": prefill_reps,
            "thermal_settle_target_c": settle_temp_c,
            "thermal_settle_margin_c": settle_margin_c,
            "device": {"index": dev.index, "name": dev.name, "tag": dev.short,
                       "vram_gb": round(dev.vram_gb, 1),
                       "compute_cap": dev.compute_cap, "driver": dev.driver,
                       "power_default_w": dev.power_default_w,
                       "power_min_w": dev.power_min_w, "power_max_w": dev.power_max_w,
                       "clocks_max_memory_mhz": dev.clocks_max_memory_mhz,
                       "clocks_max_graphics_mhz": dev.clocks_max_graphics_mhz,
                       "ecc_mode": DEV.ecc_mode(gpu_index)},
            "neighbour_devices_at_start": neighbours,
            "thermal_settle_max_wait_s": settle_max_wait_s,
            "cache_prompt": cache_prompt,
            "n_prompts": len(P.PROMPTS),
            "prompt_classes": {c: len(P.by_class(c)) for c in P.CLASSES},
            "full_prompt_set": len(P.PROMPTS) == 25,
            # A short prompt set is either a cheap dry run or a matrix's declared design. Only
            # the first invalidates the result, so the two are recorded apart rather than being
            # collapsed into one "not full" flag that a later reader would have to guess about.
            "prompt_subset_declared_by_matrix": PROMPT_SUBSET_IS_DELIBERATE,
            "prompt_tags": [p.tag for p in P.PROMPTS],
            "context_filler_tokens_requested": context_filler_tokens,
        },
        "baseline_map": baseline_map or {},
        "arms": {a.name: {"extra_args": a.extra_args, "tree": a.tree,
                          "expects_drafter": a.expects_drafter,
                          "temperature": a.temperature, "note": a.note,
                          "gpu_state": (None if a.gpu_state is None else {
                              "name": a.gpu_state.name,
                              "mem_transfer_offset": a.gpu_state.mem_transfer_offset,
                              "core_offset": a.gpu_state.core_offset,
                              "power_limit_w": a.gpu_state.power_limit_w})}
                 for a in arms},
        "records": [],
        "incidents": [],
    }

    # baseline text per (prompt, pass) -> used as the reference for divergence + degeneracy
    baseline_text: dict[tuple[str, int], str] = {}
    first_pass_text: dict[tuple[str, str], str] = {}
    baseline_name = arms[0].name

    for p_idx in range(1, passes + 1):
        # Rotate arm order every pass. Interleaving alone still leaves a fixed position effect:
        # whichever arm always runs first meets a cooler card and an emptier page cache than
        # whichever always runs last. Rotation spreads that position effect across arms instead
        # of assigning it to one.
        rot = (p_idx - 1) % len(arms)
        pass_arms = arms[rot:] + arms[:rot]
        result.setdefault("arm_order_by_pass", {})[str(p_idx)] = [a.name for a in pass_arms]
        for arm in pass_arms:
            tag = f"pass{p_idx:02d}_{arm.name}"
            print(f"\n=== {tag} ===", flush=True)
            log_path = log_dir / f"{tag}.log"
            binary = binaries[arm.tree]

            try:
                # Nothing else may hold GPU memory: power is sampled device-wide, so a second
                # tenant corrupts every energy figure, and a competing workload corrupts every
                # timing figure.
                T.assert_gpu_exclusive(gpu_index)
                # Clocks and power limit must be identical to what the run started with.
                # Recording the state once at startup does not catch a change made halfway
                # through -- which is exactly how one development run was contaminated.
                # An arm may deliberately declare a resource condition (Phase R). The run owns
                # the lock, so it is allowed to move the clocks; anything else is not.
                if arm.gpu_state is not None:
                    applied = G.apply(arm.gpu_state, gpu_index, force=True)
                    result.setdefault("arm_pass_gpu_applied", {})[tag] = applied
                    print(f"  gpu condition '{arm.gpu_state.name}': mem_off="
                          f"{applied.get('mem_transfer_offset')} core_off="
                          f"{applied.get('core_offset')} pl={applied.get('power_limit_w')}W "
                          f"mem_clk={applied.get('clocks_max_memory_mhz')}MHz", flush=True)
                    expect = {"mem_transfer_offset": arm.gpu_state.mem_transfer_offset,
                              "core_offset": arm.gpu_state.core_offset,
                              "power_limit_w": float(arm.gpu_state.power_limit_w)}
                else:
                    expect = {k: declared_state.get(k)
                              for k in ("mem_transfer_offset", "core_offset", "power_limit_w",
                                        "clocks_max_memory_mhz")}

                now_state = G.read_state(gpu_index)
                drift = {k: (v, now_state.get(k)) for k, v in expect.items()
                         if v is not None and now_state.get(k) != v}
                if drift:
                    result["incidents"].append({
                        "pass": p_idx, "arm": arm.name, "kind": "gpu_state_changed_mid_run",
                        "detail": f"declared vs now: {drift}"})
                    raise RuntimeError(
                        f"GPU clock/power state changed during the run: {drift}. Every number "
                        f"measured after the change is on a different machine than the ones "
                        f"before it. Aborting rather than mixing them.")
                result.setdefault("arm_pass_gpu_state", {})[tag] = now_state
                # Every arm starts from the same thermal state. Without this the card sits on
                # its power cap and loses ~9% of its SM clock over a pass, handing whichever
                # arm runs first a measurably faster GPU than whichever runs last.
                settle = T.settle_gpu(gpu_index, target_temp_c=settle_temp_c,
                                      max_wait_s=settle_max_wait_s)
                result.setdefault("arm_pass_settle", {})[tag] = settle
                if not settle["reached_target"]:
                    result["incidents"].append({
                        "pass": p_idx, "arm": arm.name, "kind": "thermal_settle_timeout",
                        "detail": f"entry temp {settle['entry_temp_c']} C did not reach "
                                  f"{settle_temp_c} C within {settle_max_wait_s}s"})
                print(f"  settled: {settle['start_temp_c']}C -> {settle['entry_temp_c']}C "
                      f"in {settle['waited_s']}s (clock {settle['entry_sm_clock_mhz']} MHz)",
                      flush=True)
                h = S.start(binary, model, arm.extra_args, port=port, log_path=log_path,
                            common_args=common_args, gpu_index=gpu_index)
            except S.ServerError as e:
                result["incidents"].append(
                    {"pass": p_idx, "arm": arm.name, "kind": "server_start_failed",
                     "detail": str(e)[:4000]})
                print(f"  !! server start failed: {str(e)[:300]}", flush=True)
                continue

            drafter_evidence = None
            try:
                if arm.expects_drafter:
                    drafter_evidence = S.assert_drafter_loaded(h, arm.name)
                    print(f"  drafter: {drafter_evidence[:120]}", flush=True)
                print(f"  ready in {h.ready_s:.1f}s", flush=True)
                # After the server is up, it is the ONLY permitted GPU tenant.
                T.assert_gpu_exclusive(gpu_index, allow_pids=_pid_tree(h.proc.pid))

                # Long-context phases prepend a shared block of real prose. It is built once
                # per arm, against this server's own tokenizer, and the REALISED token count is
                # recorded: a depth that did not materialise must be visible in the data.
                filler_text, filler_n = "", 0
                if context_filler_tokens:
                    filler_text, filler_n = FILLER.filler_of(port, context_filler_tokens)
                    result.setdefault("arm_pass_filler", {})[tag] = {
                        "requested": context_filler_tokens, "realised": filler_n,
                        "chars": len(filler_text)}
                    print(f"  filler: {filler_n} tokens realised "
                          f"({context_filler_tokens} requested)", flush=True)
                    if abs(filler_n - context_filler_tokens) > 64:
                        result["incidents"].append({
                            "pass": p_idx, "arm": arm.name, "kind": "filler_depth_missed",
                            "detail": f"requested {context_filler_tokens} tokens, "
                                      f"realised {filler_n}"})

                for i in range(warmup):
                    wr = S.chat(port, "You are concise.", f"warmup {i}",
                                max_tokens=32, temperature=arm.temperature,
                                cache_prompt=cache_prompt)
                # Behavioural proof, on a real generation, that the speculative path actually
                # ran. The server log says what llama.cpp printed; t_draft_n says what it did.
                if arm.expects_drafter:
                    n_drafted = S.assert_drafting_observed(wr, arm.name)
                    print(f"  drafting confirmed: t_draft_n={n_drafted} on warmup", flush=True)

                for pr in P.PROMPTS:
                    # Prefill calibration: same prompt, one token out, REPEATED so the window
                    # is long enough to integrate. The power sampler cannot see where prefill
                    # ends inside a non-streaming request, so prefill energy is measured
                    # directly and subtracted, giving a decode-only tok/J rather than an
                    # end-to-end figure that silently includes prompt processing.
                    #
                    # The repetition is not optional. A single prefill of these prompts takes
                    # on the order of 0.1 s at ~1300 tok/s prompt-processing, which at a 0.1 s
                    # sampling interval yields one or two samples -- below the two-sample
                    # minimum the integrator requires, so energy would come back None and the
                    # entire decode tok/J column would be empty. Repeating K times and dividing
                    # gives a window that can actually be integrated.
                    # With the cache on, repeats after the first are served from it and measure
                    # nothing. A long-context prefill is seconds long anyway, so one pass is
                    # plenty to integrate.
                    reps_here = 1 if cache_prompt else prefill_reps
                    with T.sampling(index=gpu_index, interval_s=power_interval_s) as pfs:
                        for _ in range(reps_here):
                            # Must be the SAME prompt as the measured request, filler included.
                            # Calibrating against a short prompt and subtracting that from a
                            # long-context request would leave most of the prefill energy in
                            # the "decode" figure.
                            pf = S.chat(port, pr.system, _with_filler(filler_text, pr.user),
                                        max_tokens=1, temperature=arm.temperature,
                                        think=pr.think, cache_prompt=cache_prompt)
                    prefill_power = pfs.summary()
                    if prefill_power.get("energy_j") is not None:
                        prefill_power["energy_j"] /= reps_here
                    prefill_power["reps"] = reps_here

                    with T.sampling(index=gpu_index, interval_s=power_interval_s) as ps:
                        r = S.chat(port, pr.system, _with_filler(filler_text, pr.user),
                                   max_tokens=max_tokens, temperature=arm.temperature,
                                   think=pr.think, cache_prompt=cache_prompt)
                    power = ps.summary()

                    text = (r.get("reasoning_content") or "") + (r.get("content") or "")
                    predicted_n = int(r.get("t_predicted_n") or r.get("completion_tokens") or 0)
                    rate = r.get("t_predicted_per_second")
                    if rate is None and r.get("t_predicted_ms"):
                        rate = predicted_n / (r["t_predicted_ms"] / 1000.0)

                    energy = power.get("energy_j")
                    prefill_energy = prefill_power.get("energy_j")
                    tok_per_j_e2e = (predicted_n / energy) if (energy and predicted_n) else None
                    decode_energy = None
                    tok_per_j_decode = None
                    if cache_prompt:
                        # With the prompt cache on, the calibration request above has already
                        # populated the cache, so the measured request skips the prefill and its
                        # energy is decode energy. Subtracting a separately measured prefill here
                        # would remove work the measured request never did, and at long context
                        # that lands the figure below zero.
                        decode_energy = energy
                        prefill_power["subtracted"] = False
                    elif energy is not None and prefill_energy is not None:
                        decode_energy = energy - prefill_energy
                        prefill_power["subtracted"] = True
                    if decode_energy and decode_energy > 0 and predicted_n:
                        tok_per_j_decode = predicted_n / decode_energy

                    # Direct verification that prompt caching is really off, rather than
                    # trusting that the request field was honoured: llama.cpp reports how many
                    # prompt tokens it served from cache. Anything above zero means a later
                    # prompt was partly free, which would show up as speed and be attributed
                    # to the arm.
                    cache_n = int(r.get("t_cache_n") or 0)
                    if cache_n > 0:
                        result["incidents"].append({
                            "pass": p_idx, "arm": arm.name, "prompt": pr.tag,
                            "kind": "prompt_cache_hit",
                            "detail": f"t_cache_n={cache_n} despite cache_prompt=False; "
                                      f"this request was partly served from cache"})

                    deg = quality.assess_degeneracy(text)
                    rec = {
                        "pass": p_idx,
                        "arm": arm.name,
                        "prompt": pr.tag,
                        "class": pr.cls,
                        "think": pr.think,
                        "temperature": arm.temperature,
                        "decode_tok_s": rate,
                        "predicted_n": predicted_n,
                        "hit_cap": predicted_n >= max_tokens,
                        "finish_reason": r.get("finish_reason"),
                        "prompt_tokens": r.get("prompt_tokens"),
                        "filler_tokens": filler_n,
                        "cache_n": cache_n,
                        "wall_ms": r.get("wall_ms"),
                        "timings": {k: v for k, v in r.items() if k.startswith("t_")},
                        "power": power,
                        "prefill_power": prefill_power,
                        "prefill_predicted_n": int(pf.get("t_predicted_n") or 0),
                        "decode_energy_j": decode_energy,
                        "tok_per_joule_e2e": tok_per_j_e2e,
                        "tok_per_joule_decode": tok_per_j_decode,
                        "degeneracy": asdict(deg),
                        "text_len": len(text),
                        "text": text,
                    }

                    # Greedy + fixed seed + prompt cache disabled should reproduce the same
                    # bytes on every pass. If it does not, the run-to-run spread being measured
                    # is not purely timing noise and that must be visible in the record rather
                    # than averaged away.
                    prev = first_pass_text.get((arm.name, pr.tag))
                    if prev is None:
                        first_pass_text[(arm.name, pr.tag)] = text
                    elif arm.temperature == 0.0:
                        rec["deterministic_vs_pass1"] = (prev == text)
                        if prev != text:
                            result["incidents"].append({
                                "pass": p_idx, "arm": arm.name, "prompt": pr.tag,
                                "kind": "nondeterministic_greedy_output",
                                "detail": "greedy output differs from pass 1 for the same "
                                          "arm/prompt/seed with cache_prompt=False"})

                    # Comparison against baseline is deferred to a post-pass step: with arm
                    # order rotated, the baseline arm is not guaranteed to have run yet.
                    if arm.name == baseline_name:
                        baseline_text[(pr.tag, p_idx)] = text

                    result["records"].append(rec)
                    _append_jsonl(jsonl_path, rec)

                    flags = []
                    if not rec["hit_cap"]:
                        flags.append(f"SHORT({predicted_n})")
                    if deg.is_degenerate:
                        flags.append("DEGEN")
                    rstr = f"{rate:7.2f}" if rate else "   n/a "
                    print(f"  [{pr.cls:6s}] {pr.tag:22s} {rstr} tok/s  "
                          f"n={predicted_n:4d}  {'  '.join(flags)}", flush=True)

                acc = S.parse_acceptance_from_log(h)
                if acc:
                    # Per-ARM aggregate. Attaching it to records[-1] would bolt one arm's
                    # acceptance onto a different arm's record whenever an arm produced no
                    # records of its own.
                    result.setdefault("arm_pass_acceptance", {})[tag] = acc

            except Exception as e:                        # noqa: BLE001
                result["incidents"].append(
                    {"pass": p_idx, "arm": arm.name, "kind": type(e).__name__,
                     "detail": str(e)[:4000]})
                print(f"  !! {type(e).__name__}: {str(e)[:300]}", flush=True)
            finally:
                # Keyed by arm-pass, never appended to a neighbouring arm's record.
                result.setdefault("arm_pass_gpu", {})[tag] = {
                    "at_start": h.gpu_at_start,
                    "at_end": T.gpu_snapshot(gpu_index),
                    "ready_s": getattr(h, "ready_s", None),
                }
                if drafter_evidence:
                    result.setdefault("drafter_evidence", {})[tag] = drafter_evidence
                try:
                    S.stop(h.proc, port=port)
                except Exception as e:                    # noqa: BLE001
                    result["incidents"].append(
                        {"pass": p_idx, "arm": arm.name, "kind": "server_stop_failed",
                         "detail": str(e)[:2000]})
                # (thermal settling now happens at arm ENTRY via T.settle_gpu, which gates on
                # a measured temperature rather than guessing with a fixed sleep)

            _atomic_write_json(out_path, result)

        # ---- post-pass: compare every arm against the baseline's own text for this pass ----
        _attach_baseline_comparisons(result, baseline_text, baseline_name, p_idx)
        _atomic_write_json(out_path, result)

    if any(a.gpu_state is not None for a in arms):
        try:
            result["gpu_state_restored"] = G.apply(G.STOCK, gpu_index, force=True)
        except Exception as e:                                   # noqa: BLE001
            result["incidents"].append({"kind": "stock_restore_failed", "detail": repr(e)})
    result["gpu_state_at_end"] = G.read_state(gpu_index)
    G.release_lock()
    _atomic_write_json(out_path, result)
    return result


def _attach_baseline_comparisons(result: dict, baseline_text: dict, baseline_name: str,
                                 p_idx: int) -> None:
    """Attach divergence + relative-degeneracy to every non-baseline record of one pass.

    Deferred until the pass ends because arm order is rotated, so the baseline arm may run
    after some treatment arms within a pass.
    """
    for rec in result["records"]:
        if rec["pass"] != p_idx or rec["arm"] == baseline_name:
            continue
        if "rel_degeneracy" in rec:
            continue
        ref = baseline_text.get((rec["prompt"], p_idx))
        if ref is None:
            rec["baseline_comparison_unavailable"] = True
            continue
        text = rec.get("text", "")
        if rec.get("temperature") == 0.0:
            rec["divergence"] = asdict(quality.compare_outputs(ref, text))
        rec["rel_degeneracy"] = asdict(quality.assess_against_baseline(ref, text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True, help="python module in harness/matrices/")
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--port", type=int, default=18138)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=P.MAX_TOKENS)
    ap.add_argument("--settle-floor", action="store_true",
                    help="gate arm entry on the device's own measured idle floor + margin "
                         "instead of a fixed 60 C; use on any card other than this RTX 3090")
    ap.add_argument("--allow-non-stock", action="store_true",
                    help="permit running on an overclocked card; only for the deliberate "
                         "overclock phase, and the state is recorded either way")
    ap.add_argument("--prompts-per-class", type=int, default=0,
                    help="dry-run aid: keep only the first N prompts of each class. "
                         "0 = use the full frozen set. Any value other than 0 is recorded in "
                         "the result so a reduced run can never be mistaken for a full one.")
    args = ap.parse_args()

    import importlib, sys
    sys.path.insert(0, str(HERE / "matrices"))
    mod = importlib.import_module(args.matrix)

    # A matrix may declare a reduced prompt set as part of its design: Phase L varies context
    # depth, not prompt class, and a full 25-prompt arm at 96 K would run for hours. That is a
    # different thing from --prompts-per-class, which exists to dry-run a matrix cheaply, and the
    # two are labelled differently so a dry run can never be read back as a result.
    declared_ppc = int(getattr(mod, "PROMPTS_PER_CLASS", 0))
    per_class = args.prompts_per_class or declared_ppc
    global PROMPT_SUBSET_IS_DELIBERATE
    deliberate = bool(declared_ppc) and not args.prompts_per_class
    PROMPT_SUBSET_IS_DELIBERATE = deliberate
    if per_class:
        kept = []
        for cls in P.CLASSES:
            kept.extend(P.by_class(cls)[:per_class])
        P.PROMPTS = tuple(kept)
        if deliberate:
            print(f"prompt set: {len(P.PROMPTS)} prompts ({per_class}/class), "
                  f"declared by the {args.matrix} matrix.")
        else:
            print(f"!! REDUCED PROMPT SET: {len(P.PROMPTS)} prompts "
                  f"({per_class}/class). This is a dry run, not a result.")

    # A matrix may also shorten generation; Phase L does, because a 400-token generation past the
    # decode cliff takes minutes. An explicit --max-tokens on the command line still wins.
    max_tokens = args.max_tokens
    if max_tokens == P.MAX_TOKENS and hasattr(mod, "MAX_TOKENS"):
        max_tokens = int(mod.MAX_TOKENS)
        print(f"max_tokens: {max_tokens}, declared by the {args.matrix} matrix.")

    run_matrix(
        mod.ARMS,
        binaries=mod.BINARIES,
        trees=mod.TREES,
        model=mod.MODEL,
        common_args=mod.COMMON_ARGS,
        passes=args.passes,
        port=args.port,
        out_path=Path(args.out),
        gpu_index=args.gpu,
        max_tokens=max_tokens,
        allow_non_stock=args.allow_non_stock,
        baseline_map=getattr(mod, "BASELINE_MAP", None),
        required_vram_gb=getattr(mod, "REQUIRES_VRAM_GB", 0.0),
        context_filler_tokens=getattr(mod, "CONTEXT_FILLER_TOKENS", 0),
        cache_prompt=getattr(mod, "CACHE_PROMPT", False),
        settle_temp_c=(None if args.settle_floor else 60.0),
    )


if __name__ == "__main__":
    main()
