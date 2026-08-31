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
import random
import platform
import subprocess
import time
import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path

import devices as DEV
import filler as FILLER
import gpustate as G
import prompts as P
import quality
import server as S
import kernel_facts as KF
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
    # A matrix that compares two models has to hold both. Without this every arm ran the one
    # model the matrix declared, so Phase M could only ever be a single-architecture replication
    # and its dense side had to come from a different result file measured on a different day.
    model: "Path | None" = None


# Every field of PowerSampler.summary() that is an absolute quantity of energy over the sampled
# window. The prefill calibration's window covers several requests, so each of these has to be
# divided by that count before one request's worth is subtracted from the measured request. A test
# asserts this tuple covers every `energy_j*` key summary() emits: the defect it exists to stop was
# two such fields being added to summary() while the call site kept normalising only the first.
def effective_passes(passes: int, n_arms: int, latin_arms: bool) -> int:
    """How many passes will actually run, which is what `design` has to record.

    `--latin-arms` runs one pass per arm so the rotation closes. It used to do that by
    reassigning `passes` further down, AFTER the result dict -- and its `"passes": passes`
    -- had already been built, so a run under the flag recorded the pre-override count.
    Phase E5's first attempt recorded `design.passes = 5` while three passes ran, and
    `audit_results.py` correctly called the file 225 records of an expected 375 and failed
    it: 55 minutes of card time, and the phase had to be relaunched with `--passes 3`, which
    reaches the same rotation without going through the override at all.

    Two truths for one quantity is the defect, so this resolves it once and both the design
    block and the loop read the result. Nothing else between them touched `passes`, and
    `arms` is a parameter this function never reassigns, so hoisting is safe.

    E5 was the first use of the flag in this study; no committed file was produced with it,
    and the audit being green on all of them is the evidence, since a file with an inflated
    `design.passes` is exactly what that check reports as short.
    """
    return n_arms if (latin_arms and passes != n_arms) else passes


PREFILL_ABSOLUTE_ENERGY_FIELDS = ("energy_j", "energy_j_instant", "energy_j_nvml")


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
        out = subprocess.check_output(["ps", "-eo", "pid,ppid", "--no-headers"],
                                      text=True, timeout=15)
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
                                       text=True, stderr=subprocess.DEVNULL, timeout=30).strip()
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
        # TEN MINUTES, not the fifteen seconds the nvidia-smi calls get. This
        # hashes MODEL files: twenty gigabytes off cold storage is minutes of
        # honest work, and a short bound here would manufacture the failure it
        # is meant to prevent. It is cached by mtime, so it runs once per model
        # per change, and it already degrades to "unknown" rather than raising.
        h = subprocess.check_output(["sha256sum", str(path)], text=True,
                                    timeout=600).split()[0]
    except Exception:
        return "unknown"
    try:
        cache.write_text(f"{stamp} {h}\n")
    except OSError:
        pass
    return h


def _size(path: Path) -> int | None:
    """Bytes of a regular file, or None. Never raises: a snapshot that dies takes the run with it.

    Regular files only. stat() on a directory succeeds and returns 4096, which as a model size
    produces a bits-per-weight figure that is absurd but still a number -- the shape of error
    that gets plotted rather than caught.
    """
    try:
        return path.stat().st_size if path.is_file() else None
    except OSError:
        return None


def environment_snapshot(trees: dict[str, Path], model: Path) -> dict:
    def _cmd(*c) -> str:
        try:
            # environment probes: uname, a version string, a driver query. All
            # of them return at once or are broken.
            return subprocess.check_output(c, text=True, stderr=subprocess.DEVNULL,
                                           timeout=30).strip()
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
        # Recorded because the quantization ladders DELETE their weights once a rung verifies,
        # and bits per weight -- the axis those ladders should be plotted against, since a label
        # like UD-Q5_K_XL is not a number -- is file size over parameter count. The hash proves
        # which file ran; the size is what makes it quantitative. Without this the figure is
        # recoverable only from a download log or by re-fetching 20 GB.
        "model_size_bytes": _size(model),
    }


def arm_model_snapshot(arms) -> dict:
    """{path -> sha256} for every model an arm overrides to, so a two-model run is auditable.

    env.model records the matrix default only. Phase M compares a dense target with an MoE one in
    a single run, and a hash per file is what makes that checkable after the fact.
    """
    out = {}
    for a in arms:
        if a.model and str(a.model) not in out:
            out[str(a.model)] = _sha256(a.model)
    return out


# Set by main() when the matrix itself declares a reduced prompt set (Phase L), as opposed to
# --prompts-per-class being passed on the command line to dry-run a matrix.
PROMPT_SUBSET_IS_DELIBERATE = False


def server_log_path(log_dir: Path, out_path: Path, tag: str) -> Path:
    """Where one arm-pass's server log goes.

    The result stem is in the name because every phase writes into the same
    results/server_logs/, and `pass01_baseline@master` is a name most matrices produce. Without
    it a phase silently overwrites an earlier phase's log: 41 filenames in this repository were
    written by more than one phase before this was fixed, one of them by seven. Nothing in a
    result points at its log either, so the overwrite leaves no trace -- the reader finds a log
    with the right name sitting beside the right result, and it belongs to a different run.
    """
    return log_dir / f"{out_path.stem}_{tag}.log"


def matrix_provenance_snapshot(mod, module_name: str, argv: list[str]) -> dict:
    """Which matrix produced a result, and with which knobs.

    Several matrices are parameterised through the environment and read the variable at import
    time: QWEN_Q_TARGET, QWEN_QS_TARGET, QWEN_L_DEPTH, QWEN_WARP_BUILD, QWEN_WARP_DIR. Nothing
    recorded that, so a result could be tied back to its configuration only by reading the
    parameter out of the arm names, which works until two configurations share a name.

    Every QWEN_* variable in the environment is recorded, not only the ones this matrix reads:
    which ones it reads is a property of the file version, and that is what file_sha256 pins.
    """
    import hashlib
    mfile = Path(mod.__file__)
    return {
        "module": module_name,
        "file": mfile.name,
        "file_sha256": hashlib.sha256(mfile.read_bytes()).hexdigest(),
        "knobs": {k: v for k, v in sorted(os.environ.items()) if k.startswith("QWEN_")},
        "argv": list(argv),
    }


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
    power_roll_s: float = 0.0,
    power_trace: bool = False,
    prefill_reps: int = 8,
    baseline_map: dict[str, str] | None = None,
    matrix_provenance: dict | None = None,
    settle_temp_c: float | None = 60.0,
    settle_margin_c: float = 8.0,
    required_vram_gb: float = 0.0,
    context_filler_tokens: int = 0,
    cache_prompt: bool = False,
    settle_max_wait_s: float = 240.0,
    allow_non_stock: bool = False,
    latin_arms: bool = False,
    shuffle_prompts: bool = False,
    prompt_seed: int = 20260825,
) -> dict:
    if required_vram_gb:
        DEV.assert_capacity(DEV.get_device(gpu_index), required_vram_gb,
                            f"matrix in {out_path.name}")
    # Resolved BEFORE anything records it. `--latin-arms` used to reassign `passes` after
    # the result dict was built, so the design block kept the caller's number while the loop
    # ran a different one; see `effective_passes`.
    if latin_arms and passes != len(arms):
        print(f"--latin-arms: running {len(arms)} passes instead of {passes} so the arm "
              f"rotation closes and every arm visits every order position exactly once",
              flush=True)
    passes = effective_passes(passes, len(arms), latin_arms)

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
    # a dying run can always clean up after itself - but armed before the lock is held, a run
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
        # Which matrix, and with which knobs. Several matrices are parameterised through the
        # environment -- QWEN_Q_TARGET, QWEN_L_DEPTH, QWEN_QS_TARGET, QWEN_WARP_BUILD -- and
        # read them at import time, so without this a result can only be tied back to the
        # configuration that produced it by reading the parameter out of the arm names.
        "matrix": matrix_provenance or {},
        # empty unless some arm overrides the matrix default; see arm_model_snapshot
        "arm_models": arm_model_snapshot(arms),
        "gpu_state_declared": declared_state,
        "design": {
            "passes": passes,
            # `--latin-arms` resolved above, so this is the count that will run and not the
            # one the caller asked for. Recorded either way; see `effective_passes`.
            "latin_arms": latin_arms,
            "interleaved": True,
            "fresh_server_per_arm_per_pass": True,
            "prompt_order": "fixed, identical across arms",
            "max_tokens": max_tokens,
            "common_args": common_args,
            "warmup_requests_discarded": warmup,
            "prefill_calibration_reps": prefill_reps,
            # The sampler's period, which decides how finely both power fields
            # are integrated. It was a function default nothing passed, so every
            # phase before E3 ran at 0.10 s and the file never said so -- and
            # three runs that differ only in this would have been three files
            # with no way to tell them apart.
            "power_interval_s": power_interval_s,
            # Seconds of idle held INSIDE the sampling window on either side of the
            # measured request. A rolled window is not comparable to an unrolled one --
            # its energy includes the roll -- so it is recorded here and
            # energy_instruments.py refuses to sweep a file that declares one.
            "power_roll_s": power_roll_s,
            "power_trace_recorded": power_trace,
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
            # The dispatch facts every width-indexed analysis depends on, read out of the tree
            # that is about to run rather than written into the analyser. Three defects this
            # study shipped were a kernel fact hard-coded in Python and later untrue.
            "kernel_facts": KF.snapshot(trees),
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
                          # env.model names the matrix default. An arm that overrides it has to
                          # say so here, or the record claims one model for a run that used two.
                          "model": str(a.model) if a.model else None,
                          # Every field of the dataclass, not a hand-picked list. The picked
                          # list silently dropped `lock_sm_mhz` when Phase R2 added it, so the
                          # result recorded what the clock measured but not what was asked for.
                          "gpu_state": (None if a.gpu_state is None
                                        else dataclasses.asdict(a.gpu_state))}
                 for a in arms},
        "records": [],
        "incidents": [],
    }

    # Reference text for divergence and relative degeneracy, keyed by which baseline it came
    # from as well as by prompt and pass. A single reference was wrong for a dual-tree run: it
    # sent the PR-branch arms to the master baseline's text and folded any branch difference
    # into the method effect. On Phase A the two baselines turned out byte-identical on all 125
    # prompt-passes so nothing moved, but that is a result of the run, not a property of the
    # design, and the next pair of trees need not agree.
    baseline_text: dict[tuple[str, str, int], str] = {}
    first_pass_text: dict[tuple[str, str], str] = {}
    baseline_names = {a.name for a in arms
                      if not a.extra_args and not getattr(a, "expects_drafter", False)}
    if not baseline_names:
        baseline_names = {arms[0].name}
    # arm -> the baseline it is measured against
    _bmap = dict(baseline_map or {})
    for a in arms:
        if a.name in _bmap or a.name in baseline_names:
            continue
        same_tree = [b for b in arms if b.name in baseline_names and b.tree == a.tree]
        if same_tree:
            _bmap[a.name] = same_tree[0].name
    result["divergence_baseline_map"] = _bmap

    for p_idx in range(1, passes + 1):
        # Rotate arm order every pass. Interleaving alone still leaves a fixed position effect:
        # whichever arm always runs first meets a cooler card and an emptier page cache than
        # whichever always runs last. Rotation spreads that position effect across arms instead
        # of assigning it to one.
        # Arm order. Rotating by pass covers len(arms) positions only if there are at least that
        # many passes; with 7 arms over 5 passes each arm visits 5 of 7, and a different 5, so the
        # residual position effect is unbalanced rather than absent. --latin-arms runs
        # len(arms) passes so the rotation closes, which is the only way to balance it exactly.
        # Off by default: it changes how many passes a matrix runs, which is not a decision this
        # function should make for a study already under way.
        rot = (p_idx - 1) % len(arms)
        pass_arms = arms[rot:] + arms[:rot]
        result.setdefault("arm_order_by_pass", {})[str(p_idx)] = [a.name for a in pass_arms]

        # Prompt order. The fixed order runs the classes in blocks - code, code, code, prose,
        # prose, ... - so a class always meets the server at the same age, and any arm-by-position
        # interaction lands on whichever classes sit late. Every arm in a pass uses the same
        # permutation, so a prompt is still compared against its own baseline under the same
        # conditions; the permutation changes between passes so the position effect is spread
        # rather than assigned. Off by default because turning it on mid-study would change the
        # design between phases; `--shuffle-prompts` is for the re-run that adopts it.
        if shuffle_prompts:
            pass_prompts = list(P.PROMPTS)
            random.Random(prompt_seed + p_idx).shuffle(pass_prompts)
        else:
            pass_prompts = list(P.PROMPTS)
        result.setdefault("prompt_order_by_pass", {})[str(p_idx)] = [pr.tag for pr in pass_prompts]
        for arm in pass_arms:
            tag = f"pass{p_idx:02d}_{arm.name}"
            print(f"\n=== {tag} ===", flush=True)
            log_path = server_log_path(log_dir, out_path, tag)
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
                # --settle-floor computed measured_floor and dropped it here; without it
                # settle_gpu has neither a target nor a floor and raises.
                settle = T.settle_gpu(gpu_index, target_temp_c=settle_temp_c,
                                      idle_floor_c=measured_floor,
                                      margin_c=settle_margin_c,
                                      max_wait_s=settle_max_wait_s)
                result.setdefault("arm_pass_settle", {})[tag] = settle
                if not settle["reached_target"]:
                    result["incidents"].append({
                        "pass": p_idx, "arm": arm.name, "kind": "thermal_settle_timeout",
                        "detail": f"entry temp {settle['entry_temp_c']} C did not reach "
                                  f"{settle['target_c']:.0f} C within {settle_max_wait_s}s"})
                print(f"  settled: {settle['start_temp_c']}C -> {settle['entry_temp_c']}C "
                      f"in {settle['waited_s']}s (clock {settle['entry_sm_clock_mhz']} MHz)",
                      flush=True)
                # The card was gated on temperature and clock; the host was gated on nothing. A
                # compiler running on this machine during Phase M pass 2 could only be found
                # afterwards by comparing object-file timestamps with server logs by hand.
                load = T.host_load()
                result.setdefault("arm_pass_host_load", {})[tag] = load
                if load["contended"]:
                    names = ", ".join(f"{c['comm']} {c['pcpu']:.0f}%" for c in load["competing"])
                    result["incidents"].append({
                        "pass": p_idx, "arm": arm.name, "kind": "host_contended",
                        "detail": f"{load['competing_pct']:.0f}% of CPU is not this run: {names}"})
                    print(f"  !! host contended at arm entry: {names}", flush=True)
                h = S.start(binary, arm.model or model, arm.extra_args, port=port,
                            log_path=log_path, common_args=common_args, gpu_index=gpu_index)
            except S.ServerError as e:
                result["incidents"].append(
                    {"pass": p_idx, "arm": arm.name, "kind": "server_start_failed",
                     "detail": str(e)[:4000]})
                print(f"  !! server start failed: {str(e)[:300]}", flush=True)
                continue

            drafter_evidence = None
            try:
                if arm.expects_drafter:
                    drafter_evidence = S.assert_drafter_loaded(h, arm.name, arm.extra_args)
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
                # An n-gram method drafts from an n-gram cache built out of the context, so it
                # can legitimately draft nothing: on a 32-token warmup none of the three fire at
                # all, and at 400 tokens ngram-cache drafts 95 while ngram-mod and ngram-map-k
                # still draft none. Asserting turns "this method does not fire on this workload"
                # into "the arm is broken", skips it, and destroys the result the arm exists to
                # produce. Whether it drafted is recorded either way and analysed as data.
                _is_ngram = any("ngram" in a for a in (arm.extra_args or ()))
                if arm.expects_drafter and not _is_ngram:
                    n_drafted = S.assert_drafting_observed(wr, arm.name)
                elif _is_ngram:
                    n_drafted = (wr.get("t_draft_n") or 0)
                    print(f"  n-gram arm: warmup drafted {n_drafted} "
                          f"(zero is a valid outcome for these and is recorded, not failed)",
                          flush=True)
                    print(f"  drafting confirmed: t_draft_n={n_drafted} on warmup", flush=True)

                for _ord, pr in enumerate(pass_prompts):
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
                    # EVERY absolute-energy field, not just the one that existed when this was
                    # written. The window covers `reps_here` calibration requests and the figure
                    # subtracted below is one request's worth, so an unnormalised sibling is
                    # `reps_here` times too large. `energy_j_instant` and `energy_j_nvml` were
                    # added to PowerSampler.summary() later and this line was not revisited, so
                    # both sat in every result file at 8x. Nothing published used them, because
                    # `decode_energy` reads `energy_j`; the first analysis that did use them
                    # produced a decode-energy saving of 43.7 % against a true 36.3 %.
                    # The percentage fields are NOT touched: summary() computes them from the raw
                    # integrals before this runs, and a ratio of two quantities scaled alike is
                    # already right.
                    for _k in PREFILL_ABSOLUTE_ENERGY_FIELDS:
                        if prefill_power.get(_k) is not None:
                            prefill_power[_k] /= reps_here
                    prefill_power["reps"] = reps_here

                    # THE ROLL, and it goes on both sides of the WINDOW rather than both
                    # sides of the request. `power.draw` smooths and lags; smoothing is
                    # linear and preserves the integral under it, a lag of d seconds loses
                    # d * (p_end - p_start) however the trace moves in between. The point of
                    # the roll is to make p_end - p_start zero by putting both ends in the
                    # same idle steady state.
                    #
                    # This sleep is OUTSIDE the `with`, and that is the whole of what makes
                    # it work. The sampler takes its first snapshot at __enter__, so a sleep
                    # placed inside cannot affect what that snapshot sees: it still catches
                    # the card falling back from the prefill calibration. A dry run at
                    # roll 1.5 with the sleep inside opened at 357 W and closed at 131 W and
                    # returned an offset of -97.7 J -- larger than the unrolled one and the
                    # other way up, because the intervention had made the two ends MORE
                    # different rather than less. The trace showed it directly: the averaged
                    # field needed about eight samples to follow the step down.
                    if power_roll_s:
                        time.sleep(power_roll_s)
                    with T.sampling(index=gpu_index, interval_s=power_interval_s) as ps:
                        r = S.chat(port, pr.system, _with_filler(filler_text, pr.user),
                                   max_tokens=max_tokens, temperature=arm.temperature,
                                   think=pr.think, cache_prompt=cache_prompt)
                        # And inside at the end, because the last sample has to be taken
                        # after the averaged field has caught up with the card going quiet,
                        # which cannot happen once the window is closed.
                        if power_roll_s:
                            time.sleep(power_roll_s)
                    power = ps.summary()
                    power_trace_rec = ps.trace() if power_trace else None

                    text = (r.get("reasoning_content") or "") + (r.get("content") or "")
                    predicted_n = int(r.get("t_predicted_n") or r.get("completion_tokens") or 0)
                    # llama.cpp's own figure, which is (predicted_n - 1) / t_gen_ms. The minus
                    # one is right: its source comments that "the first token is free, it comes
                    # from the logits of the last prompt batch", and t_gen_ms is timed from the
                    # end of prompt processing, so numerator and denominator cover the same
                    # tokens. Never seen absent on this build.
                    rate = r.get("t_predicted_per_second")
                    if rate is None and r.get("t_predicted_ms"):
                        # Match the server's definition rather than inventing one. Dividing
                        # predicted_n by t_predicted_ms would put N tokens over the time for
                        # N - 1 of them, overstating the rate by N/(N-1).
                        rate = ((predicted_n - 1) / (r["t_predicted_ms"] / 1000.0)
                                if predicted_n > 1 else None)

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
                        # The prefill calibration is a max_tokens=1 request, so subtracting it
                        # removes prompt processing AND the first output token. What is left
                        # covers tokens 2..N, so the numerator is N-1, matching the decode rate
                        # which already uses (predicted_n - 1) / t_predicted_ms. Using N here
                        # divided a decode-only denominator by an all-tokens numerator.
                        tok_per_j_decode = max(predicted_n - 1, 0) / decode_energy

                    # Check the invariant this matrix declared, not one of them twice. llama.cpp
                    # reports how many prompt tokens it served from cache, so both directions are
                    # verifiable and each phase has one that matters:
                    #
                    #   cache off, cache_n > 0    a later prompt was partly free, and that speed is
                    #                             attributed to the arm
                    #   cache on,  cache_n == 0   the shared prefix was re-prefilled, so this request
                    #                             paid a cost the rest of the arm did not, and at 96 K
                    #                             that is most of its wall time
                    #
                    # The second is the one a depth ladder needs. A prefill calibration precedes every
                    # measured request and warms the cache, so a miss means eviction rather than a cold
                    # start, which is a real risk at the rungs that sit near the capacity of the card.
                    #
                    # This tested only the first case and hard-coded "despite cache_prompt=False" into
                    # the message. phase_l sets CACHE_PROMPT = True deliberately, to avoid re-prefilling
                    # its filler once per request, so every one of its requests was reported as an
                    # incident against a condition it never claimed.
                    cache_n = int(r.get("t_cache_n") or 0)
                    if not cache_prompt and cache_n > 0:
                        result["incidents"].append({
                            "pass": p_idx, "arm": arm.name, "prompt": pr.tag,
                            "kind": "prompt_cache_hit",
                            "detail": f"t_cache_n={cache_n} despite cache_prompt=False; "
                                      f"this request was partly served from cache"})
                    elif cache_prompt and cache_n == 0:
                        result["incidents"].append({
                            "pass": p_idx, "arm": arm.name, "prompt": pr.tag,
                            "kind": "prompt_cache_miss",
                            "detail": f"t_cache_n=0 with cache_prompt=True on "
                                      f"{r.get('prompt_tokens')} prompt tokens; the shared prefix "
                                      f"was re-prefilled rather than reused"})

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
                        # Absent unless --power-trace asked for it, and `design` records
                        # which. A key that is sometimes missing and sometimes null is the
                        # shape that got read as False elsewhere in this harness, so the
                        # declaration lives in `design` where it is always present.
                        **({"power_trace": power_trace_rec} if power_trace_rec else {}),
                        "prefill_power": prefill_power,
                        "prefill_predicted_n": int(pf.get("t_predicted_n") or 0),
                        "decode_energy_j": decode_energy,
                        "tok_per_joule_e2e": tok_per_j_e2e,
                        "tok_per_joule_decode": tok_per_j_decode,
                        "degeneracy": asdict(deg),
                        "text_len": len(text),
                        # where in this arm-pass the request ran. Fixed order makes this a
                        # constant per prompt and therefore useless; under a permutation it is
                        # what lets position be adjusted for rather than assumed away.
                        "ordinal": _ord,
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
                    if arm.name in baseline_names:
                        baseline_text[(arm.name, pr.tag, p_idx)] = text

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
        _attach_baseline_comparisons(result, baseline_text, _bmap, baseline_names, p_idx,
                                     [a.name for a in arms if a.name in baseline_names])
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


def _attach_baseline_comparisons(result: dict, baseline_text: dict, bmap: dict,
                                 baseline_names: set, p_idx: int,
                                 baseline_order: list | None = None) -> None:
    """Attach divergence + relative-degeneracy to every non-baseline record of one pass.

    Deferred until the pass ends because arm order is rotated, so the baseline arm may run
    after some treatment arms within a pass. Each arm is referred to the baseline built from its
    own tree, so a branch difference is never charged to the method.
    """
    for rec in result["records"]:
        if rec["pass"] != p_idx or rec["arm"] in baseline_names:
            continue
        if "rel_degeneracy" in rec:
            continue
        bname = bmap.get(rec["arm"])
        ref = baseline_text.get((bname, rec["prompt"], p_idx)) if bname else None
        if ref is None:
            rec["baseline_comparison_unavailable"] = True
            rec["baseline_comparison_wanted"] = bname
            continue
        rec["compared_against"] = bname
        text = rec.get("text", "")
        if rec.get("temperature") == 0.0:
            rec["divergence"] = asdict(quality.compare_outputs(ref, text))
        rec["rel_degeneracy"] = asdict(quality.assess_against_baseline(ref, text))

    # Every baseline is its own reference in `bmap`, so the loop above skips all of them and the
    # cross-tree control went away when `bmap` was introduced. It used to exist: at a 400-token
    # cap `baseline@pr27342` carried a divergence against `baseline@master` on 125 of 125 records,
    # all showing no divergence inside the window, which is the evidence that the branch matches
    # speculation off. The comment on `bmap` above says the next pair of trees need not agree --
    # that is the reason to keep measuring it, not to stop. It is recorded under its own key
    # because a control read as a method effect is how the one arm in the study with no fork
    # position came to be printed as a group of a fork-position partition.
    order = [b for b in (baseline_order or sorted(baseline_names)) if b in baseline_names]
    if len(order) < 2:
        return
    primary = order[0]
    for rec in result["records"]:
        if (rec["pass"] != p_idx or rec["arm"] not in baseline_names
                or rec["arm"] == primary or "tree_divergence" in rec):
            continue
        ref = baseline_text.get((primary, rec["prompt"], p_idx))
        if ref is None:
            rec["tree_comparison_unavailable"] = True
            continue
        rec["tree_compared_against"] = primary
        if rec.get("temperature") == 0.0:
            rec["tree_divergence"] = asdict(quality.compare_outputs(ref, rec.get("text", "")))


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
    ap.add_argument("--latin-arms", action="store_true",
                    help="run len(arms) passes so the arm rotation closes and every arm visits "
                         "every order position exactly once. Overrides --passes.")
    ap.add_argument("--shuffle-prompts", action="store_true",
                    help="permute the prompt order per pass, identically across arms within a "
                         "pass. Off by default: the fixed order runs classes in blocks, which "
                         "confounds class with server age, but changing it mid-study would change "
                         "the design between phases.")
    ap.add_argument("--prompt-seed", type=int, default=20260825)
    ap.add_argument("--prompts-per-class", type=int, default=0,
                    help="dry-run aid: keep only the first N prompts of each class. "
                         "0 = use the full frozen set. Any value other than 0 is recorded in "
                         "the result so a reduced run can never be mistaken for a full one.")
    ap.add_argument("--power-interval", type=float, default=0.10,
                    help="seconds the power sampler waits between nvidia-smi queries "
                         "(default 0.10, which every phase before E3 used). It was a "
                         "function parameter that nothing passed, so the rate was fixed "
                         "at 10 Hz in practice. It is a knob because it is the one that "
                         "separates a physical energy difference from an artefact of "
                         "integrating two differently-smoothed fields over the same "
                         "samples: `power.draw` is a one-second rolling average whatever "
                         "the query rate, `power.draw.instant` is not, so a coarse grid "
                         "aliases one and not the other. Recorded in the result.")
    ap.add_argument("--power-roll", type=float, default=0.0,
                    help="seconds of idle held INSIDE the sampling window on either side "
                         "of the measured request (default 0.0, which every phase before "
                         "E4 used). It is the intervention that separates a window-EDGE "
                         "offset from a per-second one: `power.draw` lags as well as "
                         "smooths, and a lag costs the integral d * (p_end - p_start) "
                         "however the trace moves in between, so putting both ends in the "
                         "same idle state should drive the offset to zero. A per-second "
                         "leak would instead grow with the longer window. A rolled window "
                         "is NOT comparable to an unrolled one -- its energy includes the "
                         "roll -- so it is recorded in `design` and the cross-file "
                         "instrument sweep refuses any file that declares it.")
    ap.add_argument("--power-trace", action="store_true",
                    help="store both power series per record, timestamped from the "
                         "window's start. Roughly seven times the size of the rest of a "
                         "record, and the only way to say WHERE in the window the two "
                         "integrals separate: one total is produced by a step at the "
                         "start, a step at the end, or a drift throughout, and those are "
                         "three different mechanisms.")
    args = ap.parse_args()
    if not 0.0 <= args.power_roll <= 30.0:
        raise SystemExit(f"--power-roll {args.power_roll} is outside [0.0, 30.0]; a roll "
                         f"longer than that spends more of the run idle than measuring, "
                         f"and the thermal settle gate is the thing that owns the card's "
                         f"entry state, not this")
    if not 0.005 <= args.power_interval <= 5.0:
        # `raise SystemExit`, not `sys.exit`: `main()` imports sys further down,
        # which makes the name local to the whole function, so referring to it
        # above that line is an UnboundLocalError rather than an exit. Caught by
        # running the validation instead of trusting it.
        raise SystemExit(f"--power-interval {args.power_interval} is outside "
                         f"[0.005, 5.0]; below that the sampler spends the run "
                         f"spawning subprocesses and above it a 6-second "
                         f"generation gets one sample")

    import importlib, sys
    sys.path.insert(0, str(HERE / "matrices"))
    mod = importlib.import_module(args.matrix)

    # A matrix whose conditions are calibrated for one specific card says so here rather than at
    # import time. phase_r and phase_r2 hard-code this 3090's 420 W default and 9751 MHz stock
    # memory clock, and used to call their device check as a module-level statement -- so the
    # module could not be imported at all without a GPU, and the CPU-only CI job could not read
    # the matrix definitions it was written to check. The check is the same and still runs before
    # anything is measured; only its position moved, from "when this file is read" to "when this
    # matrix is about to run".
    precheck = getattr(mod, "PRECHECK", None)
    if callable(precheck):
        precheck(args.gpu)

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

    matrix_provenance = matrix_provenance_snapshot(mod, args.matrix, sys.argv[1:])

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
        shuffle_prompts=args.shuffle_prompts,
        prompt_seed=args.prompt_seed,
        latin_arms=args.latin_arms,
        baseline_map=getattr(mod, "BASELINE_MAP", None),
        matrix_provenance=matrix_provenance,
        required_vram_gb=getattr(mod, "REQUIRES_VRAM_GB", 0.0),
        context_filler_tokens=getattr(mod, "CONTEXT_FILLER_TOKENS", 0),
        cache_prompt=getattr(mod, "CACHE_PROMPT", False),
        settle_temp_c=(None if args.settle_floor else 60.0),
        power_interval_s=args.power_interval,
        power_roll_s=args.power_roll,
        power_trace=args.power_trace,
    )


if __name__ == "__main__":
    main()
