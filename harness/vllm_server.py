"""vLLM server control for Phase V.

Deliberately separate from `server.py` rather than a refactor of it. That module is driving four
more llama.cpp phases while this is being written, and reworking it into an engine-agnostic
abstraction mid-run would put every remaining measurement behind an untested change. The pieces
that are genuinely engine-independent, prompts, telemetry, statistics and the degeneracy screen,
are imported by both and are not duplicated here.

Two things differ enough from llama.cpp to matter.

Acceptance is process-wide, not per-request. llama.cpp returns `draft_n` and `draft_n_accepted`
in each response's `timings`. vLLM publishes cumulative counters on `/metrics`, so a per-request
figure only exists as a difference across the request, and that difference is only attributable
to one request because the harness runs with one sequence in flight at a time. If that ever
changes, this becomes wrong silently, so `spec_delta` records the concurrency it assumed.

The metric names are discovered, not hard-coded. The documentation does not list them and they
have moved between versions; a hard-coded name that stops matching produces zeros, which read
as "speculation did nothing" rather than as a broken reader. So the counters are found by
pattern, all of them are kept, and a run that expected a drafter and found no counters is
refused rather than reported as a null result.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

# Counter families that describe speculative decoding, whatever they end up being called.
_SPEC_PATTERN = re.compile(r"^(vllm:)?spec_decode[a-z_]*(_total)?\b")


class VllmError(RuntimeError):
    pass


def _get(port: int, path: str, timeout_s: float = 10.0) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout_s) as r:
        return r.read().decode("utf-8", errors="replace")


def spec_counters(port: int) -> dict[str, float]:
    """Every speculative-decoding counter the server currently publishes.

    Returns an empty dict when the endpoint has none, which is the correct reading for a server
    started without `--speculative-config`. Distinguishing that from a broken reader is the
    caller's job, via `assert_speculation_observed`.
    """
    try:
        body = _get(port, "/metrics")
    except (urllib.error.URLError, OSError) as e:
        raise VllmError(f"/metrics unreachable on port {port}: {e}") from e
    out: dict[str, float] = {}
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, rest = line.partition(" ")
        base = name.split("{", 1)[0]
        if not _SPEC_PATTERN.match(base):
            continue
        try:
            out[name] = float(rest.strip())
        except ValueError:
            continue
    return out


def spec_delta(before: dict[str, float], after: dict[str, float]) -> dict:
    """Per-request acceptance, as the change in the cumulative counters.

    Valid only with a single sequence in flight; the assumption is recorded rather than left
    implicit, because it is the kind of thing that stops being true when someone raises
    concurrency for speed and does not think about the metrics.
    """
    delta = {k: after.get(k, 0.0) - v for k, v in before.items()}
    for k, v in after.items():
        delta.setdefault(k, v)

    def pick(*words):
        for k, v in delta.items():
            low = k.lower()
            if all(w in low for w in words):
                return v
        return None

    drafted = pick("draft", "token")
    accepted = pick("accepted", "token")
    drafts = pick("num_drafts") or pick("draft", "total")
    out = {"counters": delta, "assumes_single_sequence_in_flight": True,
           "drafted": drafted, "accepted": accepted, "drafts": drafts}
    if drafted:
        out["accept_rate"] = (accepted or 0.0) / drafted
        # Mean accepted tokens per verification step, the same quantity the llama.cpp side
        # derives, so the two engines can be compared on it.
        if drafts:
            out["mean_len"] = ((accepted or 0.0) + drafts) / drafts
    return out


def assert_speculation_observed(delta: dict, arm_name: str) -> None:
    """A `--speculative-config` can be accepted and then do nothing. Prove it did something."""
    if not delta.get("counters"):
        raise VllmError(
            f"arm {arm_name!r} expected a drafter but /metrics published no speculative "
            f"counters at all. Either the build does not export them under a name this reader "
            f"recognises, or --speculative-config was ignored. Refusing to record the arm as "
            f"speculative on no evidence.")
    if not delta.get("drafted"):
        raise VllmError(
            f"arm {arm_name!r} expected a drafter but drafted 0 tokens over the request. The "
            f"flag was accepted and the drafter never ran. Counters seen: "
            f"{sorted(delta['counters'])}")


def start(binary: str, model: str, port: int, extra_args: list[str], log_path: Path,
          *, gpu_index: int = 0, startup_timeout_s: float = 900.0) -> subprocess.Popen:
    """Launch `vllm serve` and wait until it answers.

    The timeout is long because vLLM compiles CUDA graphs on first start and a 27B model on this
    card takes minutes, not seconds. A short timeout here would look exactly like a crash.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [binary, "serve", model, "--port", str(port), *extra_args]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_index))
    with open(log_path, "w") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n")
        fh.flush()
        # start_new_session, never preexec_fn: this process runs a power-sampling thread and
        # running Python after fork in a threaded process is what crashed Phase A.
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
    t0 = time.time()
    while time.time() - t0 < startup_timeout_s:
        if proc.poll() is not None:
            raise VllmError(f"vllm exited with {proc.returncode} during startup; see {log_path}")
        try:
            _get(port, "/health", timeout_s=5.0)
            return proc
        except Exception:
            time.sleep(3.0)
    stop(proc)
    raise VllmError(f"vllm did not become healthy within {startup_timeout_s:.0f}s; see {log_path}")


def stop(proc: subprocess.Popen, timeout_s: float = 60.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def chat(port: int, system: str, user: str, *, max_tokens: int, temperature: float = 0.0,
         model: str = "", timeout_s: float = 900.0) -> dict:
    """One greedy completion through the OpenAI-compatible endpoint.

    Every sampler is pinned explicitly rather than left to the server's defaults, for the same
    reason the llama.cpp side does it: defaults differ between engines and between versions, and
    an unpinned sampler is a difference the result file cannot show.
    """
    body = {
        "model": model or "default",
        "messages": ([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "top_k": -1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": 20260824,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = json.loads(r.read())
    wall_ms = (time.time() - t0) * 1000.0
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return {
        "wall_ms": wall_ms,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "content": (choice.get("message") or {}).get("content") or "",
        "raw_usage": usage,
    }
