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


def _parse_metrics(body: str, keep) -> dict[str, float]:
    """Prometheus exposition text to {series name including labels: value}.

    The name is kept with its labels because that is what identifies a series: vLLM publishes
    one per (model_name, engine), and `spec_decode_num_accepted_tokens_per_pos` adds a third
    label so it publishes one per draft position. Collapsing them here would decide which one a
    caller meant.
    """
    out: dict[str, float] = {}
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, rest = line.partition(" ")
        if not keep(name.split("{", 1)[0]):
            continue
        try:
            out[name] = float(rest.strip())
        except ValueError:
            continue
    return out


def _canon(series_name: str) -> str:
    """A series name reduced to the metric it belongs to.

    Labels go, the `vllm:` prefix goes, and the suffix prometheus_client appends to a counter
    goes: a Counter declared `vllm:generation_tokens` is published as
    `vllm:generation_tokens_total{engine="0",model_name="..."}` plus a `_created` timestamp
    series. `_created` is deliberately NOT reduced to the same key -- it is a Unix time, and
    adding it to a token count would produce a number near 1.8e9 that still looks like a number.
    """
    base = series_name.split("{", 1)[0]
    if base.startswith("vllm:"):
        base = base[len("vllm:"):]
    if base.endswith("_total"):
        base = base[:-len("_total")]
    return base


def series_sum(counters: dict[str, float], metric: str) -> float:
    """Sum every label set of one metric, and only that metric.

    A lookup by bare metric name matches nothing at all, because every series carries labels. A
    lookup that scans for a substring matches too much: "accepted" and "token" both appear in
    `spec_decode_num_accepted_tokens_per_pos`, which is a different metric with one series per
    draft position, and which of the two a substring scan returns depends on the order the
    server happened to emit them in.
    """
    want = _canon(metric)
    return sum(v for k, v in counters.items() if _canon(k) == want)


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
    return _parse_metrics(body, lambda base: bool(_SPEC_PATTERN.match(base)))


def spec_delta(before: dict[str, float], after: dict[str, float]) -> dict:
    """Per-request acceptance, as the change in the cumulative counters.

    Valid only with a single sequence in flight; the assumption is recorded rather than left
    implicit, because it is the kind of thing that stops being true when someone raises
    concurrency for speed and does not think about the metrics.
    """
    delta = {k: after.get(k, 0.0) - v for k, v in before.items()}
    for k, v in after.items():
        delta.setdefault(k, v)

    def pick(metric):
        # Exact metric, summed over label sets. The substring scan this replaced returned the
        # first series whose name contained the words, so "accepted"+"token" could return a
        # bucket of spec_decode_num_accepted_tokens_per_pos, or a _created timestamp, depending
        # on emission order. vLLM 0.27.1 happens to emit the plain counters first
        # (vllm/v1/spec_decode/metrics.py:228-260 creates them before the per-position one), so
        # the old code was correct by accident of registry ordering rather than by construction.
        total = series_sum(delta, metric)
        return total if any(_canon(k) == _canon(metric) for k in delta) else None

    drafted = pick("vllm:spec_decode_num_draft_tokens")
    accepted = pick("vllm:spec_decode_num_accepted_tokens")
    drafts = pick("vllm:spec_decode_num_drafts")
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


# Confirmed against vLLM 0.27.1 on 2026-08-26 by harness/vllm_probe.py: these exist and are
# per-request histograms, so decode time is separable from prefill and from queueing. That
# matters because the obvious alternative -- completion_tokens over wall time -- silently
# includes prompt processing, and is exactly the error that made llama.cpp #27623 report a 25x
# decode collapse that its author withdrew on 2026-08-26 once he measured eval-only timings.
DECODE_TIME = "vllm:request_decode_time_seconds"
PREFILL_TIME = "vllm:request_prefill_time_seconds"
QUEUE_TIME = "vllm:request_queue_time_seconds"
GEN_TOKENS = "vllm:generation_tokens_total"


def timing_counters(port: int) -> dict[str, float]:
    """The request-timing series, so a decode rate can exclude prefill and queueing.

    Returned raw rather than reduced: `_sum` and `_count` are what a difference across one
    request needs, and which of them a caller wants depends on whether it is after a total or a
    mean. Reducing here would decide that for them and hide the counters that prove it.
    """
    try:
        body = _get(port, "/metrics")
    except (urllib.error.URLError, OSError) as e:
        raise VllmError(f"/metrics unreachable on port {port}: {e}") from e
    want = (DECODE_TIME, PREFILL_TIME, QUEUE_TIME, GEN_TOKENS)
    return _parse_metrics(body, lambda base: base.startswith(want))


def decode_rate(before: dict[str, float], after: dict[str, float]) -> dict:
    """tok/s over the decode phase alone, as llama.cpp's timings.predicted_per_second means it.

    Both engines then report the same quantity. Without this the vLLM side carries prefill
    inside its rate, which does NOT cancel when each engine is divided by its own baseline: a
    speculative arm decodes faster, so prefill is a larger share of its wall time and the
    speedup comes out too small.
    """
    def delta(prefix, suffix):
        # series_sum, not after[k]: every series is labelled, so a bare-name lookup returns 0.0
        # for both ends and the difference is a silent zero -- which reads as "the request
        # generated nothing" rather than as "this reader cannot find the counter".
        k = prefix + suffix
        return series_sum(after, k) - series_sum(before, k)

    dec = delta(DECODE_TIME, "_sum")
    pre = delta(PREFILL_TIME, "_sum")
    que = delta(QUEUE_TIME, "_sum")
    toks = delta(GEN_TOKENS, "")
    out = {"decode_s": dec, "prefill_s": pre, "queue_s": que, "generation_tokens": toks}
    if dec > 0 and toks > 0:
        out["decode_tok_s"] = toks / dec
    if (dec + pre) > 0:
        # How wrong a wall-clock rate would have been on this request, stated rather than left
        # for a reader to wonder about.
        out["prefill_share_of_inference"] = pre / (dec + pre)
    return out


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
