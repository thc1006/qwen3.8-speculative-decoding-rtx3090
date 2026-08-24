"""llama-server lifecycle with the guards this study depends on.

Three assertions are made before any number from a server is accepted:

1. The port was free before we started (no stale server can answer for us).
2. The process listening on the port is our own child (or its descendant).
3. If the arm declares a drafter, the server log shows the drafter actually loaded.

(3) exists because the predecessor repo shipped a headline-table row whose draft model never
attached at all -- a vocab mismatch meant the flag was silently ignored and the row was a
duplicate baseline. That was caught by eye. Here it is caught by the harness.
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
from dataclasses import dataclass, field
from pathlib import Path

from telemetry import assert_port_owned_by, gpu_snapshot, port_is_free


class ServerError(RuntimeError):
    pass


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    port: int
    log_path: Path
    ready_s: float
    gpu_at_start: dict[str, float] = field(default_factory=dict)

    def log_text(self) -> str:
        try:
            return self.log_path.read_text(errors="ignore")
        except OSError:
            return ""


# --------------------------------------------------------------------------- lifecycle

def start(
    binary: Path,
    model: Path,
    extra_args: list[str],
    *,
    port: int,
    log_path: Path,
    common_args: list[str],
    gpu_index: int = 0,
    timeout_s: float = 600.0,
) -> ServerHandle:
    if not port_is_free(port):
        raise ServerError(
            f"port {port} is already bound before we started. Refusing to run: a stale "
            "llama-server would answer our health check and we would measure it instead. "
            "Kill it and retry.")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
           *common_args, *extra_args]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    # Determinism aids: keep llama.cpp from silently varying thread counts run to run.
    env.setdefault("GGML_CUDA_NO_PINNED", "0")

    fh = open(log_path, "w")
    fh.write("CMD: " + " ".join(cmd) + "\n")
    fh.flush()
    # `start_new_session=True`, NOT `preexec_fn=os.setsid`. They do the same thing — put the
    # child in its own process group so the whole tree can be signalled — but preexec_fn runs
    # Python code in the child after fork, and that is explicitly unsafe in a process that has
    # threads. This harness runs a GPU power-sampling thread throughout, and the first full
    # 875-record run ended with `double free or corruption (out)` from glibc after roughly 7900
    # fork/exec cycles: all records were written, but the process died before attaching the last
    # pass's divergence comparisons and before releasing the run lock. start_new_session does
    # the setsid in C on the safe side of the fork.
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)

    t0 = time.perf_counter()
    url = f"http://127.0.0.1:{port}/health"
    last_err = ""
    while time.perf_counter() - t0 < timeout_s:
        if proc.poll() is not None:
            raise ServerError(
                f"llama-server exited with code {proc.returncode} during startup. "
                f"Log tail:\n{_tail(log_path)}")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    ready = time.perf_counter() - t0
                    assert_port_owned_by(port, proc.pid)
                    return ServerHandle(proc=proc, port=port, log_path=log_path,
                                        ready_s=ready, gpu_at_start=gpu_snapshot(gpu_index))
        except Exception as e:      # noqa: BLE001 - health probe, any failure means "not yet"
            last_err = repr(e)
        time.sleep(0.5)

    stop(proc)
    raise ServerError(f"server not ready in {timeout_s}s (last probe: {last_err})\n"
                      f"Log tail:\n{_tail(log_path)}")


def stop(proc: subprocess.Popen, *, port: int | None = None, grace_s: float = 25.0) -> None:
    """Terminate the process group and WAIT until the port is actually released.

    Returning before the port frees is how a sweep ends up measuring the previous arm.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=grace_s)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
        except Exception:
            pass
    if port is not None:
        deadline = time.perf_counter() + 30.0
        while time.perf_counter() < deadline:
            if port_is_free(port):
                return
            time.sleep(0.25)
        raise ServerError(f"port {port} still bound 30s after killing the server")


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="ignore").splitlines()[-n:])
    except OSError:
        return "(no log)"


# --------------------------------------------------------------------------- log assertions

# Evidence that llama.cpp ACTUALLY loaded a drafter, as opposed to us having merely passed a
# flag. Matching on our own command line was the first version of this check and it was useless:
# the log's first line is "CMD: ... --spec-type draft-mtp ...", so a regex for "spec-type"
# matched the request rather than the result and the guard passed unconditionally -- reproducing
# exactly the failure it exists to prevent.
_CMD_LINE_PREFIX = "CMD:"

# Anchored on llama.cpp's own function names rather than on guessed prose. An earlier version
# of this list guessed at wordings like "loading draft model" and missed the real line, which
# reads "common_speculative_init_result: creating MTP draft context against the target model" --
# a false negative that would have aborted every working speculative arm.
_DRAFT_LOAD_MARKERS = (
    re.compile(r"common_speculative_init_result", re.I),      # authoritative: spec ctx created
    re.compile(r"common_speculative_init", re.I),
    re.compile(r"draft\s+acceptance\s*=", re.I),             # appears once requests have run
    re.compile(r"llama_model_loader:\s*loaded meta data.*draft", re.I),
    re.compile(r"load(ing|ed)?\s+draft\s+model", re.I),
    re.compile(r"n_draft\s*=\s*\d+", re.I),
)

# Lines that prove the flag was REJECTED or silently ignored.
_DRAFT_REJECT_MARKERS = (
    re.compile(r"draft.*vocab.*mismatch", re.I),
    re.compile(r"vocab.*mismatch.*draft", re.I),
    re.compile(r"ignoring.*draft", re.I),
    re.compile(r"speculative.*not supported", re.I),
    re.compile(r"disabl(ing|ed).*speculative", re.I),
)


def assert_drafter_loaded(handle: ServerHandle, arm_name: str) -> str:
    """Return the log line proving a drafter loaded, or raise. Never accept an arm on faith.

    Our own `CMD:` line is excluded from the search: it records what we ASKED for, and the whole
    point of this check is to find out what llama.cpp actually DID.
    """
    lines = [ln for ln in handle.log_text().splitlines()
             if not ln.lstrip().startswith(_CMD_LINE_PREFIX)]
    body = "\n".join(lines)

    for pat in _DRAFT_REJECT_MARKERS:
        m = pat.search(body)
        if m:
            line = next((ln for ln in lines if pat.search(ln)), m.group(0))
            raise ServerError(
                f"arm '{arm_name}': the server log says the drafter was rejected or ignored: "
                f"{line.strip()!r}")

    for pat in _DRAFT_LOAD_MARKERS:
        for line in lines:
            if pat.search(line):
                return line.strip()

    raise ServerError(
        f"arm '{arm_name}' declares a drafter but the server log contains no evidence that one "
        f"loaded (our own CMD line is excluded from this search on purpose). The flag was very "
        f"likely accepted and ignored -- this is how the predecessor repo's 'draft-qwen3-0.6b' "
        f"row became a silent duplicate of baseline.\nLog tail:\n{_tail(handle.log_path, 60)}")


# Observed format (build 200 / c060ca9):
#   slot print_timing: id 0 | task 0 | draft acceptance = 0.83007 ( 127 accepted / 153 generated), mean len = 3.44
_ACCEPT_RE = re.compile(
    r"draft acceptance\s*(?:rate)?\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+"
    r"generated\s*\)(?:\s*,\s*mean len\s*=\s*([0-9.]+))?",
    re.I)


def parse_acceptance_from_log(handle: ServerHandle) -> list[dict[str, float]]:
    """Every per-request acceptance line llama.cpp printed, in order.

    `mean len` is the mean number of tokens emitted per target forward pass -- the quantity that
    actually determines speedup, and one that the per-request API timings do not expose.
    """
    out = []
    for rate, acc, gen, mlen in _ACCEPT_RE.findall(handle.log_text()):
        rec = {"accept_rate": float(rate), "accepted": int(acc), "generated": int(gen)}
        if mlen:
            rec["mean_draft_len"] = float(mlen)
        out.append(rec)
    return out


def assert_drafting_observed(response: dict, arm_name: str) -> int:
    """Behavioural proof that the drafter ran, from the request itself.

    Log archaeology tells us what llama.cpp printed; this tells us what it DID. An arm that
    declares a drafter must report a non-zero drafted-token count on a real generation. This is
    the check that cannot be defeated by a wording change upstream.
    """
    n = int(response.get("t_draft_n") or 0)
    if n <= 0:
        raise ServerError(
            f"arm '{arm_name}' declares a drafter, the server started, but the first real "
            f"generation reports t_draft_n={n}. No tokens were drafted: the speculative path "
            f"did not run. Timing fields present: "
            f"{sorted(k for k in response if k.startswith('t_'))}")
    return n


# --------------------------------------------------------------------------- requests

def chat(
    port: int,
    system: str,
    user: str,
    *,
    max_tokens: int,
    temperature: float = 0.0,
    seed: int = 20260824,
    think: bool = False,
    timeout_s: float = 600.0,
) -> dict:
    body = {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": False,
        # Qwen3.8 exposes reasoning control through the chat template.
        "chat_template_kwargs": {"enable_thinking": bool(think)},
        # MUST stay false. llama-server reuses the prompt prefix across requests by default;
        # prompts in this set share a system message within a class, so later prompts would be
        # served partly from cache and read faster for a reason unrelated to the arm. Worse,
        # prefix caching is known to INTERACT with speculative decoding rather than merely
        # offset it -- vLLM issue #38182 reports MTP dropping prefix-cache hit rate from ~92%
        # to ~71%, and a confound of exactly this shape is what forced the retraction of the
        # sibling repo's first published vLLM MTP result. Measuring with it on would repeat
        # that mistake.
        "cache_prompt": False,
    }
    if temperature == 0.0:
        # "temperature 0" is not by itself a guarantee of greedy decoding: llama-server carries
        # its own default sampler chain, and any repetition/presence penalty left active would
        # make the output depend on what was generated so far in a way that differs between a
        # speculative arm and the baseline. Every sampler is pinned explicitly so that "greedy"
        # means the same thing in every arm, and so the byte-level divergence check is testing
        # the speculative path rather than a sampler difference.
        body.update({
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "typical_p": 1.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "dry_multiplier": 0.0,
            "mirostat": 0,
        })

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = json.loads(r.read())
    wall_ms = (time.perf_counter() - t0) * 1000.0

    timings = data.get("timings") or {}
    usage = data.get("usage") or {}
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""

    out = {
        "wall_ms": wall_ms,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "content": content,
        "reasoning_content": reasoning,
    }
    # Copy every timing field llama.cpp reports rather than a hand-picked subset: field names
    # have changed across versions and a missing key must not silently become a zero.
    for k, v in timings.items():
        out[f"t_{k}"] = v
    return out
