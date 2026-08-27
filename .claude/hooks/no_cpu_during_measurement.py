#!/usr/bin/env python3
"""Refuse CPU-heavy shell commands while a GPU measurement holds the lock.

Written because detection was not prevention. `telemetry.host_load` samples the host once per
arm-pass and records what it finds as a `host_contended` incident, and `audit_results.py` then
marks the whole file FAIL. That machinery worked perfectly and caught four real events -- three
of them mine (`sha256sum` on a 17 GB model at 57 %, `gh` at 43 %, `git log --since` at 162 %) and
one from another session's mutation suite at 585 %. Every one was noticed after the run it spoiled.

Two things this guard has to get right, and the second is the one that bites:

  * It must FAIL OPEN. A hook that raises, or that cannot parse its input, or that meets a lock
    file it does not understand, must let the command through. Blocking every shell command
    because the guard itself is broken is a worse outcome than any contaminated benchmark.

  * A command does not have to be slow to poison the record. `ps` reports `pcpu` as an average
    over the process's whole lifetime, so a 0.2 s process that used one core reads as ~100 % and
    is recorded as contention whether or not it displaced any real work. `nvidia-smi` is on the
    deny list for exactly that reason: the phase B incident it caused was probably harmless to
    the measurement and still made the file unusable under this repository's own rule.

Unmatched commands are allowed with a reminder rather than denied. An allow-list would be wrong
several times an hour and I would learn to work around it, which is how a guard stops guarding.
"""
import json
import os
import re
import sys
import time

# The GPU is the shared resource, so every repository that can hold its lock is checked, not just
# the one this session happens to be sitting in. The shell's cwd is often the other one.
LOCK_DIRS = (
    "/home/thc1006/dev/qwen3.8-speculative-decoding-rtx3090",
    "/home/thc1006/dev/qwen3.6-speculative-decoding-rtx3090",
)
LOCK_NAME = ".gpu-in-use.lock"
OVERRIDE = "MEASUREMENT_OVERRIDE=1"

# Anchored on what actually happened, not on a guess about what might be expensive.
HEAVY = [
    (r"\b(sha256sum|sha1sum|md5sum|shasum|b3sum|cksum)\b", "hashing a file reads it end to end"),
    # `gh` has to be in command position. `\bgh\b` matched the directory in `~/.config/gh/`, which
    # denied a peer session's `curl` for mentioning a config path. A command is preceded by a
    # boundary that is not a path character and is followed by whitespace.
    (r"(?<![\w./~-])gh(?=\s)(?!\s*[-=])",
     "the GitHub CLI is a Go binary that spends real CPU on startup and JSON"),
    (r"\bnvidia-smi\b", "short, but it reads as ~100 % pcpu and lands in the incident record"),
    (r"\bgit\s+(grep|blame|gc|fsck|repack|clone|bisect)\b", "git walks the whole object store"),
    # `git log -1` reads one commit; `git log -S` reads every blob in the history. The first
    # version of this line denied both, and the first thing it denied was the commit that would
    # have installed it -- blocking `git log --oneline -1` is precisely how a guard teaches you
    # to route around it. It then denied the edit that fixes it, because the deny list matches
    # the text of a command and cannot tell running a pattern from writing one into a file.
    # That is a real limit of this design: use Read/Edit for those, not the override.
    (r"\bgit\s+log\b[^|;&]*?\s(-[SG]\b|--(since|until|grep|follow|all)\b)", "a git history search"),
    (r"\b(python3?\s+-m\s+)?(unittest|pytest|nose2?)\b", "a test suite"),
    # These three require the script to be INVOKED, not merely named. The first version matched
    # any command containing the path, so `grep -n coverage harness/coverage_sim.py` was denied
    # as though it were running the analyser, and `cat scripts/verify_everything.sh` with it.
    # Reading a file is not what costs CPU; a guard that cannot tell the two apart is one you
    # start overriding out of habit, which is worse than not having it.
    # `\b` is too weak a boundary for an interpreter name, because filenames end in the same
    # letters. `git add reproduce_phase_a.sh scripts/verify_everything.sh` matched: `\bsh` found
    # the tail of `_a.sh`, and the rest of the pattern did the work. A lookbehind that also
    # excludes `.` and `-` is what distinguishes the command `sh` from the suffix `.sh`.
    (r"((?<![\w.-])(bash|sh|source)\s+\S*|\./\S*)verify_everything\.sh\b",
     "it says so itself: CPU-heavy, do not run during a measurement"),
    # The path has to sit where an interpreter would take it as the script to run: straight after
    # `python3` and any flags. Allowing it anywhere after `python3` still denied
    # `python3 -c "...open('harness/coverage_sim.py')..."`, which only reads the file. Matching a
    # command string cannot recover intent, so the precision has to come from position.
    (r"\bpython3?\s+(-[A-Za-z]\S*\s+)*harness/(audit_results|analyze|analyze_depth|"
     r"analyze_cross_device|cost_model|mechanism_b|coverage_sim|width_groups|divergence_report|"
     r"truncation_audit|anchor_verdict|warp_intervention|completeness|ladder_trend|cross_rung|"
     r"quality|compare_reproduction|render_evidence)\.py", "an analyser"),
    (r"\bpython3?\s+(-[A-Za-z]\S*\s+)*(tests/(data_)?mutate|analysis/(verify_claims|"
     r"matrix_report|rederive_from_logs|check_data_integrity|past_threshold_fit|plot_\w+))\.py",
     "the other repository's suite"),
    (r"\bcompileall\b|\bpyflakes\b|\bshellcheck\b|\bmypy\b|\bruff\b", "a whole-tree linter"),
    (r"\b(cmake|ninja|nvcc|cargo|gcc|g\+\+|clang\+?\+?)\b|\bmake\s+-j", "a build"),
    (r"\bpip3?\s+install\b|\bnpm\s+(i|install|ci)\b|\bhf\s+download\b", "a download and unpack"),
    (r"\b(tar|zstd|gzip|bzip2|xz|zip|unzip)\b", "compression"),
    (r"\bfind\s+/(?!home/thc1006/dev/\S+\s)", "a filesystem-wide walk"),
    (r"\b(grep|rg|ag)\b[^|]*\s-\w*r", "a recursive search over the tree"),
    (r"\bdu\b[^|]*\s-\w*s|\bdf\s+-h\s+/\s*$", "a disk walk"),
    (r"\bffmpeg\b|\bconvert\b|\bmagick\b", "media processing"),
]

CHEAP = re.compile(
    r"^\s*(cat|head|tail|sed\s+-n|wc|ls|stat|echo|printf|date|pwd|true|jq|cut|sort|uniq|"
    r"basename|dirname|readlink|realpath|git\s+(status|diff|show|rev-parse|config|add|commit))\b")


def lock_state():
    """(path, info) for the first live lock found, or (None, None). Never raises."""
    for d in LOCK_DIRS:
        p = os.path.join(d, LOCK_NAME)
        try:
            with open(p) as fh:
                body = fh.read(4096)
        except Exception:
            continue
        info = {"path": p, "body": " ".join(body.split())[:200], "pid": None, "since": None}
        m = re.search(r"pid=(\d+)", body)
        if m:
            info["pid"] = int(m.group(1))
        m = re.search(r"since=(\S+)", body)
        if m:
            info["since"] = m.group(1)
        # A lock whose process is gone is not a measurement, it is litter from a crash. Saying so
        # is more useful than either blocking on it or ignoring it silently.
        if info["pid"] is not None and not os.path.isdir(f"/proc/{info['pid']}"):
            info["stale"] = True
        return p, info
    # No lock file, so look for the process anyway. `bench.py` acquires once at the top and
    # releases once at the bottom -- checked, one call site each -- so the lock does not blink
    # between arm-passes, and a peer session that saw it absent was seeing the ninety minutes
    # between two runs. But the file is written a moment after the process starts, and a crashed
    # run can leave the process without it. The process is the thing that must not be disturbed;
    # the file is only how it announces itself.
    running = bench_process()
    if running:
        where = "a running bench.py, with no lock file"
        return where, {"path": where, "pid": running[0], "since": None, "body": running[1]}
    return None, None


def bench_process():
    """(pid, cmdline) of a running benchmark or its server, or None. Never raises.

    `/proc/<pid>/cmdline` is NUL-separated, which is the real argv. An earlier version joined it
    on spaces and matched a regex against the result -- throwing away the argument boundaries and
    then guessing them back. A peer session working on the sibling repository hit the same problem
    from the other side, with a plain `"bench.py" in cmdline`, and its fix is the better one: split
    on NUL and compare basenames at the positions where an interpreter would take a script.
    `vim harness/bench.py`, `grep -rn bench.py .` and a shell command that merely mentions the path
    all miss; the two things that actually hold the GPU both hit.
    """
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                argv = [a for a in fh.read().decode("utf-8", "replace").split("\0") if a]
        except Exception:
            continue
        if not argv:
            continue
        exe = os.path.basename(argv[0])
        hit = exe in ("llama-server", "llama-bench") or (
            exe.startswith("python")
            and any(os.path.basename(a) == "bench.py" for a in argv[1:4]))
        if hit:
            return int(pid), " ".join(argv)[:90]
    return None


HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1.*?^\s*\2\s*$", re.S | re.M)


MESSAGE_ARG = re.compile(r"(?<![\w-])(-m|--message)(=|\s+)('[^']*'|\"[^\"]*\"|\S+)")


def _command_only(command):
    """The command with the spans that are data rather than command removed.

    Two of them, both found by being denied.

      * A heredoc body. `cat >> notes.md <<'EOF' ... EOF` was denied for containing the word
        `sha256sum`, in a sentence about an old contamination incident.
      * The argument of `-m` / `--message`. `git commit -m "gh matched a config path"` was denied
        for the word `gh` in its own prose. A commit message is data by definition.

    Quoted spans in general are NOT stripped, and that is deliberate: `bash -c "sha256sum big"`
    really does run the thing inside the quotes. The guard reads a command string and cannot
    recover intent, so it is told about the spans where the answer is unambiguous and left
    conservative everywhere else.
    """
    return MESSAGE_ARG.sub(r"\1 MSG", HEREDOC.sub("<<HEREDOC", command))


VERSION_ONLY = re.compile(r"^\s*\S+(\s+\S+)?\s+(--version|-V|-dumpversion|--help)\s*$")


def _is_version_query(text):
    """True when every segment of the command only asks something to print its version.

    `gcc --version` costs nothing and tells a reproduction record which compiler built the tree,
    but `gcc` is on the deny list because a build is expensive. Recording a toolchain should not
    require the override.
    """
    parts = [p for p in re.split(r"&&|\|\||;|\|", text) if p.strip()]
    return bool(parts) and all(VERSION_ONLY.match(p) for p in parts)


def decide(command, info):
    text = _command_only(command)
    if _is_version_query(text):
        return "allow", ""
    for pattern, why in HEAVY:
        if re.search(pattern, text):
            return "deny", why
    if CHEAP.match(text):
        return "allow", ""
    return "warn", ""


def emit(decision, reason):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
        "permissionDecisionReason": reason,
    }}))


def main():
    raw = sys.stdin.read()
    data = json.loads(raw)
    if data.get("tool_name") != "Bash":
        return 0
    command = (data.get("tool_input") or {}).get("command") or ""
    if OVERRIDE in command:
        return 0
    path, info = lock_state()
    if not path:
        return 0
    if info.get("stale"):
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext":
                f"{path} exists but pid {info['pid']} is gone. That lock is litter from a crashed "
                f"run, not a measurement. Nothing is being protected; remove it before relying on "
                f"any guard here.",
        }}))
        return 0

    decision, why = decide(command, info)
    held = f"pid {info['pid']}" + (f", since {info['since']}" if info["since"] else "")
    if decision == "deny":
        emit("deny", (
            f"A GPU measurement holds {path} ({held}). This command is {why}, and the harness "
            f"samples the host once per arm-pass and records what it finds as a host_contended "
            f"incident -- which makes the whole result file FAIL under this repository's own audit "
            f"rule, whether or not the measurement itself moved. Wait for the run, or prepend "
            f"{OVERRIDE} to say you accept the incident. Lock says: {info['body']}"))
        return 0
    if decision == "warn":
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext":
                f"A GPU measurement holds {path} ({held}). This command is not on the heavy list, "
                f"so it is allowed, but anything that runs longer than a moment or uses a whole "
                f"core can still be recorded as contention.",
        }}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:                                  # noqa: BLE001
        # Fail open, loudly enough to be fixed. Never block on the guard's own failure.
        print(f"no_cpu_during_measurement hook failed open: {exc!r}", file=sys.stderr)
        sys.exit(0)
