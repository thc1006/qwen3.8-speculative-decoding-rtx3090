#!/usr/bin/env bash
# Reproduce Phase A end to end, on a machine that has none of it yet.
#
# Everything version-specific is read from repro/phase_a.lock.json rather than written here, so
# this script and the record of what was measured cannot drift apart. The checks below are the
# ones whose absence has actually cost this study a run: a short commit that resolved to the
# wrong object, a checksum manifest covering models this phase never downloads, a card that was
# not at stock clocks, and an output path that overwrote the artifact it was meant to compare
# against.
set -euo pipefail
cd "$(dirname "$0")/.."
LOCK=repro/phase_a.lock.json
die() { echo "FAIL: $*" >&2; exit 1; }
val() { python3 -c "import json,sys;print(json.load(open('$LOCK'))$1)"; }

echo "== 1. tools"
for t in git cmake ninja python3 sha256sum nvidia-smi; do
  command -v "$t" >/dev/null || die "$t not on PATH"
done
CUDA_VER=$(val "['cuda']")
NVCC=/usr/local/cuda-$CUDA_VER/bin/nvcc
[ -x "$NVCC" ] || die "$NVCC not found; the lock file pins CUDA $CUDA_VER"
# `hf` is optional only when both pinned model files are already on disk. This printed a note and
# then called `hf download` unconditionally three steps later, so a machine without it failed at
# step 5 with a command-not-found instead of at step 1 with a reason.
TARGET_GGUF=models/target/Qwen3.8-27B-UD-Q4_K_XL.gguf
DRAFTER_GGUF=models/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
if [ ! -f "$TARGET_GGUF" ] || [ ! -f "$DRAFTER_GGUF" ]; then
  command -v hf >/dev/null || die "'hf' is required: a pinned model file is missing and step 5 downloads it"
fi

echo "== 1b. this repository's own version, and the toolchain"
# A lock file cannot name its own commit, but it can name a tag created afterwards on the commit
# that carries it. Without this the script pins llama.cpp, CUDA, the models and the card, and
# leaves the harness that drives all of them free to be any later version of itself.
WANT_TAG=$(val "['repo_tag']")
# Compare COMMITS, not tag names, and stop rather than warn.
#
# `git describe --tags --exact-match` was here and it is wrong twice over. It answers "what is one
# name for this commit", and two tags point at the commit phase-a-v1 marks -- phase-a-v1 and
# v1.0.0 -- so which name comes back is not something the caller controls. And it only printed a
# warning, so `./scripts/reproduce_phase_a.sh` on a later master pinned llama.cpp, CUDA, the models
# and the card, and then ran them through whatever the harness had since become. That is not the
# experiment the lock file describes, and it looked like it was.
WANT_COMMIT=$(git rev-parse "${WANT_TAG}^{commit}" 2>/dev/null || true)
HAVE_COMMIT=$(git rev-parse HEAD)
if [ -z "$WANT_COMMIT" ]; then
  die "the lock names tag '$WANT_TAG' and this clone has no such tag; run: git fetch --tags"
fi
if [ "$HAVE_COMMIT" = "$WANT_COMMIT" ]; then
  echo "  repository at $WANT_TAG ($WANT_COMMIT)"
elif [ "${ALLOW_HARNESS_DRIFT:-0}" = "1" ]; then
  echo "  WARNING: ALLOW_HARNESS_DRIFT=1"
  echo "           HEAD is $HAVE_COMMIT, the lock was written for $WANT_TAG ($WANT_COMMIT)"
  echo "           llama.cpp, CUDA, the models and the card are pinned; the HARNESS is not."
  echo "           This is an independent replication with a later analysis path, not an exact rerun."
else
  echo "FAIL: exact reproduction needs the harness the measurement ran on." >&2
  echo "      lock names $WANT_TAG -> $WANT_COMMIT" >&2
  echo "      HEAD is    $HAVE_COMMIT" >&2
  echo "" >&2
  echo "  git fetch --tags && git switch --detach $WANT_TAG && git tag -v $WANT_TAG" >&2
  echo "" >&2
  echo "      or set ALLOW_HARNESS_DRIFT=1 to run an independent replication instead, which pins" >&2
  echo "      everything except this repository and is a different claim." >&2
  exit 1
fi
python3 - "$LOCK" <<'PY'
import json, re, shutil, subprocess, sys
lock = json.load(open(sys.argv[1])).get("toolchain") or {}
def first(*cmd):
    if not shutil.which(cmd[0]):
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return (r.stdout or r.stderr).splitlines()[0].strip()
    except Exception:
        return None
def ver(s):
    m = re.search(r"(\d+\.\d+(?:\.\d+)?)", s or "")
    return m.group(1) if m else None
checks = [("cxx_compiler", ver(first("g++", "--version")), "g++"),
          ("cmake", ver(first("cmake", "--version")), "cmake")]
diffs = []
for key, have, name in checks:
    want = ver(lock.get(key, ""))
    if want and have and want != have:
        diffs.append(f"{name} {have} against {want} recorded")
    elif want and not have:
        diffs.append(f"{name} not on PATH; {want} recorded")
if diffs:
    print("  toolchain differs from the recorded build: " + "; ".join(diffs))
    print("  reported, not enforced: it bounds how far the two builds can be assumed identical")
else:
    print("  toolchain matches the recorded build")
PY

echo "== 2. card"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
WANT_ARCH=$(val "['cuda_arch']")
[ "$CC" = "$WANT_ARCH" ] || die "compute capability $CC, lock file pins $WANT_ARCH"
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$FREE" -ge 20000 ] || die "only ${FREE} MiB free on the card; this needs about 20 GB"

echo "== 3. trees at the pinned commits"
M=$(val "['llama_master_commit']"); D=$(val "['dflash2_commit']")
[ -d llamacpp-master ] || git clone https://github.com/ggml-org/llama.cpp llamacpp-master
git -C llamacpp-master fetch --all --quiet
git -C llamacpp-master checkout --detach "$M" --quiet
[ -d llamacpp-dflash2 ] || cp -r llamacpp-master llamacpp-dflash2
# The bundle first, the live PR ref only as a fallback. `pull/27342/head` moves: that PR's head has
# already advanced past the measured commit, and a force-push or a deleted branch makes the object
# unreachable, at which point the lock names a commit nobody can fetch. The bundle carries the ten
# commits between the pinned master and the measured one, 22 KB, and its prerequisite is an
# ancestor of the pinned master, so a clone at that commit can always complete it.
BUNDLE=$(val "['dflash2_archive']['bundle']")
if [ -f "$BUNDLE" ] && git -C llamacpp-dflash2 bundle verify "../$BUNDLE" >/dev/null 2>&1; then
  git -C llamacpp-dflash2 fetch "../$BUNDLE" 'refs/tags/*:refs/tags/*' --quiet || true
fi
git -C llamacpp-dflash2 cat-file -e "$D^{commit}" 2>/dev/null || \
  git -C llamacpp-dflash2 fetch origin pull/27342/head --quiet
git -C llamacpp-dflash2 cat-file -e "$D^{commit}" 2>/dev/null || \
  die "$D is not in the bundle and not reachable from pull/27342/head any more"
git -C llamacpp-dflash2 checkout --detach "$D" --quiet
# Full 40-character comparison. A short prefix can resolve to a different object as a repository
# grows, and the point of pinning is that it cannot.
[ "$(git -C llamacpp-master  rev-parse HEAD)" = "$M" ] || die "master tree is not at $M"
[ "$(git -C llamacpp-dflash2 rev-parse HEAD)" = "$D" ] || die "dflash2 tree is not at $D"

echo "== 4. build, identical flags on both trees"
for t in llamacpp-master llamacpp-dflash2; do
  CUDACXX="$NVCC" cmake -B "$t/build" -S "$t" -GNinja \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$WANT_ARCH" -DGGML_CCACHE=ON \
    -DCMAKE_BUILD_TYPE="$(val "['cmake_build_type']")" >/dev/null
  cmake --build "$t/build" -j --target llama-server >/dev/null
  [ -x "$t/build/bin/llama-server" ] || die "$t did not produce llama-server"
done

echo "== 5. models, and only this phase's"
mkdir -p models/target models/dflash2
[ -f "$TARGET_GGUF" ] || \
  hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf --local-dir models/target
[ -f "$DRAFTER_GGUF" ] || \
  hf download z-lab/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q4_K_M.gguf --local-dir models/dflash2
sha256sum -c "$(val "['checksums']")" || die "model checksums do not match the run's"

echo "== 6. harness tests"
python3 harness/test_harness.py >/dev/null || die "the harness's own tests do not pass"

echo "== 7. run, without overwriting the committed artifact"
mkdir -p results/reproductions
OUT="results/reproductions/phase_a_$(hostname)_$(date -u +%Y%m%dT%H%M%SZ).json"
[ -e "$OUT" ] && die "$OUT already exists"
python3 harness/bench.py --matrix phase_a \
  --passes "$(val "['passes']")" --max-tokens "$(val "['max_tokens']")" --out "$OUT"

echo "== 8. shape"
N=$(python3 -c "import json;print(len(json.load(open('$OUT'))['records']))")
WANT=$(val "['expected_records']")
[ "$N" = "$WANT" ] || die "$N records, the lock file expects $WANT"
INC=$(python3 -c "import json;print(len(json.load(open('$OUT')).get('incidents') or []))")
[ "$INC" = "0" ] || echo "  WARNING: $INC incidents recorded; read them before comparing"

echo "== 9. analyse"
python3 harness/analyze.py "$OUT"

echo
echo "== 10. compare against the committed run, on the effects rather than the record count"
# Step 8 checks a shape. A rerun that landed on entirely different throughput would pass it, and
# "compare the paired effects, not the levels" used to be printed at the end as advice for a human
# to act on rather than as something this script did. Absolute tok/s are host-specific; the paired
# effects are what a reproduction has to carry, so they are what gets compared and what decides the
# exit code.
if python3 harness/compare_reproduction.py results/phase_a.json "$OUT"; then
  echo
  echo "Done. Yours: $OUT   Committed: results/phase_a.json"
else
  echo
  echo "Done, and the comparison did NOT establish a reproduction: $OUT"
  echo "Read the section above before quoting either run against the other. If the only objection"
  echo "is a recorded incident you have checked, re-run the comparison with --allow-incidents and"
  echo "say in writing what you checked."
  exit 1
fi
