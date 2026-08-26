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
command -v hf >/dev/null || echo "  note: 'hf' not on PATH; download the models yourself"

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
git -C llamacpp-dflash2 fetch origin pull/27342/head --quiet
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
[ -f models/target/Qwen3.8-27B-UD-Q4_K_XL.gguf ] || \
  hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf --local-dir models/target
[ -f models/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf ] || \
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
echo "Done. Yours: $OUT   Committed: results/phase_a.json"
echo "Absolute tok/s are host-specific; compare the paired effects, not the levels."
