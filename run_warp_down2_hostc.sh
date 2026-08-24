#!/usr/bin/env bash
# Rebuilds the forced-down direction so that it only changes what it means to change.
#
# The first forced-down build set the whole 1-4 row to two warps. Width 1 is in that row, and a
# drafter generates one token at a time, so ncols_dst 1 is what every speculative arm's drafter
# runs at. Editing that row perturbs every arm through its drafter no matter what the
# verification width is, which is why widths 5, 6 and 8 came out different from the control while
# their kernels were byte-identical SASS. It is also why the registered baseline control could not
# hold: the baseline is width 1, and that build moved it.
#
# This one splits the row and leaves widths 1 and 2 alone:
#
#     case 1, 2      -> 4     unchanged, so the drafter and the greedy baseline are untouched
#     case 3, 4      -> 2     the intervention
#     case 5, 6, 7, 8 -> 2    unchanged
#
# The registered prediction is the same as before: widths 3 and 4 adopt the {5,6,8} fork
# positions if the warp count is what puts the widths into two groups. The registered baseline
# control is now applicable, and widths 5, 6 and 8 must come back byte-identical to the control.
set -u
cd "$HOME/qwen38-a6000" || exit 1
SRC="src/llama.cpp"
MMVQ="$SRC/ggml/src/ggml-cuda/mmvq.cu"
BUILD="forced_down2"
OUT="results/phase_warp_${BUILD}.json"
PORT=18400
EXPECT=150

log() { echo "[$(date -Is)] $*"; }
records_in() { python3 -c "
import json
try: print(len(json.load(open('$1'))['records']))
except Exception: print(0)" 2>/dev/null; }

have=$(records_in "$OUT")
[ "${have:-0}" -ge "$EXPECT" ] && { log "$BUILD already complete (${have}/${EXPECT})"; exit 0; }

cp "$MMVQ" "$MMVQ.stock2"
restore() { cp "$MMVQ.stock2" "$MMVQ"; }
trap 'restore' EXIT

log "target check against host A"
want=3f227079003add2511437e5b1e94812e
got=$(sha256sum models/target/Qwen3.8-27B-UD-Q4_K_XL.gguf | cut -c1-32)
[ "$got" = "$want" ] || { log "TARGET MISMATCH: $got != $want. Refusing."; exit 1; }
log "  match"

log "splitting the GENERIC 1-4 row so widths 1 and 2 keep four warps"
python3 - "$MMVQ" <<'PYEOF' || exit 1
import io, sys
path = sys.argv[1]
src = io.open(path, encoding="utf-8").read()
old = """    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }"""
new = """    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
                return 4;
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }"""
if old not in src:
    print("GENERIC block does not have the expected shape; refusing to guess")
    sys.exit(1)
io.open(path, "w", encoding="utf-8").write(src.replace(old, new, 1))
print("  patched: 1,2 -> 4 | 3,4 -> 2 | 5-8 -> 2")
PYEOF

cmake --build "$SRC/build" --target llama-server -j"$(nproc)" > "logs/warp_build_${BUILD}.log" 2>&1 \
  || { log "BUILD FAILED; see logs/warp_build_${BUILD}.log"; exit 1; }

mkdir -p "warp/$BUILD"
cp -a "$SRC"/build/bin/llama-server "$SRC"/build/bin/*.so* "warp/$BUILD/"
sed -n '/if (table_id == MMVQ_PARAMETERS_GENERIC) {/,/} else if/p' "$MMVQ" > "warp/$BUILD/table.txt"

# Hash the library that carries the kernel, not llama-server: that is a 17 KB wrapper which is
# byte-identical in every build and would pass even if the patch had never applied.
h=$(sha256sum "warp/$BUILD/libggml-cuda.so.0.21.0" | cut -c1-16)
log "  libggml-cuda $h  $(stat -c%s "warp/$BUILD/libggml-cuda.so.0.21.0") bytes"
for prev in control forced_up forced_down; do
  if [ -f "warp/$prev/libggml-cuda.so.0.21.0" ]; then
    ph=$(sha256sum "warp/$prev/libggml-cuda.so.0.21.0" | cut -c1-16)
    [ "$h" = "$ph" ] && { log "  GATE FAILED: identical to the $prev build. The edit did not reach the binary."; exit 1; }
  fi
done
log "  differs from all three earlier builds"

restore
log "running $BUILD"
QWEN_WARP_BUILD="$BUILD" python3 -u harness/bench.py --matrix phase_warp --passes 1 \
    --port "$PORT" --settle-floor --allow-non-stock --out "$OUT" > "logs/warp_${BUILD}.log" 2>&1
rc=$?
have=$(records_in "$OUT")
log "$BUILD exited rc=$rc with ${have}/${EXPECT} records"
[ "$rc" -eq 0 ] && [ "${have:-0}" -ge "$EXPECT" ] || { log "GATE FAILED"; exit 1; }

log "restoring the stock source and rebuilding"
restore
cmake --build "$SRC/build" --target llama-server -j"$(nproc)" > logs/warp_restore_build2.log 2>&1 \
  && log "  stock binary restored" || log "  RESTORE BUILD FAILED"
log "done"
