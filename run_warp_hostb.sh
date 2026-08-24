#!/usr/bin/env bash
# Replicates the forced-warp intervention on host B, the second RTX 3090.
#
# Host C ran this on an A6000 and the forced widths did not adopt the fork positions of the warp
# count they were given. PREREGISTRATION.md already records that the two 3090s produce fork
# positions identical to the character while the A6000 differs, so a result taken only on the
# A6000 cannot say whether that is a property of the table or of that card. This is the control
# that separates them, and it can only be taken while a second 3090 exists.
#
# Differences from the host C run, both deliberate:
#
#   - The build check hashes libggml-cuda.so, not llama-server. llama-server is a 17 KB wrapper
#     that dlopens the backend, so it is byte-identical in all three builds; host C's build log
#     hashed it and reported three matching builds, which would have looked exactly the same had
#     the patch silently failed to apply. Each build is also required to differ from the one
#     before it.
#   - The source tree is restored and rebuilt at the end, so host B is left with the stock binary
#     it had before.
#
# Environment:
#   DEADLINE   epoch seconds; the script refuses to start a run it cannot finish before this
set -u
cd "$HOME/qwen38-remote" || exit 1
SRC="src/llama.cpp"
MMVQ="$SRC/ggml/src/ggml-cuda/mmvq.cu"
PORT=18500
EXPECT=150
DEADLINE="${DEADLINE:-0}"
RUN_BUDGET=$(( 55 * 60 ))     # host C needed ~40 min per build; 55 leaves room for the thermal gate

mkdir -p warp logs results
log() { echo "[$(date -Is)] $*"; }

records_in() {
  python3 -c "
import json
try: print(len(json.load(open('$1'))['records']))
except Exception: print(0)" 2>/dev/null
}

# ---------------------------------------------------------------- the intervention itself
# Rewrites only the MMVQ_PARAMETERS_GENERIC arm of calc_nwarps. Editing by line number would
# break the moment the file shifts, so the block is located by its guard and the replacement is
# confined to the span between that guard and the next `} else if`.
patch_table() {   # $1 = value for widths 1-4, $2 = value for widths 5-8
  python3 - "$MMVQ" "$1" "$2" <<'PYEOF'
import io, re, sys
path, lo, hi = sys.argv[1], sys.argv[2], sys.argv[3]
src = io.open(path, encoding="utf-8").read()
start = src.index("if (table_id == MMVQ_PARAMETERS_GENERIC) {")
end   = src.index("} else if", start)
block = src[start:end]
# case 1..4 -> first return, case 5..8 -> second return, default -> third; replace the first two
outs = list(re.finditer(r"return\s+(\d+);", block))
assert len(outs) >= 3, "unexpected GENERIC block shape: %d returns" % len(outs)
new = (block[:outs[0].start()] + "return %s;" % lo
       + block[outs[0].end():outs[1].start()] + "return %s;" % hi
       + block[outs[1].end():])
if new == block:
    print("UNCHANGED"); sys.exit(1)
io.open(path, "w", encoding="utf-8").write(src[:start] + new + src[end:])
print("PATCHED 1-4 -> %s, 5-8 -> %s" % (lo, hi))
PYEOF
}

show_table() {
  sed -n "/if (table_id == MMVQ_PARAMETERS_GENERIC) {/,/} else if/p" "$MMVQ"
}

cp "$MMVQ" "$MMVQ.stock"
restore() { cp "$MMVQ.stock" "$MMVQ"; }
trap 'restore' EXIT

log "target check against host A"
want=3f227079003add2511437e5b1e94812e
got=$(sha256sum models/target/Qwen3.8-27B-UD-Q4_K_XL.gguf | cut -c1-32)
[ "$got" = "$want" ] || { log "TARGET MISMATCH: $got != $want. Refusing to run."; exit 1; }
log "  match"

prev_hash=""
for BUILD in control forced_up forced_down; do
  OUT="results/phase_warp_${BUILD}.json"
  have=$(records_in "$OUT")
  if [ "${have:-0}" -ge "$EXPECT" ]; then log "$BUILD already complete (${have}/${EXPECT}); skipping"; continue; fi

  if [ "$DEADLINE" -gt 0 ]; then
    left=$(( DEADLINE - $(date +%s) ))
    if [ "$left" -lt "$RUN_BUDGET" ]; then
      log "only ${left}s left before the deadline and a build needs ~${RUN_BUDGET}s."
      log "Stopping before $BUILD rather than leaving a partial result on a machine that is"
      log "about to go away. Completed builds are already on disk."
      break
    fi
  fi

  log "=== build $BUILD ==="
  restore
  case "$BUILD" in
    control)     log "  table unmodified: 1-4 -> 4, 5-8 -> 2" ;;
    forced_up)   patch_table 4 4 | sed 's/^/    /' || { log "patch failed"; exit 1; } ;;
    forced_down) patch_table 2 2 | sed 's/^/    /' || { log "patch failed"; exit 1; } ;;
  esac

  cmake --build "$SRC/build" --target llama-server -j"$(nproc)" > "logs/warp_build_${BUILD}.log" 2>&1
  rc=$?
  [ "$rc" -eq 0 ] || { log "BUILD FAILED rc=$rc; see logs/warp_build_${BUILD}.log"; exit 1; }

  mkdir -p "warp/$BUILD"
  cp -a "$SRC"/build/bin/llama-server "$SRC"/build/bin/*.so* "warp/$BUILD/"
  show_table > "warp/$BUILD/table.txt"

  # The kernel lives in libggml-cuda.so. Hashing llama-server here would compare a wrapper that
  # is identical in every build and would pass even if the patch had not applied.
  h=$(sha256sum "warp/$BUILD/libggml-cuda.so.0.21.0" | cut -c1-16)
  log "  libggml-cuda $h  $(stat -c%s "warp/$BUILD/libggml-cuda.so.0.21.0") bytes"
  if [ -n "$prev_hash" ] && [ "$h" = "$prev_hash" ]; then
    log "  GATE FAILED: this build is byte-identical to the previous one. The table edit did not"
    log "  reach the binary, so the comparison would be meaningless. Stopping."
    exit 1
  fi
  prev_hash="$h"

  log "  running $BUILD"
  QWEN_WARP_BUILD="$BUILD" python3 -u harness/bench.py --matrix phase_warp --passes 1 \
      --port "$PORT" --settle-floor --allow-non-stock --out "$OUT" \
      > "logs/warp_${BUILD}.log" 2>&1
  rc=$?
  have=$(records_in "$OUT")
  log "  $BUILD exited rc=$rc with ${have}/${EXPECT} records"
  [ "$rc" -eq 0 ] && [ "${have:-0}" -ge "$EXPECT" ] || { log "  GATE FAILED on $BUILD; stopping"; exit 1; }
done

log "restoring the stock source and rebuilding so host B keeps the binary it started with"
restore
cmake --build "$SRC/build" --target llama-server -j"$(nproc)" > logs/warp_restore_build.log 2>&1 \
  && log "  stock binary restored" || log "  RESTORE BUILD FAILED; see logs/warp_restore_build.log"

log "done"
for b in control forced_up forced_down; do
  printf "  %-12s %s records\n" "$b" "$(records_in results/phase_warp_${b}.json)"
done
