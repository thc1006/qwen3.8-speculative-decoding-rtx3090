#!/usr/bin/env bash
# The forced-warp intervention, rebuilt so that every build comes from one cmake configure.
#
# Registered in PREREGISTRATION.md Correction 8. The previous forced-down2 was compared against a
# control built under a different configure: its log begins with a cmake configure and the
# control's does not, a reconfigure regenerates flags.make for every target, and all eleven
# ggml-base sources recompiled into a different libggml-base.so. That is enough to make a width-1
# greedy baseline - no drafter, byte-identical mul_mat_vec_q machine code - come out different,
# and it is why that direction was withdrawn.
#
# Four builds, one configure, produced back to back and run back to back:
#
#   control       stock                            1-4 -> 4, 5-8 -> 2
#   forced_up     high widths take the low count   5-8 -> 4
#   forced_down2  low widths take the high count   3,4 -> 2, with 1,2 left at 4 so the drafter
#                                                  and the greedy baseline are untouched
#   control2      stock again, no source change    the guard
#
# control2 exists because nothing in the earlier design could have caught what went wrong. Two
# builds from one configure differing in no source at all must produce byte-identical output. If
# they do not, no comparison in this set means anything and the run says so instead of producing
# numbers.
set -u
cd "$HOME/qwen38-a6000" || exit 1
SRC="src/llama.cpp"
MMVQ="$SRC/ggml/src/ggml-cuda/mmvq.cu"
PORT=18400
EXPECT=150
BUILDS="control forced_up forced_down2 control2"

log() { echo "[$(date -Is)] $*"; }
h() { sha256sum "$1" 2>/dev/null | cut -c1-16; }

# Hash the code, not the file. Two builds of identical source under one configure differ in four
# bytes at offset 56408893 of libggml-cuda.so - the ASCII "1586" against "37ec" - which is an
# identifier nvcc embeds, derived from its temporary filenames. The build id is identical and the
# stripped content is byte-identical. A whole-file hash reads that identifier and calls two
# equivalent builds different, which is what stopped the first attempt at this run on a gate that
# was wrong rather than a build that was.
hcode() {
  local t; t=$(mktemp /tmp/hcode.XXXXXX.so) || return 1
  cp "$1" "$t" 2>/dev/null || { rm -f "$t"; return 1; }
  strip --strip-all "$t" 2>/dev/null
  objcopy --remove-section=.note.gnu.build-id --remove-section=.comment "$t" 2>/dev/null
  sha256sum "$t" | cut -c1-16
  rm -f "$t"
}
records_in() { python3 -c "
import json
try: print(len(json.load(open('$1'))['records']))
except Exception: print(0)" 2>/dev/null; }

cp "$MMVQ" "$MMVQ.v2stock"
restore() { cp "$MMVQ.v2stock" "$MMVQ"; }
trap restore EXIT

log "target check against host A"
want=3f227079003add2511437e5b1e94812e
got=$(sha256sum models/target/Qwen3.8-27B-UD-Q4_K_XL.gguf | cut -c1-32)
[ "$got" = "$want" ] || { log "TARGET MISMATCH: $got != $want. Refusing."; exit 1; }
log "  match"

# ---------------------------------------------------------------- one configure, up front
log "configuring once, so no build below can reconfigure under its own feet"
cmake -S "$SRC" -B "$SRC/build" > logs/v2_configure.log 2>&1 \
  || { log "configure FAILED; see logs/v2_configure.log"; exit 1; }
log "  configured"

patch_table() {   # $1 = build name
  restore
  case "$1" in
    control|control2) return 0 ;;
    forced_up)
      python3 - "$MMVQ" <<'PYEOF' || return 1
import io, sys
p = sys.argv[1]; s = io.open(p, encoding="utf-8").read()
old = """            case 5:
            case 6:
            case 7:
            case 8:
                return 2;"""
new = """            case 5:
            case 6:
            case 7:
            case 8:
                return 4;"""
assert old in s, "forced_up anchor missing"
io.open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
PYEOF
      ;;
    forced_down2)
      python3 - "$MMVQ" <<'PYEOF' || return 1
import io, sys
p = sys.argv[1]; s = io.open(p, encoding="utf-8").read()
old = """            case 1:
            case 2:
            case 3:
            case 4:
                return 4;"""
new = """            case 1:
            case 2:
                return 4;
            case 3:
            case 4:
                return 2;"""
assert old in s, "forced_down2 anchor missing"
io.open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
PYEOF
      ;;
  esac
}

# ---------------------------------------------------------------- build all four, then check
declare -A CUDA_H BASE_H
for b in $BUILDS; do
  log "=== build $b ==="
  patch_table "$b" || { log "patch failed for $b"; exit 1; }
  cmake --build "$SRC/build" --target llama-server -j"$(nproc)" > "logs/v2_build_$b.log" 2>&1 \
    || { log "BUILD FAILED for $b"; exit 1; }
  if head -3 "logs/v2_build_$b.log" | grep -q "llama.cpp version"; then
    log "  GATE FAILED: this build reconfigured, which is the thing the run exists to avoid."
    exit 1
  fi
  mkdir -p "warp_v2/$b"
  cp -a "$SRC"/build/bin/llama-server "$SRC"/build/bin/*.so* "warp_v2/$b/"
  sed -n '/if (table_id == MMVQ_PARAMETERS_GENERIC) {/,/} else if/p' "$MMVQ" > "warp_v2/$b/table.txt"
  CUDA_H[$b]=$(hcode "warp_v2/$b/libggml-cuda.so.0.21.0")
  BASE_H[$b]=$(hcode "warp_v2/$b/libggml-base.so.0.21.0")
  log "  libggml-cuda (code) ${CUDA_H[$b]}   libggml-base (code) ${BASE_H[$b]}"
done

log "=== build gates ==="
# Every build shares one configure and only mmvq.cu ever changes, so the CPU-side library must be
# the same in all four. It differing is exactly what invalidated the previous attempt.
base_ref=${BASE_H[control]}
for b in $BUILDS; do
  [ "${BASE_H[$b]}" = "$base_ref" ] || {
    log "  GATE FAILED: $b has libggml-base ${BASE_H[$b]} against control's $base_ref."
    log "  The CPU-side library differs between builds that share a configure and a source tree."
    log "  Nothing below would be a comparison of the table. Stopping."; exit 1; }
done
log "  libggml-base identical across all four: $base_ref"
[ "${CUDA_H[control]}" = "${CUDA_H[control2]}" ] || {
  log "  GATE FAILED: control and control2 differ in libggml-cuda despite identical source."
  log "  ${CUDA_H[control]} against ${CUDA_H[control2]}. Stopping."; exit 1; }
log "  control and control2 byte-identical in libggml-cuda: ${CUDA_H[control]}"
for b in forced_up forced_down2; do
  [ "${CUDA_H[$b]}" != "${CUDA_H[control]}" ] || {
    log "  GATE FAILED: $b has the same libggml-cuda as control; its edit never reached the binary."
    exit 1; }
done
log "  both forced builds differ from control in libggml-cuda, as they must"

restore
log "source restored; the four binaries are snapshots and no longer depend on the tree"

# ---------------------------------------------------------------- run all four, back to back
for b in $BUILDS; do
  OUT="results/phase_warp_v2_${b}.json"
  have=$(records_in "$OUT")
  [ "${have:-0}" -ge "$EXPECT" ] && { log "$b already complete (${have}/${EXPECT}); skipping"; continue; }
  log "=== run $b ==="
  QWEN_WARP_BUILD="$b" QWEN_WARP_DIR=warp_v2 python3 -u harness/bench.py --matrix phase_warp \
      --passes 1 --port "$PORT" --settle-floor --allow-non-stock --out "$OUT" \
      > "logs/v2_run_$b.log" 2>&1
  rc=$?
  have=$(records_in "$OUT")
  log "  $b exited rc=$rc with ${have}/${EXPECT} records"
  [ "$rc" -eq 0 ] && [ "${have:-0}" -ge "$EXPECT" ] || { log "  GATE FAILED on $b; stopping"; exit 1; }
done

log "all four complete"
for b in $BUILDS; do printf "  %-14s %s records\n" "$b" "$(records_in results/phase_warp_v2_${b}.json)"; done
