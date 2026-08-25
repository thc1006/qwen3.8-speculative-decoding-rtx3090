#!/usr/bin/env bash
# Phase L on the 48 GB card, at f16 KV, plus one q8_0 rung as the cross-device control.
#
# Every rung the 24 GB card runs uses q8_0, because that is what makes room for the depth. So a
# collapse seen there cannot be separated from the cache precision that enabled the measurement.
# 48 GB holds an f16 cache to about 102 400 tokens, which covers this ladder's deepest rung, so
# the two can be told apart.
#
# Order is by information, not by depth. The shallow f16 rung comes first because the cliff test
# is a ratio against a method's own shallowest rung, and without it the deep rungs only give
# absolute throughput, which differs between the two cards for reasons that have nothing to do
# with the cliff. The q8_0 rung at the same depth comes last and is what separates "the f16 cache
# changed it" from "the other card changed it".
set -u
cd "$HOME/qwen38-a6000" || exit 1
mkdir -p results analysis logs
PASSES="${PASSES:-3}"
PORT="${PORT:-18260}"
GPU="${GPU:-0}"
N_ARMS=4; PROMPTS=15
EXPECTED=$(( N_ARMS * PROMPTS * PASSES ))

log() { echo "[$(date -Is)] $*"; }
records_in() { python3 -c "
import json
try: print(len(json.load(open('$1'))['records']))
except Exception: print(0)" 2>/dev/null; }

# The value host A verified, the same constant run_warp_v2_hostc.sh checks against. The
# .sha256 beside the file is "<size>:<mtime> <hash>", not "<hash>  <name>", and reading it with
# cut -c1-32 took the size and mtime and reported a mismatch against a target that is correct.
# --allow-non-stock below is not waving away an overclock. This host runs no X server, so
# nvidia-settings cannot report clock offsets for this UUID and the harness records them as
# unverifiable, which it then reports as not stock. The power limit is 300 W against a 300 W
# default and the clocks are at their defaults, so what is unknown is the offsets, not whether
# the limit was moved. The four warp builds measured on this card ran the same way.
log "gpu state, recorded rather than assumed"
python3 - <<'PY' 2>/dev/null || true
import sys; sys.path.insert(0, "harness")
import telemetry as T
oc = T.overclock_state(0)
for k in ("power_limit_w", "power_default_limit_w", "clocks_max_sm_mhz", "clocks_max_memory_mhz",
          "graphics_clock_offset", "mem_transfer_rate_offset", "is_stock"):
    print(f"    {k:26} {oc.get(k)}")
PY

log "target check"
want=3f227079003add2511437e5b1e94812e
got=$(sha256sum models/target/Qwen3.8-27B-UD-Q4_K_XL.gguf | cut -c1-32)
[ "$got" = "$want" ] || { log "TARGET MISMATCH: $got != $want. Refusing."; exit 1; }
log "  ok ($got)"

# matrix:depth pairs, in the order stated above
RUNGS="${RUNGS:-phase_lf:8192 phase_lf:81920 phase_lf:98304 phase_l:81920}"
log "plan:${RUNGS}   ${EXPECTED} records each"

T0=$(date +%s)
for spec in $RUNGS; do
  MAT="${spec%%:*}"; D="${spec##*:}"
  KV=$([ "$MAT" = "phase_lf" ] && echo f16 || echo q8_0)
  OUT="results/${MAT}_${D}.json"
  got=$(records_in "$OUT")
  if [ "${got:-0}" -ge "$EXPECTED" ]; then log "$MAT $D already complete (${got}); skipping"; continue; fi
  [ "${got:-0}" -gt 0 ] && { log "$MAT $D partial (${got}); archiving"; mv "$OUT" "${OUT%.json}.partial.$(date +%s).json"; }

  log "=== ${MAT} depth ${D} (${KV} KV) ==="
  t0=$(date +%s)
  QWEN_L_DEPTH="$D" python3 -u harness/bench.py --matrix "$MAT" --passes "$PASSES" \
      --gpu "$GPU" --port "$PORT" --settle-floor --allow-non-stock \
      --out "$OUT" > "logs/${MAT}_${D}.log" 2>&1
  rc=$?; el=$(( $(date +%s) - t0 )); got=$(records_in "$OUT")
  log "  exited rc=$rc with ${got}/${EXPECTED} in ${el}s ($(( el / 60 )) min)"
  if [ "$rc" -ne 0 ] || [ "${got:-0}" -lt "$EXPECTED" ]; then
    log "  rung incomplete; keeping what is on disk and moving to the next one"
    tail -5 "logs/${MAT}_${D}.log" | sed 's/^/    /'
  fi
done
log "ladder done after $(( ($(date +%s) - T0) / 60 )) min"
ls -la results/phase_lf_*.json results/phase_l_*.json 2>/dev/null | sed 's/^/  /'
