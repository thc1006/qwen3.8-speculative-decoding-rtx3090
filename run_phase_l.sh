#!/usr/bin/env bash
# Phase L driver - the context-depth ladder, one server per depth.
#
# Depth is a server property: it sets `-c`, and therefore how much VRAM the KV cache claims. It
# cannot vary between arms of one run, so each depth is its own invocation with its own result
# file, and the ladder is assembled at analysis time.
#
# Runs shallow first. If the cliff reported in llama.cpp #27623 reproduces here, the deep rungs
# get slow - a 200-token generation at 1.4 tok/s is about 143 s - and running shallow first means
# the cheap, most-likely-to-be-informative rungs are already on disk before the expensive ones
# start. A rung that cannot allocate its KV cache is reported and skipped, not retried: on a
# 24 GB card the 96 K rung sits at about 95 % of the card and may legitimately not fit.
#
# Environment:
#   GPU=0       nvidia-smi index
#   PASSES=3
#   DEPTHS="8192 32768 65536 81920 98304"
set -u
cd /home/thc1006/dev/qwen3.8-speculative-decoding-rtx3090 || exit 1
mkdir -p results analysis logs

GPU="${GPU:-0}"
PASSES="${PASSES:-3}"
PORT="${PORT:-18230}"
DEPTHS="${DEPTHS:-8192 32768 65536 81920 98304}"
N_ARMS=4
PROMPTS=15         # phase_l declares PROMPTS_PER_CLASS = 3 over 5 classes
EXPECTED=$(( N_ARMS * PROMPTS * PASSES ))

log() { echo "[$(date -Is)] $*"; }
records_in() {
  python3 -c "
import json
try: print(len(json.load(open('$1'))['records']))
except Exception: print(0)" 2>/dev/null
}

LADDER_T0=$(date +%s)
log "depth ladder:${DEPTHS} - expecting ${EXPECTED} records each"

for D in $DEPTHS; do
  OUT="results/phase_l_${D}.json"
  got=$(records_in "$OUT")
  if [ "${got:-0}" -ge "$EXPECTED" ]; then
    log "depth $D already complete (${got}/${EXPECTED}); skipping"
    continue
  fi
  [ "${got:-0}" -gt 0 ] && { log "depth $D partial (${got}/${EXPECTED}); archiving"; \
      mv "$OUT" "${OUT%.json}.partial.$(date +%s).json"; }

  log "=== depth ${D} tokens ==="
  # This ladder exists to find out whether decode collapses past long context. If it does, a rung
  # takes as long as the collapse is deep - at the 1.4 tok/s the report describes, 160 tokens is
  # 114 seconds a request and a rung is close to six hours. Nothing in the harness has a timeout,
  # so the only protection is that the log says where it is going while there is still time to
  # decide. QWEN_L_BUDGET_S stops the ladder between rungs if one is set.
  t0=$(date +%s)
  QWEN_L_DEPTH="$D" python3 -u harness/bench.py --matrix phase_l --passes "$PASSES" \
      --gpu "$GPU" --port "$PORT" --out "$OUT" > "logs/phase_l_${D}.log" 2>&1
  rc=$?
  el=$(( $(date +%s) - t0 ))
  got_now=$(records_in "$OUT")
  log "depth $D took ${el}s for ${got_now:-0} records"
  if [ "${got_now:-0}" -gt 0 ]; then
    per=$(( el / got_now ))
    left=$(( EXPECTED - got_now ))
    log "  ${per}s per record; the remaining rungs are $(( ${#DEPTHS} )) deep and each is "\
        "${EXPECTED} records, so at this rate one more rung is about $(( per * EXPECTED / 60 )) min"
  fi
  if [ -n "${QWEN_L_BUDGET_S:-}" ]; then
    spent=$(( $(date +%s) - LADDER_T0 ))
    if [ "$spent" -ge "${QWEN_L_BUDGET_S}" ]; then
      log "  ladder budget of ${QWEN_L_BUDGET_S}s is spent after ${spent}s; stopping between rungs"
      log "  rungs completed so far keep their results; nothing is half-written"
      break
    fi
  fi
  got=$(records_in "$OUT")
  log "depth $D exited rc=$rc with ${got}/${EXPECTED} records"

  if [ "$rc" -eq 0 ] && [ "${got:-0}" -ge "$EXPECTED" ]; then
    python3 harness/analyze.py "$OUT" > "analysis/phase_l_${D}.txt" 2>&1
    log "depth $D complete; report written"
  elif grep -qiE "out of memory|failed to allocate|cudaMalloc" "logs/phase_l_${D}.log" 2>/dev/null; then
    log "depth $D DID NOT FIT on this card - expected at the top of the ladder, continuing"
  else
    log "depth $D INCOMPLETE for a reason other than capacity; see logs/phase_l_${D}.log"
  fi
done

log "ladder finished"
python3 harness/analyze_depth.py 2>/dev/null || log "run analyze_depth.py once rungs exist"
