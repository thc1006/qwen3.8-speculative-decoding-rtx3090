#!/usr/bin/env bash
# Phase Q driver - the Qwen3.8-27B target-quantization ladder, disk-staged.
#
# The four rungs total ~93 GB of GGUF and this host has far less free disk, so each rung is
# downloaded, measured, verified complete, and only then deleted. Deletion is the one
# irreversible step here, so it is gated on the result file actually containing every expected
# record: a rung whose run died halfway keeps its weights so it can be retried without
# re-downloading 25 GB.
#
# Capacity (8192 context, q8_0 KV, +1.9 GB compute buffer, measured on this host):
#   UD-Q4_K_XL 19.8 GB   -> runs on 24 GB
#   UD-Q5_K_XL 23.1 GB   -> marginal on 24 GB (96 %); attempt it, expect it may not allocate
#   UD-Q6_K_XL 27.5 GB   -> needs the 48 GB card
#   Q8_0       31.3 GB   -> needs the 48 GB card
# BF16 is 52.2 GB and fits on neither; the bf16 anchor comes from phase_qsmall instead.
#
# Environment:
#   GPU=1        nvidia-smi index of the card to use
#   PASSES=3
#   RUNGS="..."  subset of rungs, in order
#   KEEP=1       never delete weights
set -u
cd /home/thc1006/dev/qwen3.8-speculative-decoding-rtx3090 || exit 1
mkdir -p models/quant_ladder results analysis logs

GPU="${GPU:-0}"
PASSES="${PASSES:-3}"
PORT="${PORT:-18160}"
KEEP="${KEEP:-0}"
N_ARMS=4          # phase_q defines: baseline + mtp-n2 + mtp-n3 + mtp-n5

declare -A FILE=(
  [UD-Q4_K_XL]=Qwen3.8-27B-UD-Q4_K_XL.gguf
  [UD-Q5_K_XL]=Qwen3.8-27B-UD-Q5_K_XL.gguf
  [UD-Q6_K_XL]=Qwen3.8-27B-UD-Q6_K_XL.gguf
  [Q8_0]=Qwen3.8-27B-Q8_0.gguf
)
declare -A NEED_GB=(
  [UD-Q4_K_XL]=19.8 [UD-Q5_K_XL]=23.1 [UD-Q6_K_XL]=27.5 [Q8_0]=31.3
)

log() { echo "[$(date -Is)] $*"; }

# Expected record count, computed WITHOUT importing the matrix module. The matrix refuses to
# import when its target file is absent, which is precisely the state this check runs in after
# the weights have been deleted.
N_PROMPTS=$(python3 -c "
import sys; sys.path.insert(0,'harness'); import prompts as P; print(len(P.PROMPTS))")
EXPECTED=$(( N_ARMS * N_PROMPTS * PASSES ))
log "expecting ${N_ARMS} arms x ${N_PROMPTS} prompts x ${PASSES} passes = ${EXPECTED} records per rung"

records_in() {   # $1 = json path -> record count, 0 if unreadable
  python3 -c "
import json
try: print(len(json.load(open('$1'))['records']))
except Exception: print(0)" 2>/dev/null
}

vram_gb=$(python3 -c "
import sys; sys.path.insert(0,'harness')
import devices as D
try: print(f'{D.get_device($GPU).vram_gb:.1f}')
except Exception as e: print('0')")
log "GPU $GPU has ${vram_gb} GB"
if [ "$(python3 -c "print(1 if float('$vram_gb')<1 else 0)")" = "1" ]; then
  log "no such GPU - set GPU=<nvidia-smi index>"; exit 1
fi

# Default rung list: only what this card can actually hold.
if [ -z "${RUNGS:-}" ]; then
  RUNGS=""
  for r in UD-Q4_K_XL UD-Q5_K_XL UD-Q6_K_XL Q8_0; do
    if [ "$(python3 -c "print(1 if float('$vram_gb') >= ${NEED_GB[$r]} else 0)")" = "1" ]; then
      RUNGS="$RUNGS $r"
    fi
  done
  log "auto-selected rungs for a ${vram_gb} GB card:${RUNGS}"
fi

for RUNG in $RUNGS; do
  F="${FILE[$RUNG]}"
  OUT="results/phase_q_${RUNG}.json"

  got=$(records_in "$OUT")
  if [ "${got:-0}" -ge "$EXPECTED" ]; then
    log "$RUNG already complete (${got}/${EXPECTED}) - skipping"
    continue
  fi
  if [ "${got:-0}" -gt 0 ]; then
    log "$RUNG has a PARTIAL result (${got}/${EXPECTED}); archiving it and re-running"
    mv "$OUT" "${OUT%.json}.partial.$(date +%s).json"
  fi

  log "=== rung $RUNG (needs ~${NEED_GB[$RUNG]} GB VRAM) ==="

  # reuse the copy this repo already holds rather than downloading a second one
  SRC=""
  for cand in "models/target/$F" "models/quant_ladder/$F"; do
    [ -f "$cand" ] && { SRC="$cand"; break; }
  done
  if [ -z "$SRC" ]; then
    log "downloading $F"
    HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download unsloth/Qwen3.8-27B-GGUF "$F" \
      --local-dir models/quant_ladder >> logs/phase_q_download.log 2>&1 || {
        log "download of $F FAILED - see logs/phase_q_download.log"; continue; }
    SRC="models/quant_ladder/$F"
    DOWNLOADED=1
  else
    log "reusing $SRC (not downloading)"
    DOWNLOADED=0
  fi
  log "disk after staging: $(df -h / | tail -1 | awk '{print $4}') free"

  QWEN_Q_TARGET="$RUNG" python3 -u harness/bench.py \
    --matrix phase_q --passes "$PASSES" --gpu "$GPU" --port "$PORT" \
    --settle-floor --out "$OUT" > "logs/phase_q_${RUNG}.log" 2>&1
  rc=$?
  got=$(records_in "$OUT")
  log "$RUNG exited rc=$rc with ${got}/${EXPECTED} records"

  if [ "$rc" -eq 0 ] && [ "${got:-0}" -ge "$EXPECTED" ]; then
    python3 harness/analyze.py    "$OUT" > "analysis/phase_q_${RUNG}.txt"      2>&1
    python3 harness/cost_model.py "$OUT" > "analysis/phase_q_${RUNG}_cost.txt" 2>&1
    log "$RUNG complete; reports written"
    # Only ever delete a file this script downloaded into quant_ladder. models/target holds the
    # shared target for every other phase and is never touched.
    if [ "$KEEP" = "0" ] && [ "$DOWNLOADED" = "1" ] && [ "$SRC" = "models/quant_ladder/$F" ]; then
      log "removing $SRC (result verified complete)"
      rm -f "$SRC"
    fi
  else
    log "$RUNG INCOMPLETE - keeping weights for a retry; inspect logs/phase_q_${RUNG}.log"
  fi
  log "disk now: $(df -h / | tail -1 | awk '{print $4}') free"
done

log "ladder finished"
ls -1 analysis/phase_q_*.txt 2>/dev/null || log "no rung completed"
