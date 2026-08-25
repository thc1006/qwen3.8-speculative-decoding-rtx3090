#!/usr/bin/env bash
# Phase Q-small driver - the quantization ladder INCLUDING bf16, on the card already here.
#
# The 27B ladder cannot reach bf16: that file is 50 GB and fits on neither 24 nor 48 GB. Swapping
# the model rather than the hardware gets the anchor. Qwen3.5-9B-MTP ships the full ladder and its
# bf16 is 18.41 GB, and its Q4_K_M is the exact file llama.cpp #26750 reports on.
#
# The four rungs total ~42 GB and this host has less free disk, so each one is downloaded,
# measured, verified complete, and only then deleted. Deletion is the irreversible step and is
# gated on the result file holding every expected record: a rung whose run died halfway keeps its
# weights so it can be retried without downloading again.
#
# Environment:
#   GPU=<n>      nvidia-smi index of the card to use (default 0)
#   PASSES=3
#   RUNGS="..."  subset of rungs, in order
#   KEEP=1       never delete weights
set -u
cd "$(dirname "$0")" || exit 1

GPU="${GPU:-0}"
PASSES="${PASSES:-3}"
PORT="${PORT:-18170}"
KEEP="${KEEP:-0}"
RUNGS="${RUNGS:-Q4_K_M Q6_K Q8_0 BF16}"
N_ARMS=4          # phase_qsmall defines: baseline + mtp-n2 + mtp-n3 + mtp-n5

declare -A FILE=(
  [Q4_K_M]=Qwen3.5-9B-MTP-Q4_K_M.gguf
  [Q6_K]=Qwen3.5-9B-MTP-Q6_K.gguf
  [Q8_0]=Qwen3.5-9B-MTP-Q8_0.gguf
  [BF16]=Qwen3.5-9B-MTP-BF16.gguf
)
declare -A SIZE_GB=( [Q4_K_M]=5.87 [Q6_K]=7.68 [Q8_0]=9.79 [BF16]=18.41 )

log() { echo "[$(date -Is)] $*"; }
records_in() { python3 -c "
import json
try: print(len(json.load(open('$1'))['records']))
except Exception: print(0)" 2>/dev/null; }

# Computed without importing the matrix: it refuses to import when its target file is absent,
# which is exactly the state this check runs in once a rung's weights have been deleted.
N_PROMPTS=$(python3 -c "
import sys; sys.path.insert(0,'harness'); import prompts as P; print(len(P.PROMPTS))")
EXPECTED=$(( N_ARMS * N_PROMPTS * PASSES ))
log "expecting ${N_ARMS} arms x ${N_PROMPTS} prompts x ${PASSES} passes = ${EXPECTED} records per rung"

# A zero or empty count here makes every "got >= EXPECTED" test pass, which would report a rung
# that measured nothing as complete and, in this script, delete its weights. The python above can
# fail for reasons that have nothing to do with the run, so the value is checked before use.
case "${EXPECTED}" in
  ''|*[!0-9]*|0)
    log "FATAL: expected record count came out as '${EXPECTED}' (N_PROMPTS='${N_PROMPTS}')."
    log "  Refusing to continue: a zero here passes every completeness gate."
    exit 1 ;;
esac


mkdir -p models/qwen35_9b analysis logs results

for RUNG in $RUNGS; do
  F="${FILE[$RUNG]:-}"
  if [ -z "$F" ]; then log "unknown rung $RUNG; skipping"; continue; fi
  OUT="results/phase_qsmall_${RUNG}.json"

  have=$(records_in "$OUT")
  if [ "${have:-0}" -ge "$EXPECTED" ]; then
    log "$RUNG already complete (${have}/${EXPECTED}); skipping"
    continue
  fi

  free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  need=${SIZE_GB[$RUNG]}
  if [ "${free_gb:-0}" -lt "$(printf '%.0f' "$need")" ]; then
    log "$RUNG needs about ${need} GB and only ${free_gb} GB is free; skipping"
    continue
  fi

  SRC=""
  for cand in "models/qwen35_9b/$F"; do
    [ -f "$cand" ] && { SRC="$cand"; break; }
  done
  if [ -z "$SRC" ]; then
    log "downloading $F (${need} GB)"
    HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download unsloth/Qwen3.5-9B-MTP-GGUF "$F" \
      --local-dir models/qwen35_9b >> logs/phase_qsmall_download.log 2>&1 || {
        log "download of $F FAILED - see logs/phase_qsmall_download.log"; continue; }
    # a split GGUF arrives as -00001-of-000NN parts and the single-file name never appears
    if [ ! -f "models/qwen35_9b/$F" ]; then
      log "$F did not arrive as a single file; it is probably split into parts."
      log "  check the repo's file list and pass the first part to llama-server instead."
      continue
    fi
    SRC="models/qwen35_9b/$F"
    DOWNLOADED=1
  else
    log "reusing $SRC (not downloading)"
    DOWNLOADED=0
  fi
  log "disk after staging: $(df -h . | tail -1 | awk '{print $4}') free"

  QWEN_QS_TARGET="$RUNG" python3 -u harness/bench.py \
    --matrix phase_qsmall --passes "$PASSES" --gpu "$GPU" --port "$PORT" \
    --settle-floor --out "$OUT" > "logs/phase_qsmall_${RUNG}.log" 2>&1
  rc=$?
  got=$(records_in "$OUT")
  log "$RUNG exited rc=$rc with ${got}/${EXPECTED} records"

  if [ "$rc" -eq 0 ] && [ "${got:-0}" -ge "$EXPECTED" ]; then
    python3 harness/analyze.py     "$OUT" > "analysis/phase_qsmall_${RUNG}.txt"      2>&1
    python3 harness/cost_model.py  "$OUT" > "analysis/phase_qsmall_${RUNG}_cost.txt" 2>&1
    log "$RUNG complete; reports written"
    if [ "$KEEP" = "0" ] && [ "$DOWNLOADED" = "1" ]; then
      log "removing $SRC (result verified complete)"
      rm -f "$SRC"
    fi
  else
    log "$RUNG INCOMPLETE - keeping weights for a retry; inspect logs/phase_qsmall_${RUNG}.log"
  fi
done

log "done. rungs on disk:"
for RUNG in $RUNGS; do
  printf "  %-8s %s records\n" "$RUNG" "$(records_in "results/phase_qsmall_${RUNG}.json")"
done
