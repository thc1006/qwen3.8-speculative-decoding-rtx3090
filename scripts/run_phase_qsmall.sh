#!/usr/bin/env bash
# Phase Q-small driver - the quantization ladder INCLUDING bf16, on the card already here.
#
# The 27B ladder cannot reach bf16: that file is 50 GB and fits on neither 24 nor 48 GB. Swapping
# the model rather than the hardware gets the anchor. Qwen3.5-9B-MTP ships the full ladder, its
# bf16 is 17.14 GiB, and its Q4_K_M is the exact file llama.cpp #26750 reports on.
#
# The four rungs total 38.9 GiB, so each one is downloaded, measured, verified complete, and only
# then deleted. Deletion is the irreversible step and is gated on the result being a complete run
# rather than on a record count: a rung whose run died halfway keeps its weights so it can be
# retried without downloading again.
#
# Two numbers, two meanings, kept in separate tables because merging them is what broke
# scripts/run_phase_q.sh:
#   NEED_VRAM_GB   what the rung costs on the card, from the matrix's own table
#   NEED_DISK_GIB  what the GGUF costs on disk, in GiB because that is the unit `df -BG` answers
#                  in. Measured from the repo's file metadata, not converted from the VRAM figure.
#
# | rung   | file GiB | VRAM GB | 24 GB card        |
# |--------|---------:|--------:|-------------------|
# | Q4_K_M |     5.47 |     9.0 | yes               |
# | Q6_K   |     7.16 |    10.8 | yes               |
# | Q8_0   |     9.11 |    12.9 | yes               |
# | BF16   |    17.14 |    21.8 | marginal (91 %)   |
#
# Environment:
#   GPU=<n>      nvidia-smi index of the card to use (default 0)
#   PASSES=3
#   RUNGS="..."  subset of rungs, in order
#   KEEP=1       never delete weights
set -u
# The repo root, not this script's directory: the drivers moved into scripts/ and every
# path below -- harness/, models/, results/, logs/ -- is written relative to the root.
cd "$(dirname "$0")/.." || exit 1

GPU="${GPU:-0}"
PASSES="${PASSES:-3}"
PORT="${PORT:-18170}"
KEEP="${KEEP:-0}"
RUNGS="${RUNGS:-Q4_K_M Q6_K Q8_0 BF16}"
DISK_MARGIN_GIB="${DISK_MARGIN_GIB:-5}"

# These names are the ones the repo actually publishes. An earlier version of this table said
# `Qwen3.5-9B-MTP-Q4_K_M.gguf`, which does not exist: the repo is named ...-MTP-GGUF but the files
# inside it are not. Every rung would have 404'd after the guard let it through, and the matrix -
# which had the names right - would then have failed to import. test_harness.py now checks this
# table against the matrix's.
declare -A FILE=(
  [Q4_K_M]=Qwen3.5-9B-Q4_K_M.gguf
  [Q6_K]=Qwen3.5-9B-Q6_K.gguf
  [Q8_0]=Qwen3.5-9B-Q8_0.gguf
  [BF16]=Qwen3.5-9B-BF16.gguf
)
declare -A NEED_VRAM_GB=( [Q4_K_M]=9.0 [Q6_K]=10.8 [Q8_0]=12.9 [BF16]=21.8 )
declare -A NEED_DISK_GIB=( [Q4_K_M]=5.47 [Q6_K]=7.16 [Q8_0]=9.11 [BF16]=17.14 )

log() { echo "[$(date -Is)] $*"; }

N_PROMPTS=$(python3 -c "
import sys; sys.path.insert(0,'harness'); import prompts as P; print(len(P.PROMPTS))")
case "${N_PROMPTS}" in
  ''|*[!0-9]*|0)
    log "FATAL: prompt count came out as '${N_PROMPTS}'; a zero here passes every gate."; exit 1 ;;
esac
log "prompt set: ${N_PROMPTS} prompts, ${PASSES} passes"

# Completeness gate. The previous gate was `got >= N_ARMS*N_PROMPTS*PASSES` with N_ARMS=4 written
# into this file, while the matrix defines FIVE arms - baseline plus n-max 2, 3, 5 and 6. A run
# that lost every n-max 6 arm-pass would land on exactly 300 records, satisfy a gate expecting
# 300, and have its weights deleted; and n-max 6 is the arm that matches llama.cpp #26750, which
# is half the reason this phase exists. The gate now asks the matrix what the arms are and checks
# the shape of the run against them.
#
# The matrix refuses to import when its target gguf is absent, which is the state every skip
# decision runs in once a rung has been staged out. That is why the import has a fallback: the
# strong check is only load-bearing while the weights are present, because that is the only
# moment this gate deletes anything.
gate() {   # $1 = json path, $2 = rung -> "OK n [source]" or "FAIL reason"
  QWEN_QS_TARGET="$2" python3 - "$1" "$N_PROMPTS" "$PASSES" <<'PY'
import collections, importlib, json, sys
path, n_prompts, passes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sys.path.insert(0, 'harness')
try:
    want = [a.name for a in importlib.import_module('matrices.phase_qsmall').ARMS]
    arm_source = "matrix"
except Exception:
    want, arm_source = None, "result (matrix unreadable: target staged out)"
try:
    d = json.load(open(path))
except Exception:
    print("FAIL no readable result"); sys.exit()
recs = d.get('records', [])
got_arms = list(d.get('arms', []))
if want is None:
    want = got_arms
    if not want:
        print("FAIL result declares no arms and the matrix could not be read"); sys.exit()
elif sorted(got_arms) != sorted(want):
    print(f"FAIL arms {sorted(got_arms)} != matrix {sorted(want)}"); sys.exit()
c = collections.Counter((r.get('arm'), r.get('pass')) for r in recs)
expect = {(a, p) for a in want for p in range(1, passes+1)}
missing, extra = expect - set(c), set(c) - expect
if missing or extra:
    if missing: print(f"FAIL {len(missing)}/{len(expect)} arm-passes missing, e.g. {sorted(missing)[:3]}")
    else:       print(f"FAIL arm-passes present that the design does not define: {sorted(extra)[:3]}")
    sys.exit()
bad = {k: v for k, v in c.items() if v != n_prompts}
if bad:
    print(f"FAIL arm-passes with wrong prompt count: {sorted(bad.items())[:3]}"); sys.exit()
inc = d.get('incidents') or []
if inc:
    print(f"FAIL {len(inc)} incident(s): {str(inc[0])[:120]}"); sys.exit()
print(f"OK {len(recs)} [arms from {arm_source}]")
PY
}

records_in() {   # progress lines only; the gate above is what decides anything
  python3 -c "
import json
try: print(len(json.load(open('$1'))['records']))
except Exception: print(0)" 2>/dev/null
}

vram_gb=$(python3 -c "
import sys; sys.path.insert(0,'harness')
import devices as D
try: print(f'{D.get_device($GPU).vram_gb:.1f}')
except Exception: print('0')")
log "GPU $GPU has ${vram_gb} GB"
if [ "$(python3 -c "print(1 if float('$vram_gb')<1 else 0)")" = "1" ]; then
  log "no such GPU - set GPU=<nvidia-smi index>"; exit 1
fi

mkdir -p models/qwen35_9b analysis logs results

for RUNG in $RUNGS; do
  F="${FILE[$RUNG]:-}"
  if [ -z "$F" ]; then log "unknown rung $RUNG; skipping"; continue; fi
  OUT="results/phase_qsmall_${RUNG}.json"
  STAGED="models/qwen35_9b/$F"

  verdict=$(gate "$OUT" "$RUNG")
  if [ "${verdict%% *}" = "OK" ]; then
    log "$RUNG already complete (${verdict#OK }); skipping"
    continue
  fi
  got=$(records_in "$OUT")
  if [ "${got:-0}" -gt 0 ]; then
    log "$RUNG has a PARTIAL result (${got} records; ${verdict}); archiving it and re-running"
    mv "$OUT" "${OUT%.json}.partial.$(date +%s).json"
  fi

  # The card, before the disk. A rung that cannot be held is not worth downloading.
  if [ "$(python3 -c "print(1 if float('$vram_gb') < ${NEED_VRAM_GB[$RUNG]} else 0)")" = "1" ]; then
    log "SKIPPING $RUNG: needs ${NEED_VRAM_GB[$RUNG]} GB VRAM and this card has ${vram_gb}"
    continue
  fi
  headroom=$(python3 -c "print(f'{100*${NEED_VRAM_GB[$RUNG]}/float(\"$vram_gb\"):.0f}')")
  log "=== rung $RUNG (${NEED_VRAM_GB[$RUNG]} GB VRAM = ${headroom} % of the card, ${NEED_DISK_GIB[$RUNG]} GiB on disk) ==="
  [ "$headroom" -ge 88 ] && log "  NOTE: this rung is marginal on this card; an allocation failure here is expected, not a bug"

  SRC=""
  [ -f "$STAGED" ] && SRC="$STAGED"
  if [ -z "$SRC" ]; then
    free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
    need_gb=$(python3 -c "
import math; print(math.ceil(${NEED_DISK_GIB[$RUNG]} + ${DISK_MARGIN_GIB}))")
    if [ "${free_gb:-0}" -lt "$need_gb" ]; then
      log "SKIPPING $RUNG: staging needs ${need_gb} GiB (${NEED_DISK_GIB[$RUNG]} GiB file"
      log "  + ${DISK_MARGIN_GIB} GiB margin) and only ${free_gb} GiB is free."
      continue
    fi
    log "downloading $F (${free_gb} GiB free, needs ${need_gb} GiB)"
    HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download unsloth/Qwen3.5-9B-MTP-GGUF "$F" \
      --local-dir models/qwen35_9b >> logs/phase_qsmall_download.log 2>&1 || {
        log "download of $F FAILED - see logs/phase_qsmall_download.log"; continue; }
    if [ ! -f "$STAGED" ]; then
      log "$F did not arrive as a single file; it is probably split into parts."
      continue
    fi
    SRC="$STAGED"
    actual=$(python3 -c "
import os; print(f'{os.path.getsize(\"$SRC\")/2**30:.2f}')")
    log "staged $F: ${actual} GiB on disk (table says ${NEED_DISK_GIB[$RUNG]})"
  else
    log "reusing $SRC (not downloading)"
  fi
  log "disk after staging: $(df -BG --output=avail . | tail -1 | tr -dc '0-9') GiB free"

  QWEN_QS_TARGET="$RUNG" python3 -u harness/bench.py \
    --matrix phase_qsmall --passes "$PASSES" --gpu "$GPU" --port "$PORT" \
    --settle-floor --out "$OUT" > "logs/phase_qsmall_${RUNG}.log" 2>&1
  rc=$?
  verdict=$(gate "$OUT" "$RUNG")
  log "$RUNG exited rc=$rc; gate says: ${verdict}"

  if [ "$rc" -eq 0 ] && [ "${verdict%% *}" = "OK" ]; then
    ok_reports=1
    python3 harness/analyze.py           "$OUT" > "analysis/phase_qsmall_${RUNG}.txt"            2>&1 || ok_reports=0
    python3 harness/cost_model.py        "$OUT" > "analysis/phase_qsmall_${RUNG}_cost.txt"       2>&1 || ok_reports=0
    python3 harness/divergence_report.py "$OUT" > "analysis/phase_qsmall_${RUNG}_divergence.txt" 2>&1 || ok_reports=0
    python3 harness/pass_stability.py    "$OUT" > "analysis/phase_qsmall_${RUNG}_stability.txt"  2>&1 || ok_reports=0
    if [ "$ok_reports" = "1" ]; then
      log "$RUNG complete; reports written"
    else
      log "$RUNG complete but an analyser FAILED - see analysis/phase_qsmall_${RUNG}*.txt"
      log "  the result file is intact; re-run the analysers by hand after fixing them"
    fi
    # Decided by WHERE the file is, not by whether this invocation downloaded it. The old
    # condition required DOWNLOADED=1, which is false on exactly the retry path: a rung that died
    # mid-run keeps its weights, so the next invocation finds them staged and then declines to
    # clean up after succeeding.
    if [ "$KEEP" = "0" ] && [ "$SRC" = "$STAGED" ]; then
      log "removing $SRC and its staging sidecars (result verified complete)"
      rm -f "$SRC" "$SRC".sha256 "$SRC".sha256sum
      rm -rf "models/qwen35_9b/.cache/huggingface/download/$F".*
    fi
  else
    log "$RUNG INCOMPLETE - keeping weights for a retry; inspect logs/phase_qsmall_${RUNG}.log"
  fi
  log "disk now: $(df -BG --output=avail . | tail -1 | tr -dc '0-9') GiB free"
done

leftover=$(find models/qwen35_9b -maxdepth 1 -type f -name '*.gguf' 2>/dev/null | wc -l)
[ "$leftover" -gt 0 ] && log "NOTE: ${leftover} gguf still staged (incomplete rung, or KEEP=1)"

log "done. rungs:"
for RUNG in $RUNGS; do
  printf "  %-8s %s\n" "$RUNG" "$(gate "results/phase_qsmall_${RUNG}.json" "$RUNG")"
done
