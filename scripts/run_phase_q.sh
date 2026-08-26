#!/usr/bin/env bash
# Phase Q driver - the Qwen3.8-27B target-quantization ladder, disk-staged.
#
# The four rungs total ~93 GB of GGUF and this host has far less free disk, so each rung is
# downloaded, measured, verified complete, and only then deleted. Deletion is the one
# irreversible step here, so it is gated on the result file actually being a complete run:
# a rung whose run died halfway keeps its weights so it can be retried without re-downloading
# 20 GB.
#
# Two numbers, two meanings. An earlier version of this script kept ONE table and used it for
# both, which is what made it wrong:
#
#   NEED_VRAM_GB  what the rung costs on the card (weights + 8K q8_0 KV + ~1.9 GB compute
#                 buffer). Decides which rungs this GPU can hold at all.
#   NEED_DISK_GIB what the GGUF costs on disk, in GiB because that is the unit `df -BG`
#                 answers in. Decides whether staging will fit.
#
# On 2026-08-26 05:34 the old script skipped UD-Q5_K_XL saying it needed 33 GB free when 28 GB
# was available. The file is 19.44 GiB. It had taken 23.1 - a VRAM figure, in decimal GB -
# added 10, and compared the result against GiB. Disk was never the constraint; the rung was
# lost for the night. Both tables below are now measured, not derived from each other.
#
# | rung       | file GiB | +8K KV +buffer -> VRAM GB | 24 GB | 48 GB |
# |------------|---------:|--------------------------:|:-----:|:-----:|
# | UD-Q4_K_XL |    16.35 |                     19.74 | yes   | yes   |
# | UD-Q5_K_XL |    19.44 |                     23.06 | 89 %  | yes   |
# | UD-Q6_K_XL |    23.56 |                     27.48 | no    | yes   |
# | Q8_0       |    27.05 |                     31.23 | no    | yes   |
#
# UD-Q5_K_XL is 89 %, not the 96 % an earlier header claimed; that figure came from comparing a
# decimal-GB requirement against the binary MiB nvidia-smi reports. BF16 is 52.17 GB and fits on
# neither card; the bf16 anchor comes from phase_qsmall instead. See harness/matrices/phase_q.py,
# which carries the same table and is the source these were reconciled against.
#
# Environment:
#   GPU=<n>      nvidia-smi index of the card to use (default 0)
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
DISK_MARGIN_GIB="${DISK_MARGIN_GIB:-5}"   # results, server logs, the downloader's partial file

declare -A FILE=(
  [UD-Q4_K_XL]=Qwen3.8-27B-UD-Q4_K_XL.gguf
  [UD-Q5_K_XL]=Qwen3.8-27B-UD-Q5_K_XL.gguf
  [UD-Q6_K_XL]=Qwen3.8-27B-UD-Q6_K_XL.gguf
  [Q8_0]=Qwen3.8-27B-Q8_0.gguf
)
# What the rung costs on the card. Selects rungs.
declare -A NEED_VRAM_GB=(
  [UD-Q4_K_XL]=19.8 [UD-Q5_K_XL]=23.1 [UD-Q6_K_XL]=27.5 [Q8_0]=31.3
)
# What the GGUF costs on disk, GiB. UD-Q5_K_XL is measured (20876938144 B); the others are the
# published decimal sizes converted, and are checked against the file once it lands.
declare -A NEED_DISK_GIB=(
  [UD-Q4_K_XL]=16.35 [UD-Q5_K_XL]=19.44 [UD-Q6_K_XL]=23.56 [Q8_0]=27.05
)

log() { echo "[$(date -Is)] $*"; }

N_PROMPTS=$(python3 -c "
import sys; sys.path.insert(0,'harness'); import prompts as P; print(len(P.PROMPTS))")
case "${N_PROMPTS}" in
  ''|*[!0-9]*|0)
    log "FATAL: prompt count came out as '${N_PROMPTS}'; a zero here passes every gate."; exit 1 ;;
esac
log "prompt set: ${N_PROMPTS} prompts, ${PASSES} passes"

# Completeness gate. The old gate was `len(records) >= EXPECTED`, which a run could satisfy with
# the wrong shape - one arm measured twelve times reaches 300 records just as four arms in three
# passes do - and satisfying it deletes 20 GB. This checks the shape, and checks it against the
# arm list the MATRIX declares rather than against a constant kept in this file: a matrix that
# grows a fifth width would otherwise be called complete at four arms' worth of records.
# It also refuses a run that logged incidents, because this repo does not keep results it would
# have to disown, and re-running is much cheaper while the weights are still on disk.
gate() {   # $1 = json path, $2 = rung -> prints "OK n" or "FAIL reason"
  QWEN_Q_TARGET="$2" python3 - "$1" "$N_PROMPTS" "$PASSES" <<'PY'
import collections, importlib, json, sys
path, n_prompts, passes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sys.path.insert(0, 'harness')
# The matrix REFUSES TO IMPORT when its target gguf is absent, which is exactly the state this
# gate runs in every time it is asked about a rung whose weights were already deleted. Binding
# the gate to the matrix without this fallback made a finished rung fail its own gate, get
# archived as partial, and start re-downloading 20 GB -- observed on 2026-08-26.
# The strong check is only load-bearing while the weights are still present, because that is
# the only moment this gate deletes anything; when they are gone the gate can only skip.
try:
    want = [a.name for a in importlib.import_module('matrices.phase_q').ARMS]
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
    # Falling back to what bench.py recorded from the matrix at run time. This still checks the
    # shape of the run against its own declared arms; what it cannot notice is a matrix that has
    # GROWN an arm since the run, which would make a complete-looking result actually short.
    want = got_arms
    if not want:
        print("FAIL result declares no arms and the matrix could not be read"); sys.exit()
elif sorted(got_arms) != sorted(want):
    print(f"FAIL arms {sorted(got_arms)} != matrix {sorted(want)}"); sys.exit()
# Compare against the explicit product, not against its size. A count of len(want)*passes is
# also what one arm repeated that many times produces, and that shape reaches the same total
# record count as a real run - which is exactly how the previous gate was fooled.
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

records_in() {   # $1 = json path -> record count, 0 if unreadable; for progress lines only
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

if [ -z "${RUNGS:-}" ]; then
  RUNGS=""
  for r in UD-Q4_K_XL UD-Q5_K_XL UD-Q6_K_XL Q8_0; do
    if [ "$(python3 -c "print(1 if float('$vram_gb') >= ${NEED_VRAM_GB[$r]} else 0)")" = "1" ]; then
      RUNGS="$RUNGS $r"
    fi
  done
  log "auto-selected rungs for a ${vram_gb} GB card:${RUNGS}"
fi

for RUNG in $RUNGS; do
  F="${FILE[$RUNG]}"
  OUT="results/phase_q_${RUNG}.json"
  STAGED="models/quant_ladder/$F"

  verdict=$(gate "$OUT" "$RUNG")
  if [ "${verdict%% *}" = "OK" ]; then
    log "$RUNG already complete (${verdict#OK }) - skipping"
    continue
  fi
  got=$(records_in "$OUT")
  if [ "${got:-0}" -gt 0 ]; then
    log "$RUNG has a PARTIAL result (${got} records; ${verdict}); archiving it and re-running"
    mv "$OUT" "${OUT%.json}.partial.$(date +%s).json"
  fi

  log "=== rung $RUNG (${NEED_VRAM_GB[$RUNG]} GB VRAM, ${NEED_DISK_GIB[$RUNG]} GiB on disk) ==="

  # Reuse a copy this repo already holds rather than downloading a second one. Which directory it
  # came from decides, later, whether it may be deleted: models/target is the shared target for
  # every other phase, models/quant_ladder is this script's staging area.
  SRC=""
  for cand in "models/target/$F" "$STAGED"; do
    [ -f "$cand" ] && { SRC="$cand"; break; }
  done
  if [ -z "$SRC" ]; then
    free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
    need_gb=$(python3 -c "
import math; print(math.ceil(${NEED_DISK_GIB[$RUNG]} + ${DISK_MARGIN_GIB}))")
    if [ "${free_gb:-0}" -lt "$need_gb" ]; then
      log "SKIPPING $RUNG: staging needs ${need_gb} GiB free (${NEED_DISK_GIB[$RUNG]} GiB file"
      log "  + ${DISK_MARGIN_GIB} GiB margin) and only ${free_gb} GiB is available."
      continue
    fi
    log "downloading $F (${free_gb} GiB free, needs ${need_gb} GiB)"
    HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download unsloth/Qwen3.8-27B-GGUF "$F" \
      --local-dir models/quant_ladder >> logs/phase_q_download.log 2>&1 || {
        log "download of $F FAILED - see logs/phase_q_download.log"; continue; }
    # unsloth ships the larger quants split into `-00001-of-000NN` parts. The download above
    # fetches only the single-file name, so a split rung would land with nothing to load.
    if [ ! -f "$STAGED" ]; then
      log "$F did not arrive as a single file; it is probably split into parts."
      log "  check the repo's file list and pass the first part to llama-server instead."
      continue
    fi
    SRC="$STAGED"
    # The disk table above is what the guard trusts; check it against what actually landed, so a
    # rung whose published size moves is caught here rather than by a full root filesystem.
    actual=$(python3 -c "
import os; print(f'{os.path.getsize(\"$SRC\")/2**30:.2f}')")
    log "staged $F: ${actual} GiB on disk (table says ${NEED_DISK_GIB[$RUNG]})"
  else
    log "reusing $SRC (not downloading)"
  fi
  log "disk after staging: $(df -BG --output=avail / | tail -1 | tr -dc '0-9') GiB free"

  QWEN_Q_TARGET="$RUNG" python3 -u harness/bench.py \
    --matrix phase_q --passes "$PASSES" --gpu "$GPU" --port "$PORT" \
    --settle-floor --out "$OUT" > "logs/phase_q_${RUNG}.log" 2>&1
  rc=$?
  verdict=$(gate "$OUT" "$RUNG")
  log "$RUNG exited rc=$rc; gate says: ${verdict}"

  if [ "$rc" -eq 0 ] && [ "${verdict%% *}" = "OK" ]; then
    # A failing analyser used to be silent: its traceback went into the .txt where a report
    # belongs, and the weights were deleted anyway. Reports are cheap to regenerate from the
    # result file, but only if someone knows they are missing.
    ok_reports=1
    python3 harness/analyze.py    "$OUT" > "analysis/phase_q_${RUNG}.txt"      2>&1 || ok_reports=0
    python3 harness/cost_model.py "$OUT" > "analysis/phase_q_${RUNG}_cost.txt" 2>&1 || ok_reports=0
    if [ "$ok_reports" = "1" ]; then
      log "$RUNG complete; reports written"
    else
      log "$RUNG complete but an analyser FAILED - see analysis/phase_q_${RUNG}*.txt"
      log "  the result file is intact; re-run the analysers by hand after fixing them"
    fi
    # Deletion is decided by WHERE the file is, not by whether this invocation downloaded it.
    # The old condition also required DOWNLOADED=1, which is false on exactly the retry path -
    # a rung that died mid-run keeps its weights, so the next invocation finds them already
    # staged, sets DOWNLOADED=0, and leaves 20 GB behind after succeeding.
    if [ "$KEEP" = "0" ] && [ "$SRC" = "$STAGED" ]; then
      log "removing $SRC and its staging sidecars (result verified complete)"
      rm -f "$SRC" "$SRC".sha256 "$SRC".sha256sum
      rm -rf "models/quant_ladder/.cache/huggingface/download/$F".*
    elif [ "$SRC" != "$STAGED" ]; then
      log "keeping $SRC (shared target, not this script's to delete)"
    fi
  else
    log "$RUNG INCOMPLETE - keeping weights for a retry; inspect logs/phase_q_${RUNG}.log"
  fi
  log "disk now: $(df -BG --output=avail / | tail -1 | tr -dc '0-9') GiB free"
done

# Nothing should be left in the staging area once every rung has been accounted for.
leftover=$(find models/quant_ladder -maxdepth 1 -type f -name '*.gguf' 2>/dev/null | wc -l)
[ "$leftover" -gt 0 ] && log "NOTE: ${leftover} gguf still staged in models/quant_ladder (incomplete rung, or KEEP=1)"

log "ladder finished"
ls -1 analysis/phase_q_*.txt 2>/dev/null || log "no rung completed"
