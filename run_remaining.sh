#!/usr/bin/env bash
# Runs the remaining phases in sequence, each gated on the previous one being complete.
#
# A phase only starts if the one before it wrote every record it promised. A crashed or
# half-finished phase stops the chain instead of handing partial data to the next stage, which
# is what happened when Phase A died after its last record and the completeness gate correctly
# refused to start Phase R.
set -u
cd /home/thc1006/dev/qwen3.8-speculative-decoding-rtx3090 || exit 1
mkdir -p analysis logs results

log() { echo "[$(date -Is)] $*"; }

records_in() {
  python3 -c "
import json
try: print(len(json.load(open('$1'))['records']))
except Exception: print(0)" 2>/dev/null
}

expect_for() {   # arms x prompts x passes, without importing the matrix
  python3 -c "
import sys, os; sys.path.insert(0,'harness'); sys.path.insert(0,'harness/matrices')
import prompts as P, importlib
m = importlib.import_module('$1')
print(len(m.ARMS) * len(P.PROMPTS) * $2)" 2>/dev/null
}

wait_for_lock_release() {
  while [ -f .gpu-in-use.lock ]; do
    lp=$(sed -n 's/^pid=//p' .gpu-in-use.lock | head -1)
    if [ -z "$lp" ] || ! kill -0 "$lp" 2>/dev/null; then
      log "lock is stale (pid ${lp:-none}); clearing"
      rm -f .gpu-in-use.lock
      break
    fi
    sleep 30
  done
}

analyse() {   # $1 = result json, $2 = tag
  python3 harness/analyze.py     "$1" > "analysis/$2_report.txt"    2>&1
  python3 harness/cost_model.py  "$1" > "analysis/$2_cost.txt"      2>&1
  python3 harness/divergence_report.py "$1" > "analysis/$2_divergence.txt" 2>&1
  log "wrote analysis/$2_{report,cost,divergence}.txt"
}

run_phase() {   # $1 matrix, $2 passes, $3 port, $4 out, $5 extra args
  local matrix="$1" passes="$2" port="$3" out="$4" extra="${5:-}"
  local want; want=$(expect_for "$matrix" "$passes")
  local have; have=$(records_in "$out")
  if [ "${have:-0}" -ge "${want:-1}" ]; then
    log "$matrix already complete (${have}/${want}); skipping"
    return 0
  fi
  [ "${have:-0}" -gt 0 ] && { log "$matrix has a partial result (${have}/${want}); archiving"; \
      mv "$out" "${out%.json}.partial.$(date +%s).json"; }

  wait_for_lock_release
  log "starting $matrix: expecting ${want} records"
  # shellcheck disable=SC2086
  python3 -u harness/bench.py --matrix "$matrix" --passes "$passes" --port "$port" \
      --out "$out" $extra > "logs/${matrix}_run.log" 2>&1
  local rc=$?
  have=$(records_in "$out")
  log "$matrix exited rc=$rc with ${have}/${want} records"
  if [ "$rc" -ne 0 ] || [ "${have:-0}" -lt "${want:-1}" ]; then
    log "GATE FAILED on $matrix. Chain stops here; inspect logs/${matrix}_run.log"
    return 1
  fi
  analyse "$out" "$matrix"
  return 0
}

log "waiting for whatever holds the lock now"
wait_for_lock_release

# Phase R2 is normally already running when this script starts; the wait above covers it.
if [ -f results/phase_r2.json ]; then
  want=$(expect_for phase_r2 3); have=$(records_in results/phase_r2.json)
  log "phase_r2: ${have}/${want}"
  if [ "${have:-0}" -ge "${want:-1}" ]; then
    analyse results/phase_r2.json phase_r2
    python3 harness/elasticity.py results/phase_r2.json > analysis/phase_r2_elasticity.txt 2>&1
    log "wrote analysis/phase_r2_elasticity.txt"
  else
    log "GATE FAILED: phase_r2 incomplete. Not starting Phase C."
    exit 1
  fi
fi

run_phase phase_c 3 18220 results/phase_c.json || exit 1

log "chain complete"
