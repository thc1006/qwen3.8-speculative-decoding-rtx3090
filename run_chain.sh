#!/usr/bin/env bash
# Phase A -> completeness gate -> analyse -> Phase R pre-flight gate -> Phase R -> analyse.
#
# Two hard gates. Neither is advisory:
#
#   Gate 1 — Phase A must be COMPLETE. A crashed run releases no lock, and an earlier version of
#   this script would have read "lock gone" as "finished" and handed partial data to Phase R.
#
#   Gate 2 — the Phase R pre-flight must prove every resource condition applies AND that the
#   power-limit conditions leave the memory clock alone. If a power condition drags the memory
#   P-state with it, that condition varies both resources at once, the elasticity decomposition
#   is invalid for it, and the design must be revised rather than run.
set -u
cd /home/thc1006/dev/qwen3.8-speculative-decoding-rtx3090 || exit 1
mkdir -p analysis logs
log() { echo "[$(date -Is)] $*"; }

EXPECTED_A=875     # 7 arms x 25 prompts x 5 passes

count_field() {   # $1 = json path, $2 = key
  python3 -c "
import json,sys
try:
    d=json.load(open('$1')); print(len(d.get('$2',[])))
except Exception: print(-1)" 2>/dev/null
}

log "waiting for Phase A"
while :; do
  [ -f .gpu-in-use.lock ] || break
  lp=$(sed -n 's/^pid=//p' .gpu-in-use.lock | head -1)
  if [ -z "$lp" ] || ! kill -0 "$lp" 2>/dev/null; then
    log "lock is stale (pid ${lp:-none} gone) — Phase A did not finish cleanly"
    break
  fi
  sleep 20
done

recs=$(count_field results/phase_a.json records)
inc=$(count_field results/phase_a.json incidents)
log "Phase A: ${recs}/${EXPECTED_A} records, ${inc} incidents"

if [ "${recs:-0}" -lt "$EXPECTED_A" ]; then
  log "GATE 1 FAILED: Phase A is incomplete. Not analysing, not starting Phase R."
  log "Investigate logs/phase_a_run.log before continuing."
  exit 1
fi
log "gate 1 passed"

log "analysing Phase A"
python3 harness/analyze.py results/phase_a.json > analysis/phase_a_report.txt 2>&1
sed -n '1,70p' analysis/phase_a_report.txt

log "Phase R pre-flight"
if ! python3 harness/preflight_r.py > logs/preflight_r.log 2>&1; then
  log "GATE 2 FAILED: pre-flight rejected at least one resource condition."
  log "Phase R NOT started. See logs/preflight_r.log"
  tail -45 logs/preflight_r.log
  exit 2
fi
tail -30 logs/preflight_r.log
log "gate 2 passed"

log "launching Phase R (15 arms x 25 prompts x 3 passes)"
python3 -u harness/bench.py --matrix phase_r --passes 3 --port 18150 \
  --out results/phase_r.json > logs/phase_r_run.log 2>&1
rc=$?
log "Phase R exited rc=${rc}"

python3 harness/analyze.py results/phase_r.json > analysis/phase_r_report.txt 2>&1
sed -n '1,80p' analysis/phase_r_report.txt
log "chain complete"
