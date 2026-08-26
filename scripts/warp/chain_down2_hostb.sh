#!/usr/bin/env bash
# Runs the corrected forced-down direction on host B once its three-build chain is done.
#
# Host B is running the original forced-down, whose 1-4 row includes width 1 and therefore moves
# every arm's drafter. That run is not wasted: reproducing the same pattern on a second device is
# what confirms the drafter account rather than leaving it as an inference from one A6000. But the
# direction it was meant to measure still needs the corrected build, and host B goes away at 09:35.
set -u
cd /home/thc1006/dev/qwen3.8-speculative-decoding-rtx3090 || exit 1
HB="thc1006@100.112.135.98"
DEADLINE=$(date -d "09:35" +%s)
NEED=$(( 50 * 60 ))     # one build plus one 150-record run, with room for the thermal gate

log() { echo "[$(date -Is)] $*"; }
rsh() { timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=20 "$HB" "$@"; }

log "waiting for the host B three-build chain to finish"
while rsh "pgrep -f '[r]un_warp_hostb.sh' >/dev/null && echo yes || echo no" | grep -q yes; do
  n=$(rsh "cd qwen38-remote && for b in control forced_up forced_down; do
        python3 -c \"
import json
try: print(len(json.load(open('results/phase_warp_\$b.json'))['records']), end=' ')
except Exception: print(0, end=' ')\"; done" 2>/dev/null)
  log "  builds: ${n:-?}"
  sleep 180
done
log "chain finished"

left=$(( DEADLINE - $(date +%s) ))
if [ "$left" -lt "$NEED" ]; then
  log "only ${left}s left before host B goes away and forced_down2 needs ~${NEED}s. Not starting."
  exit 1
fi

log "shipping and running the corrected build"
scp -q -o BatchMode=yes "$(dirname "$0")/run_warp_down2_hostb.sh" "$HB:~/qwen38-remote/" || { log "scp failed"; exit 1; }
# NOT through rsh(): that wrapper carries a 120 s timeout for short status queries, and this
# job builds and then runs 150 records. Sending it through rsh killed the ssh at 120 s and
# reported rc=124 as if the run had failed, while the remote job carried on regardless and its
# result had nobody left to collect it. Detached, then polled.
timeout 120 ssh -o BatchMode=yes "$HB" \
  "cd qwen38-remote && nohup bash run_warp_down2_hostb.sh > logs/warp_down2_chain.log 2>&1 & echo started"
log "forced_down2 launched detached on host B"
while rsh "pgrep -f '[r]un_warp_down2_hostb.sh' >/dev/null && echo yes || echo no" | grep -q yes; do
  n=$(rsh "cd qwen38-remote && python3 -c \"
import json
try: print(len(json.load(open('results/phase_warp_forced_down2.json'))['records']))
except Exception: print(0)\"" 2>/dev/null | tr -d '\r')
  log "  forced_down2 ${n:-0}/150"
  [ "$(date +%s)" -ge "$DEADLINE" ] && { log "deadline reached while it ran"; break; }
  sleep 180
done
rc=0
log "forced_down2 on host B finished"

for f in results/phase_warp_forced_down2.json results/phase_warp_forced_down2.records.jsonl \
         logs/warp_forced_down2.log logs/warp_down2_chain.log; do
  scp -q -o BatchMode=yes "$HB:~/qwen38-remote/$f" "$(dirname "$f")/$(basename "${f%.*}")_hostB.${f##*.}" 2>/dev/null \
    && log "  pulled $(basename "$f")"
done
scp -q -o BatchMode=yes "$HB:~/qwen38-remote/warp/forced_down2/table.txt" \
    upstream/llamacpp/warp_builds_hostB/forced_down2_table.txt 2>/dev/null
log "done"
