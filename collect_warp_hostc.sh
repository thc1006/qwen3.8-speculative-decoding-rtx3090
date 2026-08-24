#!/usr/bin/env bash
# Waits out the forced-warp intervention on host C, brings the results home, and leaves the host
# as it was found.
#
# Host C is a shared machine and the card is only free at night, so the standing rule for it is:
# run, transfer back, and remove everything - builds, weights, intermediate files. Deletion is the
# irreversible step, so it happens only after the transferred copy has been verified against the
# remote checksum. A failed transfer keeps the remote data and stops; it does not clean up and
# hope.
set -u
cd /home/thc1006/dev/qwen3.8-speculative-decoding-rtx3090 || exit 1

HOST="scc@100.103.103.73"
KEY="$HOME/.ssh/id_ed25519"
RDIR="qwen38-a6000"
EXPECT=150                       # 6 arms x 25 prompts x 1 pass, from harness/matrices/phase_warp.py
POLL=120
MAX_WAIT=$(( 3 * 3600 ))

log() { echo "[$(date -Is)] $*"; }
rsh() { timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=20 -i "$KEY" "$HOST" "$@"; }

remote_records() {   # $1 = tag -> record count, 0 if unreadable or mid-write
  rsh "cd $RDIR && python3 -c \"
import json
try: print(len(json.load(open('results/phase_warp_$1.json'))['records']))
except Exception: print(0)\"" 2>/dev/null | tr -d '\r'
}

remote_alive() {     # is a bench.py still running there?
  rsh "pgrep -f 'bench.py --matrix phase_warp' >/dev/null && echo yes || echo no" 2>/dev/null | tr -d '\r'
}

# ---------------------------------------------------------------- wait for forced_down
log "waiting for forced_down on host C (expecting ${EXPECT} records)"
waited=0
while :; do
  n=$(remote_records forced_down); n=${n:-0}
  alive=$(remote_alive)
  log "  forced_down ${n}/${EXPECT}  bench running: ${alive}"
  [ "$n" -ge "$EXPECT" ] && [ "$alive" = "no" ] && { log "forced_down complete"; break; }
  if [ "$alive" = "no" ] && [ "$n" -lt "$EXPECT" ]; then
    log "GATE FAILED: bench.py is gone but only ${n}/${EXPECT} records exist."
    log "Leaving host C untouched so the partial run can be inspected."
    exit 1
  fi
  waited=$(( waited + POLL ))
  [ "$waited" -ge "$MAX_WAIT" ] && { log "timed out after ${MAX_WAIT}s; host C left untouched"; exit 1; }
  sleep "$POLL"
done

# ---------------------------------------------------------------- transfer, then verify
log "transferring"
FILES="results/phase_warp_forced_down.json results/phase_warp_forced_down.records.jsonl
       logs/warp_forced_down.log logs/warp_run.log logs/warp_builds.log"
for f in $FILES; do
  timeout 600 scp -q -i "$KEY" "$HOST:$RDIR/$f" "$(dirname "$f")/" || { log "scp of $f FAILED; host C untouched"; exit 1; }
done

log "verifying every transferred file against the remote checksum"
ok=1
for f in $FILES; do
  r=$(rsh "cd $RDIR && sha256sum $f 2>/dev/null | cut -d' ' -f1" | tr -d '\r')
  l=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)
  if [ -n "$r" ] && [ "$r" = "$l" ]; then
    log "  ok  $(basename "$f")"
  else
    log "  MISMATCH $(basename "$f")  remote=${r:-none} local=${l:-none}"; ok=0
  fi
done
[ "$ok" -eq 1 ] || { log "checksum mismatch; NOT cleaning up host C"; exit 1; }

# the three build trees must also come home: they are the evidence that the intervention was
# what it claims to be, and they are about to be deleted there
log "collecting the build provenance before it is deleted"
mkdir -p upstream/llamacpp/warp_builds
for d in control forced_up forced_down; do
  timeout 300 scp -q -i "$KEY" "$HOST:$RDIR/warp/$d/table.txt" "upstream/llamacpp/warp_builds/${d}_table.txt" \
    && log "  got ${d}_table.txt"
done
rsh "cd $RDIR && for d in control forced_up forced_down; do
       printf '%-12s %s  %s bytes\n' \$d \$(sha256sum warp/\$d/libggml-cuda.so.0.21.0 | cut -c1-16) \$(stat -c%s warp/\$d/libggml-cuda.so.0.21.0);
     done" > upstream/llamacpp/warp_builds/libggml_cuda_hashes.txt 2>/dev/null
log "  recorded libggml-cuda.so hashes (llama-server itself is a 17 KB wrapper and carries no kernel)"

# ---------------------------------------------------------------- release the host
log "restoring GPU state and removing everything this study put on host C"
rsh "nvidia-smi -i 0 -rgc >/dev/null 2>&1; nvidia-smi -i 0 -rmc >/dev/null 2>&1; true"
rsh "cd $RDIR 2>/dev/null && du -sh . 2>/dev/null | cut -f1" | sed 's/^/    occupied before: /'
rsh "rm -rf ~/$RDIR" && log "  removed ~/$RDIR"
rsh "ls -d ~/$RDIR 2>/dev/null && echo STILL_THERE || echo gone" | sed 's/^/    /'
rsh "nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.current.graphics --format=csv,noheader" | sed 's/^/    host C now: /'

# ---------------------------------------------------------------- score both directions
log "scoring the intervention against the registered outcomes"
python3 harness/warp_intervention.py \
  results/phase_warp_control.json \
  results/phase_warp_forced_up.json \
  results/phase_warp_forced_down.json > analysis/warp_intervention.txt 2>&1
log "wrote analysis/warp_intervention.txt"
sed -n '1,200p' analysis/warp_intervention.txt

log "done"
