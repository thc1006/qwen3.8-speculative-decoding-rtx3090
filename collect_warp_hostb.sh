#!/usr/bin/env bash
# Brings the host B forced-warp replication home before that machine goes away.
#
# The window is four hours from launch; DEADLINE overrides it. The driver on host B carries its
# own, slightly earlier deadline, which only ever makes it refuse a build sooner - the safe way
# for the two to disagree.
#
# Host B writes results/phase_warp_<build>.json, which is the same name host C used. Pulling those
# into results/ would silently overwrite the A6000 run with the 3090 one; that is exactly how the
# two different results_27572.json files ended up indistinguishable, one of which was already lost
# to the local copy. Everything from host B therefore lands with a _hostB suffix, and the script
# refuses to write over an existing file it did not create.
set -u
cd /home/thc1006/dev/qwen3.8-speculative-decoding-rtx3090 || exit 1

HB="thc1006@100.112.135.98"
RDIR="qwen38-remote"
EXPECT=150
POLL=180
DEADLINE=${DEADLINE:-$(date -d "+4 hours" +%s)}   # host B is available for four more hours

log() { echo "[$(date -Is)] $*"; }
rsh() { timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=20 "$HB" "$@"; }

remote_records() {
  rsh "cd $RDIR && python3 -c \"
import json
try: print(len(json.load(open('results/phase_warp_$1.json'))['records']))
except Exception: print(0)\"" 2>/dev/null | tr -d '\r'
}

pull() {   # $1 = remote path relative to RDIR, $2 = local destination
  timeout 600 scp -q -o BatchMode=yes "$HB:$RDIR/$1" "$2" || return 1
  local r l
  r=$(rsh "cd $RDIR && sha256sum '$1' 2>/dev/null | cut -d' ' -f1" | tr -d '\r')
  l=$(sha256sum "$2" 2>/dev/null | cut -d' ' -f1)
  [ -n "$r" ] && [ "$r" = "$l" ]
}

log "watching host B; hard stop at $(date -d @$DEADLINE '+%H:%M')"
while :; do
  done_n=0; line=""
  for b in control forced_up forced_down; do
    n=$(remote_records "$b"); n=${n:-0}
    line="$line $b=$n"
    [ "$n" -ge "$EXPECT" ] && done_n=$(( done_n + 1 ))
  done
  alive=$(rsh "pgrep -f 'run_warp_hostb.sh' >/dev/null && echo yes || echo no" 2>/dev/null | tr -d '\r')
  log " ${line}  builds complete: ${done_n}/3  driver: ${alive}"
  [ "$done_n" -ge 3 ] && { log "all three builds complete"; break; }
  if [ "$alive" = "no" ]; then
    log "the driver has exited with ${done_n}/3 complete; collecting what exists"
    break
  fi
  now=$(date +%s)
  if [ "$now" -ge $(( DEADLINE - 900 )) ]; then
    log "15 minutes to the deadline; collecting whatever is complete now"
    break
  fi
  sleep "$POLL"
done

# ---------------------------------------------------------------- collect
mkdir -p upstream/llamacpp/warp_builds_hostB
got=0
for b in control forced_up forced_down; do
  n=$(remote_records "$b"); n=${n:-0}
  if [ "$n" -lt "$EXPECT" ]; then
    log "skipping $b: only ${n}/${EXPECT} records, a partial build is not comparable"
    continue
  fi
  for pair in "results/phase_warp_${b}.json|results/phase_warp_${b}_hostB.json" \
              "results/phase_warp_${b}.records.jsonl|results/phase_warp_${b}_hostB.records.jsonl" \
              "logs/warp_${b}.log|logs/warp_${b}_hostB.log" \
              "warp/${b}/table.txt|upstream/llamacpp/warp_builds_hostB/${b}_table.txt"; do
    src="${pair%%|*}"; dst="${pair##*|}"
    if pull "$src" "$dst"; then
      log "  ok  $dst"
    else
      log "  FAILED or checksum mismatch: $src -> $dst"
    fi
  done
  got=$(( got + 1 ))
done
timeout 120 scp -q -o BatchMode=yes "$HB:$RDIR/logs/warp_hostb_chain.log" logs/ 2>/dev/null && log "  ok  logs/warp_hostb_chain.log"

rsh "cd $RDIR && for d in control forced_up forced_down; do
       [ -f warp/\$d/libggml-cuda.so.0.21.0 ] && printf '%-12s %s  %s bytes\n' \$d \
         \$(sha256sum warp/\$d/libggml-cuda.so.0.21.0 | cut -c1-16) \
         \$(stat -c%s warp/\$d/libggml-cuda.so.0.21.0); done" \
  > upstream/llamacpp/warp_builds_hostB/libggml_cuda_hashes.txt 2>/dev/null
log "recorded the host B libggml-cuda.so hashes"

# ---------------------------------------------------------------- release host B
if [ "$got" -eq 3 ]; then
  log "all three collected; removing the build snapshots from host B"
  rsh "rm -rf ~/$RDIR/warp" && log "  removed ~/$RDIR/warp"
  rsh "nvidia-smi -i 0 -rgc >/dev/null 2>&1; nvidia-smi -i 0 -rmc >/dev/null 2>&1; true"
  rsh "nvidia-smi --query-gpu=memory.used,clocks.current.graphics --format=csv,noheader" | sed 's/^/    host B now: /'
else
  log "only ${got}/3 builds collected; leaving host B alone so nothing is destroyed"
fi

# ---------------------------------------------------------------- score
if [ -f results/phase_warp_forced_up_hostB.json ]; then
  log "scoring the 3090 replication"
  python3 harness/warp_intervention.py \
    results/phase_warp_control_hostB.json \
    results/phase_warp_forced_up_hostB.json \
    results/phase_warp_forced_down_hostB.json > analysis/warp_intervention_hostB.txt 2>&1
  log "wrote analysis/warp_intervention_hostB.txt"
  sed -n '1,200p' analysis/warp_intervention_hostB.txt
fi
log "done"
