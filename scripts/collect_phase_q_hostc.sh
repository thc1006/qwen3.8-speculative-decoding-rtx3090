#!/usr/bin/env bash
# Brings the A6000's quantization ladder home, and checks that nothing was left on a shared host.
#
# Why all four rungs run there rather than only the two the 3090 cannot hold. `c` is a slope in
# milliseconds, so it belongs to the card that produced it; the committed
# `results/phase_q_UD-Q4_K_XL.json` and `..._UD-Q5_K_XL.json` were measured on host A's RTX 3090
# and this repository's standing rule is that only dimensionless quantities and within-host deltas
# cross between machines. Two rungs from one card and two from another is not a ladder. Four rungs
# on one card is, and it gives host A's two a cross-host replication as well, which is what
# `phase_a_hostB.json` is to Phase A.
#
# The results arrive tagged `_hostC` for the same reason: the registry gives a second host its own
# phase entry instead of pooling it into the first host's glob.
#
# This script does not start anything. The run is started on the host by
#   QWEN_HOST_TAG=_hostC GPU=<n> scripts/run_phase_q.sh
# and this only watches, fetches, verifies and cleans.
set -u
cd "$(dirname "$0")/.." || exit 1
HOST="${QWEN_HOSTC:-scc@100.103.103.73}"
KEY="${QWEN_HOSTC_KEY:-$HOME/.ssh/id_ed25519}"
RDIR="${QWEN_HOSTC_DIR:-qwen38-a6000}"
TAG="_hostC"
RUNGS="UD-Q4_K_XL UD-Q5_K_XL UD-Q6_K_XL Q8_0"
POLL="${POLL:-300}"

log() { echo "[$(date -Is)] $*"; }
rsh() { timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=20 -i "$KEY" "$HOST" "$@"; }

if ! rsh true 2>/dev/null; then
  log "FATAL: $HOST is not reachable. Nothing was started and nothing was changed."
  exit 1
fi

n_of() {   # $1 = rung -> record count on the remote, 0 if absent or unreadable
  rsh "cd $RDIR && python3 -c \"
import json
try: print(len(json.load(open('results/phase_q_$1${TAG}.json'))['records']))
except Exception: print(0)\"" 2>/dev/null | tr -d '\r'
}

# Wait for the driver, reporting per-rung progress. A driver that dies is reported rather than
# waited on for ever; whatever finished is still collected below.
while :; do
  line=""; done_n=0
  for r in $RUNGS; do n=$(n_of "$r"); n=${n:-0}; line="$line $r=$n"; [ "$n" -ge 300 ] && done_n=$((done_n+1)); done
  alive=$(rsh "pgrep -f '[r]un_phase_q.sh' >/dev/null && echo yes || echo no" | tr -d '\r')
  log " ${line}  complete ${done_n}/4  driver: ${alive}"
  [ "$done_n" -ge 4 ] && break
  [ "$alive" = "no" ] && { log "driver gone with ${done_n}/4 rungs complete"; break; }
  sleep "$POLL"
done

# Fetch, and verify by hash on both ends rather than trusting scp's exit code.
for r in $RUNGS; do
  n=$(n_of "$r")
  if [ "${n:-0}" -lt 300 ]; then log "skipping $r at ${n:-0}/300"; continue; fi
  for f in "results/phase_q_${r}${TAG}.json" "logs/phase_q_${r}${TAG}.log"; do
    if ! timeout 900 scp -q -i "$KEY" "$HOST:$RDIR/$f" "$(dirname "$f")/" 2>/dev/null; then
      log "  missing $f"; continue
    fi
    rem=$(rsh "cd $RDIR && sha256sum '$f' | cut -d' ' -f1" | tr -d '\r')
    loc=$(sha256sum "$f" | cut -d' ' -f1)
    [ "$rem" = "$loc" ] && log "  ok  $f" || log "  CHECKSUM MISMATCH $f"
  done
done

# The analysers run here, not there: host C needs nothing but python3 and the binary to measure.
for r in $RUNGS; do
  out="results/phase_q_${r}${TAG}.json"
  [ -f "$out" ] || continue
  python3 harness/analyze.py    "$out" > "analysis/phase_q_${r}${TAG}.txt"      2>&1 \
    || log "  analyze.py FAILED on $r; see the report"
  python3 harness/cost_model.py "$out" > "analysis/phase_q_${r}${TAG}_cost.txt" 2>&1 \
    || log "  cost_model.py FAILED on $r; see the report"
done

# Host C is a shared machine. The driver deletes each staged GGUF once its rung passes the
# completeness gate, so anything left here is an incomplete rung or a KEEP=1 run -- either way it
# is 20 to 27 GB of someone else's disk and it gets named.
log "--- what is still on the remote ---"
rsh "cd $RDIR && du -sh models/quant_ladder 2>/dev/null; ls -1 models/quant_ladder/*.gguf 2>/dev/null || echo '  staging area is empty'" | sed 's/^/  /'
rsh "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader" \
  | sed 's/^/  still on the GPU: /' || true

log "collected $(ls -1 results/phase_q_*${TAG}.json 2>/dev/null | wc -l)/4 rungs"
