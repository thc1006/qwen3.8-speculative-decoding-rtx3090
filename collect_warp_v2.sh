#!/usr/bin/env bash
# Brings the four-build intervention home from host C and scores both directions.
set -u
cd /home/thc1006/dev/qwen3.8-speculative-decoding-rtx3090 || exit 1
HOST="scc@100.103.103.73"; KEY="$HOME/.ssh/id_ed25519"; RDIR="qwen38-a6000"; EXPECT=150
BUILDS="control forced_up forced_down2 control2"
log() { echo "[$(date -Is)] $*"; }
rsh() { timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=20 -i "$KEY" "$HOST" "$@"; }
n_of() { rsh "cd $RDIR && python3 -c \"
import json
try: print(len(json.load(open('results/phase_warp_v2_$1.json'))['records']))
except Exception: print(0)\"" 2>/dev/null | tr -d '\r'; }

while :; do
  line=""; done_n=0
  for b in $BUILDS; do n=$(n_of "$b"); n=${n:-0}; line="$line $b=$n"; [ "$n" -ge "$EXPECT" ] && done_n=$((done_n+1)); done
  alive=$(rsh "pgrep -f '[r]un_warp_v2_hostc.sh' >/dev/null && echo yes || echo no" | tr -d '\r')
  log " ${line}  complete ${done_n}/4  driver: ${alive}"
  [ "$done_n" -ge 4 ] && break
  [ "$alive" = "no" ] && { log "driver gone with ${done_n}/4"; break; }
  sleep 180
done

for b in $BUILDS; do
  n=$(n_of "$b"); [ "${n:-0}" -ge "$EXPECT" ] || { log "skipping $b at ${n:-0}/${EXPECT}"; continue; }
  for f in "results/phase_warp_v2_${b}.json" "results/phase_warp_v2_${b}.records.jsonl" "logs/v2_run_${b}.log"; do
    timeout 600 scp -q -i "$KEY" "$HOST:$RDIR/$f" "$(dirname "$f")/" 2>/dev/null || { log "  missing $f"; continue; }
    r=$(rsh "cd $RDIR && sha256sum '$f' | cut -d' ' -f1" | tr -d '\r'); l=$(sha256sum "$f" | cut -d' ' -f1)
    [ "$r" = "$l" ] && log "  ok  $f" || log "  CHECKSUM MISMATCH $f"
  done
  timeout 300 scp -q -i "$KEY" "$HOST:$RDIR/warp_v2/$b/table.txt" \
      "upstream/llamacpp/warp_builds_v2_${b}_table.txt" 2>/dev/null
done
timeout 300 scp -q -i "$KEY" "$HOST:$RDIR/logs/warp_v2_chain.log" logs/ 2>/dev/null

log "scoring both directions against the registered predictions"
python3 harness/warp_intervention.py results/phase_warp_v2_control.json \
    results/phase_warp_v2_forced_up.json results/phase_warp_v2_forced_down2.json \
    > analysis/warp_intervention_v2.txt 2>&1
python3 - <<'PYEOF' | tee -a analysis/warp_intervention_v2.txt
import json
def load(p):
    d=json.load(open(p))
    return {(r["arm"],r["prompt"],r["pass"]):(r.get("text") or "") for r in d["records"]}
try:
    C=load("results/phase_warp_v2_control.json"); C2=load("results/phase_warp_v2_control2.json")
except Exception as e:
    print("  cannot run the guard:", e); raise SystemExit
print("\n=== the guard: two builds from one configure, no source difference ===")
ks=sorted(set(C)&set(C2)); same=sum(1 for k in ks if C[k]==C2[k])
print(f"  control against control2: {same}/{len(ks)} byte-identical")
if same != len(ks):
    print("  *** THE GUARD FAILED. Two builds that differ in nothing produced different output,")
    print("      so no comparison in this set means anything. Nothing above should be read. ***")
else:
    print("  The guard holds, so a difference between control and a forced build is the table.")
PYEOF
sed -n '1,150p' analysis/warp_intervention_v2.txt
log "done"
