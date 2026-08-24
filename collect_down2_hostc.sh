#!/usr/bin/env bash
# Brings forced_down2 home from host C and scores it against the predictions registered in
# PREREGISTRATION.md Correction 6, which were written before this run finished.
set -u
cd /home/thc1006/dev/qwen3.8-speculative-decoding-rtx3090 || exit 1
HOST="scc@100.103.103.73"; KEY="$HOME/.ssh/id_ed25519"; RDIR="qwen38-a6000"; EXPECT=150
log() { echo "[$(date -Is)] $*"; }
rsh() { timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=20 -i "$KEY" "$HOST" "$@"; }

while :; do
  n=$(rsh "cd $RDIR && python3 -c \"
import json
try: print(len(json.load(open('results/phase_warp_forced_down2.json'))['records']))
except Exception: print(0)\"" 2>/dev/null | tr -d '\r'); n=${n:-0}
  # the bracket keeps pgrep from matching the pattern in its own remote command line
  alive=$(rsh "pgrep -f '[r]un_warp_down2_hostc.sh' >/dev/null && echo yes || echo no" 2>/dev/null | tr -d '\r')
  log "  forced_down2 ${n}/${EXPECT}  driver: ${alive}"
  [ "$n" -ge "$EXPECT" ] && [ "$alive" = "no" ] && break
  if [ "$alive" = "no" ] && [ "$n" -lt "$EXPECT" ]; then
    log "driver gone with ${n}/${EXPECT}; leaving host C alone"; exit 1
  fi
  sleep 120
done
log "complete; transferring"
for f in results/phase_warp_forced_down2.json results/phase_warp_forced_down2.records.jsonl \
         logs/warp_forced_down2.log logs/warp_down2_chain.log; do
  timeout 600 scp -q -i "$KEY" "$HOST:$RDIR/$f" "$(dirname "$f")/" || { log "scp of $f failed"; exit 1; }
  r=$(rsh "cd $RDIR && sha256sum $f | cut -d' ' -f1" | tr -d '\r'); l=$(sha256sum "$f" | cut -d' ' -f1)
  [ "$r" = "$l" ] && log "  ok  $(basename "$f")" || { log "  CHECKSUM MISMATCH $f"; exit 1; }
done
timeout 300 scp -q -i "$KEY" "$HOST:$RDIR/warp/forced_down2/table.txt" \
    upstream/llamacpp/warp_builds/forced_down2_table.txt 2>/dev/null

log "scoring against Correction 6"
python3 harness/warp_intervention.py results/phase_warp_control.json \
    results/phase_warp_forced_up.json results/phase_warp_forced_down2.json \
    > analysis/warp_intervention_down2.txt 2>&1
python3 - <<'PYEOF' | tee -a analysis/warp_intervention_down2.txt
import json
# Correction 6 registered three gates for this build before the data existed.
def load(p):
    d=json.load(open(p))
    return {(r["arm"],r["prompt"],r["pass"]):(r.get("text") or "") for r in d["records"]}
C=load("results/phase_warp_control.json"); D=load("results/phase_warp_forced_down2.json")
W={"baseline":1,"mtp-n2":3,"mtp-n3":4,"mtp-n4":5,"mtp-n5":6,"mtp-n7":8}
print("\n=== Correction 6, registered before this run ===")
for arm,w in sorted(W.items(), key=lambda x:x[1]):
    ks=[k for k in C if k[0]==arm and k in D]
    same=sum(1 for k in ks if D[k]==C[k])
    if w in (1,5,6,8):
        ok = same==len(ks); want="identical to control"
    else:
        ok = same<len(ks);  want="must change"
    print(f"  w={w:<2} {arm:<11} {same}/{len(ks)} identical   registered: {want:<22} "
          f"{'PASS' if ok else '*** FAIL ***'}")
print("  w=1 moving would mean the drafter account is wrong. w=5,6,8 moving would mean the")
print("  rebuild is no cleaner than the first one. w=3,4 not moving would mean the edit never")
print("  reached the kernel.")
PYEOF
sed -n '1,140p' analysis/warp_intervention_down2.txt
log "done"
