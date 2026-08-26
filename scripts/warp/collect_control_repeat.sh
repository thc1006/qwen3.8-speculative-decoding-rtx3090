#!/usr/bin/env bash
# Collects the control-build repeat from host C and answers the one question it exists for:
# does a given build produce the same output twice?
set -u
cd /home/thc1006/dev/qwen3.8-speculative-decoding-rtx3090 || exit 1
HOST="scc@100.103.103.73"; KEY="$HOME/.ssh/id_ed25519"; RDIR="qwen38-a6000"; EXPECT=150
log() { echo "[$(date -Is)] $*"; }
rsh() { timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=20 -i "$KEY" "$HOST" "$@"; }

while :; do
  n=$(rsh "cd $RDIR && python3 -c \"
import json
try: print(len(json.load(open('results/phase_warp_control_repeat.json'))['records']))
except Exception: print(0)\"" 2>/dev/null | tr -d '\r'); n=${n:-0}
  alive=$(rsh "pgrep -f '[r]un_control_repeat_hostc.sh' >/dev/null && echo yes || echo no" | tr -d '\r')
  log "  control_repeat ${n}/${EXPECT}  driver: ${alive}"
  [ "$n" -ge "$EXPECT" ] && [ "$alive" = "no" ] && break
  [ "$alive" = "no" ] && { log "driver gone at ${n}/${EXPECT}"; break; }
  sleep 150
done

for f in results/phase_warp_control_repeat.json results/phase_warp_control_repeat.records.jsonl \
         logs/warp_control_repeat.log; do
  timeout 600 scp -q -i "$KEY" "$HOST:$RDIR/$f" "$(dirname "$f")/" 2>/dev/null || { log "missing $f"; continue; }
  r=$(rsh "cd $RDIR && sha256sum $f | cut -d' ' -f1" | tr -d '\r'); l=$(sha256sum "$f" | cut -d' ' -f1)
  [ "$r" = "$l" ] && log "  ok  $f" || log "  CHECKSUM MISMATCH $f"
done

python3 - <<'PYEOF' | tee analysis/control_determinism.txt
import json
def load(p):
    d=json.load(open(p))
    return {(r["arm"],r["prompt"],r["pass"]):(r.get("text") or "") for r in d["records"]}
try:
    A=load("results/phase_warp_control.json"); B=load("results/phase_warp_control_repeat.json")
except Exception as e:
    print("  cannot compare:", e); raise SystemExit
W={"baseline":1,"mtp-n2":3,"mtp-n3":4,"mtp-n4":5,"mtp-n5":6,"mtp-n7":8}
print("="*92)
print("DOES ONE BUILD PRODUCE THE SAME OUTPUT TWICE?")
print("="*92)
print("  the same binary, the same card, the same prompts, greedy, run twice\n")
tot_s=tot_n=0
for arm,w in sorted(W.items(), key=lambda x:x[1]):
    ks=[k for k in A if k[0]==arm and k in B]
    same=sum(1 for k in ks if A[k]==B[k])
    tot_s+=same; tot_n+=len(ks)
    flag="" if same==len(ks) else "   <-- NOT REPRODUCIBLE"
    print(f"  w={w:<2} {arm:<11} {same}/{len(ks)} byte-identical between the two runs{flag}")
print(f"\n  overall {tot_s}/{tot_n}")
if tot_s==tot_n:
    print("\n  The build is deterministic. Every difference measured between builds is therefore a")
    print("  build difference, and forced_down2 changing the width-1 baseline while its width-1")
    print("  kernels are byte-identical SASS is a real effect that still needs naming.")
else:
    print("\n  The build is NOT deterministic run to run. Differences attributed to the table edit")
    print("  cannot be separated from this, and every forced-warp comparison in the study needs")
    print("  re-reading against it, starting with the forced_up gates that passed 25/25 and 50/50.")
PYEOF
log "done"
