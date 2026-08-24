#!/usr/bin/env bash
# Collects the forced_down2 run already in flight on host B.
#
# The chain script sent that job through a wrapper carrying a 120 s timeout, so the ssh died and
# the collector ran its pull step against a file that did not exist yet and declared itself done.
# The remote job was never affected and is still running. This picks it up.
set -u
cd /home/thc1006/dev/qwen3.8-speculative-decoding-rtx3090 || exit 1
HB="thc1006@100.112.135.98"; EXPECT=150
DEADLINE=$(date -d "09:35" +%s)
log() { echo "[$(date -Is)] $*"; }
rsh() { timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=20 "$HB" "$@"; }

while :; do
  n=$(rsh "cd qwen38-remote && python3 -c \"
import json
try: print(len(json.load(open('results/phase_warp_forced_down2.json'))['records']))
except Exception: print(0)\"" 2>/dev/null | tr -d '\r'); n=${n:-0}
  alive=$(rsh "pgrep -f '[r]un_warp_down2_hostb.sh' >/dev/null && echo yes || echo no" | tr -d '\r')
  log "  forced_down2 ${n}/${EXPECT}  driver: ${alive}"
  [ "$n" -ge "$EXPECT" ] && [ "$alive" = "no" ] && break
  if [ "$alive" = "no" ]; then log "driver gone at ${n}/${EXPECT}; collecting what exists"; break; fi
  [ "$(date +%s)" -ge "$DEADLINE" ] && { log "deadline"; break; }
  sleep 150
done

for pair in "results/phase_warp_forced_down2.json|results/phase_warp_forced_down2_hostB.json" \
            "results/phase_warp_forced_down2.records.jsonl|results/phase_warp_forced_down2_hostB.records.jsonl" \
            "logs/warp_forced_down2.log|logs/warp_forced_down2_hostB.log" \
            "warp/forced_down2/table.txt|upstream/llamacpp/warp_builds_hostB/forced_down2_table.txt"; do
  src="${pair%%|*}"; dst="${pair##*|}"
  timeout 600 scp -q -o BatchMode=yes "$HB:qwen38-remote/$src" "$dst" 2>/dev/null || { log "  missing $src"; continue; }
  r=$(rsh "cd qwen38-remote && sha256sum '$src' | cut -d' ' -f1" | tr -d '\r')
  l=$(sha256sum "$dst" | cut -d' ' -f1)
  [ "$r" = "$l" ] && log "  ok  $dst" || log "  CHECKSUM MISMATCH $dst"
done

if [ -f results/phase_warp_forced_down2_hostB.json ]; then
  log "scoring the 3090 rebuild against Correction 6"
  python3 harness/warp_intervention.py results/phase_warp_control_hostB.json \
      results/phase_warp_forced_up_hostB.json results/phase_warp_forced_down2_hostB.json \
      > analysis/warp_intervention_down2_hostB.txt 2>&1
  python3 - <<'PYEOF' | tee -a analysis/warp_intervention_down2_hostB.txt
import json
def load(p):
    d=json.load(open(p))
    return {(r["arm"],r["prompt"],r["pass"]):(r.get("text") or "") for r in d["records"]}
try:
    C=load("results/phase_warp_control_hostB.json"); D=load("results/phase_warp_forced_down2_hostB.json")
except Exception as e:
    print("  cannot score:", e); raise SystemExit
W={"baseline":1,"mtp-n2":3,"mtp-n3":4,"mtp-n4":5,"mtp-n5":6,"mtp-n7":8}
print("\n=== Correction 6 on the 3090, the same gates as the A6000 ===")
for arm,w in sorted(W.items(), key=lambda x:x[1]):
    ks=[k for k in C if k[0]==arm and k in D]
    same=sum(1 for k in ks if D[k]==C[k])
    want = "identical to control" if w in (1,5,6,8) else "must change"
    ok = (same==len(ks)) if w in (1,5,6,8) else (same<len(ks))
    print(f"  w={w:<2} {arm:<11} {same}/{len(ks)} identical   registered: {want:<22} "
          f"{'PASS' if ok else '*** FAIL ***'}")
PYEOF
  sed -n '1,120p' analysis/warp_intervention_down2_hostB.txt
fi
log "done"
