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

# ---------------------------------------------------------------- public commitments first
# Both of these were stated in llama.cpp #25618 comment 5396293373 as running or queued, so they
# come before the phases that only this study cares about.

# f16 KV control. snick525 showed in that thread that q8_0 KV moves greedy output on its own, so
# every fork position reported so far was taken under a cache that is not output-preserving.
run_phase phase_kv 1 18215 results/phase_kv.json || exit 1
python3 harness/divergence_report.py results/phase_kv.json > analysis/phase_kv_divergence.txt 2>&1
python3 - <<'PYEOF' | tee -a analysis/phase_kv_divergence.txt
import json, collections
W = {"mtp-n2": 3, "mtp-n3": 4, "dflash2-n4": 5, "mtp-n5": 6, "dflash2-n7": 8}
print("\n=== does the width grouping survive f16 KV? ===")
for tag, path, pas in [("q8_0 KV (Phase A)", "results/phase_a.json", 1),
                       ("f16  KV (Phase KV)", "results/phase_kv.json", 1)]:
    try:
        d = json.load(open(path))
    except Exception as e:
        print(f"  {tag}: unreadable ({e})"); continue
    pos = collections.defaultdict(dict)
    for r in d["records"]:
        if r["arm"] in W and r["pass"] == pas and r.get("divergence"):
            v = r["divergence"]
            pos[r["prompt"]][W[r["arm"]]] = "same" if v["identical"] else v["first_diff_char"]
    full = [p for p in pos if len(pos[p]) == 5]
    lo = sum(1 for p in full if len({pos[p][3], pos[p][4]}) == 1)
    hi = sum(1 for p in full if len({pos[p][5], pos[p][6], pos[p][8]}) == 1)
    dif = sum(1 for p in full if len({pos[p][3], pos[p][4]}) == 1
              and len({pos[p][5], pos[p][6], pos[p][8]}) == 1
              and {pos[p][3]} != {pos[p][5]})
    print(f"  {tag}: {len(full)} prompts | w3==w4 on {lo} | w5==w6==w8 on {hi} | "
          f"groups differ on {dif}")
print("  The grouping survives if the f16 row matches the q8_0 row. If it does not, the")
print("  partition was an artefact of the quantized cache and the #25618 comment needs a")
print("  correction posted to that thread.")
PYEOF
log "wrote analysis/phase_kv_divergence.txt"

# The n-max ladder. Phase A fitted k(w) = k0 + c(w-1) on three MTP widths and two DFlash2 widths;
# two points do not fit a line, and the DFlash2 coefficient was reported as meaningless for that
# reason. This runs MTP at 1 through 8 and DFlash2 at 2, 4, 6, 8, so both coefficients are
# fitted rather than asserted. It also produces the mtp-n1 arm that Phase V needs: vLLM's MTP
# only runs at one speculative token, so K=1 is the only depth the two engines share.
run_phase phase_nmax 3 18225 results/phase_nmax.json || exit 1
python3 harness/cost_model.py results/phase_nmax.json > analysis/phase_nmax_cost.txt 2>&1
python3 harness/divergence_report.py results/phase_nmax.json > analysis/phase_nmax_divergence.txt 2>&1
log "wrote analysis/phase_nmax_{cost,divergence}.txt: the k(w) fit on 8 MTP widths, and the"
log "  width-2/7/9 groups that llama.cpp #25618 comment 5396293373 predicts"

run_phase phase_c 3 18220 results/phase_c.json || exit 1

# Phase L is a ladder, not a matrix: context depth sets `-c`, which is a server property, so each
# rung is its own run. Its driver handles the per-rung gating and skips a rung that cannot fit.
log "starting the Phase L depth ladder"
GPU=0 PASSES=3 bash run_phase_l.sh >> logs/phase_l_chain.log 2>&1
log "Phase L ladder returned rc=$? (a rung that does not fit is expected at the top)"
python3 harness/analyze_depth.py > analysis/phase_l_ladder.txt 2>&1
log "wrote analysis/phase_l_ladder.txt"

# Phase M swaps the target for the 35B-A3B MoE. It must clear its replication anchor before any
# of its other arms mean anything, so the check is printed right after the run.
run_phase phase_m 3 18240 results/phase_m.json || exit 1
python3 - <<'PYEOF' | tee -a analysis/phase_m_anchor.txt
import json, statistics as st
d = json.load(open("results/phase_m.json"))
by = {}
for r in d["records"]:
    by.setdefault(r["arm"], []).append(r["decode_tok_s"])
b = st.median(by.get("baseline-moe", [0]))
a = st.median(by.get("moe-draft08b-n8", [0]))
print("REPLICATION ANCHOR (predecessor reported 138.9 -> 77.0 tok/s, -44.6 %)")
print(f"  baseline-moe     {b:7.1f} tok/s")
print(f"  moe-draft08b-n8  {a:7.1f} tok/s   net {(a - b) / b * 100:+.1f} %" if b else "  no baseline")
if b and a:
    delta = (a - b) / b * 100
    print("  anchor holds; the MoE penalty reproduces on this harness" if delta < -25 else
          "  ANCHOR DOES NOT HOLD. The predecessor's loss did not reproduce here, so nothing "
          "else in Phase M should be read as a statement about MoE until this is understood.")
PYEOF

# Phase Q needs disk that Phase M's 22 GB MoE target is sitting on, so the deletion happens here
# rather than inside run_phase_q.sh, and only against a Phase M result that is actually complete.
if [ -f results/phase_m.json ]; then
  mrec=$(records_in results/phase_m.json); mwant=$(expect_for phase_m 3)
  if [ "${mrec:-0}" -ge "${mwant:-1}" ] && [ -f models/moe/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf ]; then
    log "Phase M complete (${mrec}/${mwant}); releasing the 22 GB MoE target"
    rm -f models/moe/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
    log "disk now: $(df -h / | tail -1 | awk '{print $4}') free"
  else
    log "keeping the MoE target: Phase M is ${mrec}/${mwant}"
  fi
fi

# The 24 GB card reaches only the first two rungs of the target ladder, and the second sits at
# 96 % of the card. The driver auto-selects what fits and says what it skipped.
log "starting the Phase Q target-quantization ladder"
GPU=0 PASSES=3 bash run_phase_q.sh >> logs/phase_q_chain.log 2>&1
log "Phase Q returned rc=$?"

log "chain complete. Phase V is next and is not chained: it needs its own virtualenv"
log "  and a 18 GB download. See docs/PHASE_V_DESIGN.md."
