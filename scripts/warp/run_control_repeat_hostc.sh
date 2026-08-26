#!/usr/bin/env bash
# Runs the control build a second time and compares it against its own first run.
#
# Every conclusion drawn from the forced-warp builds assumes that a given build produces the same
# output twice. Nothing has tested that. The forced_down2 result makes it urgent: its width-1
# kernels are byte-identical SASS to the control's and its baseline still differs on 23 of 25
# prompts, which is either a build effect this study cannot yet name or run-to-run
# nondeterminism, and those two have opposite consequences for everything above.
#
# No rebuild: warp/control is the binary the first run used.
set -u
cd "$HOME/qwen38-a6000" || exit 1
OUT="results/phase_warp_control_repeat.json"
EXPECT=150
log() { echo "[$(date -Is)] $*"; }
records_in() { python3 -c "
import json
try: print(len(json.load(open('$1'))['records']))
except Exception: print(0)" 2>/dev/null; }

have=$(records_in "$OUT")
[ "${have:-0}" -ge "$EXPECT" ] && { log "already complete (${have}/${EXPECT})"; exit 0; }

h=$(sha256sum warp/control/libggml-cuda.so.0.21.0 | cut -c1-16)
log "re-running the control build, libggml-cuda $h (unchanged since the first run)"
QWEN_WARP_BUILD=control python3 -u harness/bench.py --matrix phase_warp --passes 1 \
    --port 18400 --settle-floor --allow-non-stock --out "$OUT" > logs/warp_control_repeat.log 2>&1
rc=$?
have=$(records_in "$OUT")
log "exited rc=$rc with ${have}/${EXPECT} records"
