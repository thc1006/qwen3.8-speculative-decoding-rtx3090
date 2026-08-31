#!/bin/bash
# Phase E5. One invocation, because the rotation is inside it: three passes over three
# arms, so `rot = (p_idx - 1) % len(arms)` closes and each cap visits each order position
# exactly once. E3 and E4 could leave arm order fixed because their estimand was within-arm
# across invocations; here the contrast IS between arms, since the cap is what the arm names.
#
# `--passes 3` AND NOT `--latin-arms`, which would set the same 3 and is the flag that says
# why. bench.py writes its `design` block, `"passes": passes` included, BEFORE the
# `--latin-arms` override reassigns that variable, so a run under that flag records the
# pre-override count. The first attempt at this phase recorded `design.passes = 5` while
# three passes ran, and audit_results.py -- correctly -- called the file 225 of 375 records
# and failed it. Checked against the audit's own code rather than inferred from line
# numbers, and then confirmed in the partial file the killed run left behind.
#
# The flag is worth fixing, but not by editing the measurement path in the hour before a
# two-hour run: nothing here would have been through the gate. `--passes 3` states the same
# intent, is recorded correctly, and touches no code at all.
#
# About two hours: the 150 W cap generates at 8.9 tok/s, so a 400-token answer takes 45 s
# there against 10 s at stock, and the 4 s roll adds 8 s to every record at every cap.
cd /home/thc1006/dev/qwen3.8-speculative-decoding-rtx3090
set -u
out="results/phase_e5.json"
echo "=== phase E5  -> $out  $(date -Is) ==="
python3 harness/bench.py --matrix phase_e5 --passes 3 \
    --power-roll 4.0 --power-trace --out "$out"
rc=$?
echo "=== E5 done $(date -Is), exit $rc ==="
exit "$rc"
