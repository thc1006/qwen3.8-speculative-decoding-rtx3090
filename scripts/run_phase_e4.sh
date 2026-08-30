#!/bin/bash
# Phase E4, interleaved. Three pre/post-roll settings x three rounds, the roll
# order rotated each round so that no roll sits in one part of the session. Each
# roll appears in slots summing to the same mean position: 0.0 in 1/6/8, 1.5 in
# 2/4/9, 4.0 in 3/5/7, all averaging 5.0 of 9.
#
# The traces are recorded, so these files are about seven times the size of E3's.
# That is the point of the phase: a total cannot say WHERE in the window the two
# integrals separate.
cd /home/thc1006/dev/qwen3.8-speculative-decoding-rtx3090
set -u
declare -a R1=(0.0 1.5 4.0)
declare -a R2=(1.5 4.0 0.0)
declare -a R3=(4.0 0.0 1.5)
fail=0
for round in 1 2 3; do
  eval "rolls=(\"\${R${round}[@]}\")"
  for roll in "${rolls[@]}"; do
    tag=$(echo "$roll" | tr -d '.')
    out="results/phase_e4_roll${tag}_r${round}.json"
    echo "=== round $round  roll $roll s  -> $out  $(date -Is) ==="
    if ! python3 harness/bench.py --matrix phase_e4 --passes 1 \
           --power-roll "$roll" --power-trace --out "$out"; then
      echo "!!! FAILED: round $round roll $roll" >&2
      fail=$((fail+1))
    fi
  done
done
echo "=== E4 done $(date -Is), $fail failed invocation(s) ==="
exit "$fail"
