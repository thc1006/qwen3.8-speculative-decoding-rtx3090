#!/bin/bash
# Phase E3, interleaved. Three intervals x three rounds, the interval order
# rotated each round so that no interval sits in one part of the session. Same
# number of model loads as three block runs, and the time position is balanced
# out of the comparison instead of left in it.
cd /home/thc1006/dev/qwen3.8-speculative-decoding-rtx3090
set -u
declare -a R1=(0.05 0.10 0.20)
declare -a R2=(0.10 0.20 0.05)
declare -a R3=(0.20 0.05 0.10)
fail=0
for round in 1 2 3; do
  eval "ivs=(\"\${R${round}[@]}\")"
  for iv in "${ivs[@]}"; do
    tag=$(echo "$iv" | tr -d '.')
    out="results/phase_e3_iv${tag}_r${round}.json"
    echo "=== round $round  interval $iv  -> $out  $(date -Is) ==="
    if ! python3 harness/bench.py --matrix phase_e3 --passes 1 \
           --power-interval "$iv" --out "$out"; then
      echo "!!! FAILED: round $round interval $iv" >&2
      fail=$((fail+1))
    fi
  done
done
echo "=== E3 done $(date -Is), $fail failed invocation(s) ==="
exit "$fail"
