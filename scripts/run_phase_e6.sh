#!/bin/bash
# Phase E6, interleaved. Three generation lengths x three rounds, the order of the lengths
# rotated each round so no length sits in one part of the session. One arm, so there is no
# arm rotation to close and --passes 1 is right; the manipulation is across invocations,
# exactly as in E3 and E4.
#
# About 80 minutes: 25 prompts at 12.9, 17.8 and 27.5 s per record, plus a model load each.
cd /home/thc1006/dev/qwen3.8-speculative-decoding-rtx3090
set -u
declare -a R1=(200 400 800)
declare -a R2=(400 800 200)
declare -a R3=(800 200 400)
fail=0
for round in 1 2 3; do
  eval "toks=(\"\${R${round}[@]}\")"
  for t in "${toks[@]}"; do
    out="results/phase_e6_tok${t}_r${round}.json"
    echo "=== round $round  max_tokens $t  -> $out  $(date -Is) ==="
    if ! python3 harness/bench.py --matrix phase_e6 --passes 1 \
           --max-tokens "$t" --power-roll 4.0 --power-trace --out "$out"; then
      echo "!!! FAILED: round $round tokens $t" >&2
      fail=$((fail+1))
    fi
  done
done
echo "=== E6 done $(date -Is), $fail failed invocation(s) ==="
exit "$fail"
