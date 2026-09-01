# Salvaged partial data from the aborted overclocked Phase A run

Not a result. Recorded because it is a well-matched observation that would otherwise be
thrown away: the same prompts, same binary, same harness, differing only in the card's
clock state. Baseline arm only -- the run was stopped before it reached any speculative arm.

Card state during these observations: memory +400 MHz, core +100 MHz, power limit 450 W.
See docs/GPU_AS_FOUND.md.

The observations are in [`oc_baseline_partial.txt`](oc_baseline_partial.txt): 21 of the 25
prompts, three columns -- class, prompt tag, decode tok/s -- ranging 43.29 to 43.60. The run
stopped after the baseline arm's 21st prompt.

This file used to reproduce that table inline and the copy was corrupt: sixteen of the twenty-one
rows had lost their throughput value and gained a stray `]`, and every row had lost its class
column. Only the five `reason_*` rows survived. A table that a reader cannot check against
anything is worse than a pointer to the file that holds it.

For the comparison this exists to support, the stock-clock baseline over the same prompts is
`baseline@master` in `results/phase_a.json`.
