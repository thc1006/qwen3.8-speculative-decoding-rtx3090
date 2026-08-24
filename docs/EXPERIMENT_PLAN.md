# Experiment plan

Revised 2026-08-24, after the pilot measurements and the overclock discovery. Supersedes the
phase sketch in `TODO.md`; hypotheses remain as committed in `PREREGISTRATION.md`.

## What the pilot established (verified, not assumed)

| finding | evidence |
|---|---|
| the card is **power**-capped, not thermally capped | `SW Power Cap: Active`, `HW Thermal Slowdown: Not Active`, 15.5 h of accumulated power capping |
| it loses **9.3 % of SM clock** over one pass (1950 -> 1769 MHz) | per-request clock sampling across a 7-arm pass |
| **baseline decode is memory-bandwidth-bound** | removing a +4.1 % memory overclock cost exactly 4.1 % throughput (43.40 -> 41.6 tok/s, matched prompts) |
| the two llama.cpp trees introduce no confound | baseline 43.69 vs 43.72 tok/s, and **byte-identical output on 5/5 prompts** |
| MTP and DFlash2 both run on sm86 | MTP n3 88.90 tok/s @ 84 % accept, mean draft len 3.51; DFlash2 n7 91.92 tok/s @ 67 %, mean len 5.64 |
| speculation **changes the emitted text** | 60-80 % of prompts fork from baseline at greedy; fork position is identical across different drafters and shifts with n-max, i.e. it tracks verification batch shape |

## The instrument the pilot handed us

Memory clock and power limit are independently settable, reversible, and measurable on this
card. That converts the mechanism question from an indirect inference into a direct
manipulation, and the baseline's measured bandwidth elasticity of ~1.0 is a built-in calibration
point.

Define, for method `m`, the elasticities

    e_BW(m) = d ln(decode tok/s) / d ln(memory clock)
    e_P(m)  = d ln(decode tok/s) / d ln(power limit)

| | e_BW | e_P |
|---|---|---|
| baseline (measured) | **~ 1.0** | ~ 0 until power-starved |
| if **H2'** (quantization x arithmetic intensity) holds | speculative arms **below** baseline, falling further as n-max rises | speculative arms **above** baseline |
| if **H2** (Gated DeltaNet state rollback) holds | speculative arms **at or above** baseline - reconstructing recurrent state is memory traffic | speculative arms ~ baseline |

The two hypotheses predict **opposite signs** for the same contrast. That is what makes this
decisive where the p-min sweep alone is only suggestive. The PR #27342 author advances H2' from
`llama-batched-bench` per-step costs; nobody has manipulated the resources directly.

## Phases, in priority order

| phase | question | arms | passes | est. |
|---|---|---|---|---|
| **A** (running) | headline effects with intervals, at verified stock | 7 | 5 | ~2.3 h |
| **R** | **resource response - the decisive mechanism test** | 15 | 3 | ~2.7 h |
| **B** | drafted-token vs rejected-token cost (p-min x n-max) | 7 | 5 | ~2.3 h |
| **C** | method breadth on one host: drafter-quant ladder, classic draft, ngram family, DSpark | ~10 | 3 | ~2 h |
| **L** | does speculation survive the ~25x decode collapse past ~80 K ([#27623](https://github.com/ggml-org/llama.cpp/issues/27623))? | 4 x depths | 3 | ~2 h |
| **M** | dense-hybrid vs A3B MoE under one protocol | 4 | 5 | ~2 h + 21 GB |
| **V** | llama.cpp vs vLLM on the same single card | 4 | 3 | ~2 h + disk juggling |
| **X** | thinking on/off x method, multimodal, concurrency | - | - | later |

### Phase R design

Five resource conditions, three methods, fully crossed. Every condition is applied by the
harness, verified by reading the state back, and restored afterwards.

| condition | mem offset | core offset | power limit | what moves |
|---|---:|---:|---:|---|
| `stock` | 0 | 0 | 420 W | reference |
| `bw-lo` | -400 | 0 | 420 W | bandwidth down |
| `bw-hi` | +400 | 0 | 420 W | bandwidth up |
| `pw-lo` | 0 | 0 | 250 W | compute down |
| `pw-vlo` | 0 | 0 | 175 W | compute downdown |

Methods: `baseline`, `mtp-n3`, `mtp-n7`. Deep speculation is included because both hypotheses
predict the divergence grows with n-max.

Three bandwidth points allow a linearity check not a two-point slope, and two power
reductions distinguish "compute-insensitive" from "not yet power-starved".

Safety: memory overclock is the only setting that can destabilise the card. Each condition is
validated with a short generation and a degeneracy screen before the arm is measured; an unstable
condition is recorded as an incident and dropped, never silently retried.

### What Phase R cannot settle

Elasticities are measured at one target quantization (Q4_K_XL) on one card. H2' is explicitly a
claim about how quantization changes the compute/bandwidth ratio, so a full test of it wants a
second target quant. That is a candidate Phase R2 if disk allows (`UD-Q6_K` is 25.3 GB).
