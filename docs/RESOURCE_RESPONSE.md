# Clock response (Phases R and R2)

Extracted from the README so the front page stays readable. Part of
[`thc1006/qwen3.8-speculative-decoding-rtx3090`](https://github.com/thc1006/qwen3.8-speculative-decoding-rtx3090).

## Phase A's arms boosted lower than their baselines

<details>
<summary>The speculative arms boosted lower than their own baselines, so the speedups are understated</summary>

Every speculative arm ran at a **lower SM clock than the baseline it is compared against** - 1.98 %
lower for `dflash2-n7`, 4.17 % lower for `mtp-n5` - and **2 to 3 degrees hotter**. That range used
to read "3 to 5"; measured three ways -- arm means, prompt-paired means and prompt-paired medians
-- the per-arm difference is 2.1 to 3.3 C on both `temp_mean_c` and `temp_max_c`, and 5 appears
only on single prompts, where the worst reaches 7.0. Nothing was pinned;
this is the card boosting less because a speculative arm draws more power for the same wall time.

The direction matters. A treatment arm running *faster* than its control would inflate the effect;
one running slower deflates it. Correcting with this study's own SM-clock elasticity for the
interval those clocks sit in, 0.78 for the speculative arms:

| arm | clock vs its baseline | measured | at matched clock |
|---|---:|---:|---:|
| `mtp-n2` | -3.87 % | +59.77 % | ~ +64.7 % |
| `mtp-n3` | -4.01 % | +52.32 % | ~ +57.2 % |
| `dflash2-n4` | -3.06 % | +51.94 % | ~ +55.7 % |
| `mtp-n5` | -4.17 % | +32.10 % | ~ +36.5 % |
| `dflash2-n7` | -1.98 % | +22.63 % | ~ +24.6 % |

The measured column stays the headline, for two reasons. It is the conservative one, and it is
what the card actually delivers to a user who has not pinned anything. The matched-clock column is
an estimate from an elasticity measured in a different phase, not a measurement, and it is here so
that the boost difference is not mistaken for something working in the study's favour.

</details>

## What each workload responds to

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../analysis/plot_bound_by_dark.png">
  <img alt="Two panels. Top: memory-clock elasticity against SM-clock elasticity at the top of
  the clock range. The baseline sits at 0.80 memory and 0.27 SM; both speculative arms sit near
  0.15 memory and 0.78 SM, in the opposite corner, and all three lie close to a dotted line
  where the two elasticities sum to one. Bottom: SM-clock elasticity by interval. From 600 to
  1200 MHz all three are between 0.80 and 0.93; from 1200 to 1710 MHz the baseline falls to
  0.27 while the speculative arms stay at 0.76 and 0.80. These are response measurements, not a
  roofline." src="../analysis/plot_bound_by.png">
</picture>

Phase R varied memory bandwidth and power budget independently, 1125 request records, and confirmed
the assumption the design rests on: lowering the power limit to 250 W and 175 W leaves the memory
clock at 9501 MHz unchanged, so the two levers are separable on this card. Its own review then
found that a power cap is a poor compute lever, because the clock it produces is an outcome rather
than a setting. **Phase R2** re-ran the compute axis with the SM clock pinned at 600, 1200 and
1710 MHz, 1575 request records, 0 incidents, and it is the one quoted here.

At the top of the clock range the two workloads sit in opposite corners:

| | bandwidth elasticity | compute elasticity |
|---|---:|---:|
| baseline | **0.79-0.81** | **0.27** |
| mtp-n3 | 0.13-0.15 | 0.76 |
| mtp-n7 | 0.17-0.18 | 0.81 |

The two elasticities very nearly swap: over these clock intervals the baseline responds mostly to
the memory clock and the speculative arms mostly to the SM clock. That is consistent with one
target pass scoring several positions at once and raising arithmetic pressure. It is a response
measurement, not a roofline: nothing here counts bytes moved or arithmetic issued, so it does not
establish which resource either workload is bound by.

The regime matters and the intervals show where it changes. From 600 to 1200 MHz everything is
clock-sensitive and everything scales with it: baseline 0.804, mtp-n3 0.913, mtp-n7 0.931.
From 1200 to 1710 MHz the baseline's SM-clock response collapses to 0.266 while the speculative
arms keep scaling at 0.759 and 0.805. The ratio between them therefore is not one
number: it is 1.14x in the low regime and 2.85x in the high one, which is why this repo reports
elasticities per interval and never pools them across a regime change.

Pinning tightened the intervals to the third decimal, and it binds at five of the seven
conditions. It does not bind at the top two. A pin holds only while the power limit does not, and
a speculative arm draws more at the same clock, so at `sm1700` the methods land on 1710, 1698 and
1708 MHz against one request, and at `sm1700-bwhi` on 1710, 1689 and 1703. The elasticities in the
table above cross `sm1700`, and `harness/elasticity.py` marks every interval that crosses an
unmatched condition rather than leaving it to be noticed in the clock columns. The arithmetic is
unaffected, since each elasticity divides by that arm's own log clock ratio, but the comparison
between arms there spans slightly different ranges. Phase R, for contrast, was mismatched at 30.0 %
and 35.8 %.

