# GPU as-found state, 2026-08-24, BEFORE any reset by this study

Discovered while designing the overclock phase, ~10 minutes into the first full Phase A run.
The card was **not** at stock settings, and an earlier draft of README.md and
PREREGISTRATION.md wrongly described it as a "450 W stock cap". That Phase A run was
aborted rather than kept, because every number in it would have carried an undisclosed
overclock (archived as `logs/phase_a_ABORTED_oc_undisclosed.log`).

```
GPUMemoryTransferRateOffset[4] = 800   # memory clock +400 MHz over stock
GPUGraphicsClockOffset[4]      = 100   # core +100 MHz
power.limit [W], power.default_limit [W], power.min_limit [W], power.max_limit [W], clocks.max.graphics [MHz], clocks.max.memory [MHz]
450.00 W, 420.00 W, 100.00 W, 450.00 W, 2220 MHz, 10151 MHz
```

## Why this matters

The predecessor repo's `BENCHMARK_ENV.md` documents its v2/v3 cards explicitly as
"Stock clocks, no overclocking": memory 9751 MHz, max graphics 2100 MHz, 350 W limit.

| | predecessor v2/v3 card | this card, as found | delta |
|---|---:|---:|---:|
| memory clock (max) | 9751 MHz | 10151 MHz | +4.1 % |
| graphics clock (max) | 2100 MHz | 2220 MHz | +5.7 % |
| power limit | 350 W | 450 W (default 420 W) | +28.6 % |

So this host differs from the earlier work on a **third** axis beyond board and OS.
Memory-bandwidth overclock in particular is not a neutral variable for this study:
batch-1 decode responds strongly to the memory clock and speculative verification barely
does, measured later as elasticities of 0.79-0.81 against 0.13-0.18, so an undisclosed memory
overclock shifts the two arms by different amounts.

## Decision

Overclock is promoted from an uncontrolled constant to a **controlled factor**.
Phase A is re-run at stock (offsets 0, power limit at the card's 420 W default) so it is
comparable to the predecessor repo and to the stock rows in community tables. The overclock
is then varied deliberately in its own phase as a mechanistic probe; see PREREGISTRATION.md.
