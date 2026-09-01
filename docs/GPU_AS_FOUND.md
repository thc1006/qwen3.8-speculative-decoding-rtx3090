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

After the reset the residual is smaller, and it is the one that applies to every published
number here: memory clock **9751 MHz, identical** to the predecessor's card; max graphics 2130
against 2100, **+1.4 %**; power limit 420 against 350 W, **+20 %**. So the axis the next paragraph
says matters most is closed, and the table above is the as-found state rather than the measured
one.

So this host differs from the earlier work on a **third** axis beyond board and OS.
Memory-bandwidth overclock in particular is not a neutral variable for this study:
batch-1 decode responds strongly to the memory clock and speculative verification barely
does, measured later as elasticities of 0.79-0.81 against 0.13-0.18, so an undisclosed memory
overclock shifts the two arms by different amounts.

## Decision

Overclock is promoted from an uncontrolled constant to a **controlled factor**.
Phase A is re-run at stock (offsets 0, power limit at the card's 420 W default) so it is
comparable to the predecessor repo and to the stock rows in community tables.

**The gate has since grown a third axis, and Phase A predates it.** Since 2026-08-29,
Correction 40, `harness/telemetry.is_stock` also requires `fan_control == "auto"` and treats an
unreadable value as not-stock, because a fan curve raises the sustained clock the way an offset
does and nothing recorded it. Phase A's `overclock_state` carries no fan field, so its recorded
`is_stock: true` is the two-axis verdict and cannot be re-derived as the three-axis one. Phase E
onward records all three. The overclock
is then varied deliberately in its own phase as a mechanistic probe; see PREREGISTRATION.md.
