# Second-device plan: RTX A6000 (48 GB)

Prepared 2026-08-24, before the card exists on this host. Everything here is written so that it
either runs correctly on arrival or refuses with a specific reason; nothing is left to be
noticed at run time.

## What the card actually changes

| | RTX 3090 (present) | RTX A6000 |
|---|---|---|
| architecture | GA102, **sm_86** | GA102, **sm_86** - identical |
| VRAM | 24 GB GDDR6X | **48 GB** GDDR6 (ECC capable) |
| memory bandwidth | **936 GB/s** | **768 GB/s** (-18 %) |
| power | 420 W default, 100-450 adjustable | 300 W |
| bus width | 384-bit | 384-bit |

Same architecture means the same CUDA kernels, so the divergence findings and the acceptance
behaviour should be **identical**, and that is used as a control rather than reported as a result.
What differs is the resource mix: more memory, less bandwidth, less power.

## What it unlocks, ranked

1. **The target-quantization ladder on the 27B itself.** With the arithmetic corrected to the
   8192 context this study actually uses, a 24 GB card reaches `UD-Q4_K_XL` (19.8 GB) and
   *marginally* `UD-Q5_K_XL` (23.1 GB, 96 % of the card). `UD-Q6_K_XL` (27.5 GB) and `Q8_0`
   (31.3 GB) need the larger card. This matters because two separate open claims turn on target
   quantization: llama.cpp #25618's finding that divergence is quantization-dependent, and the
   PR #27342 author's account of the n-max ceiling as a quantization x arithmetic-intensity
   effect (recorded as H2'). Measuring `c`, the marginal cost per verified position, at each
   rung tests H2' on the coefficient not on throughput.
2. **Long context without a KV-quantization confound.** At 48 GB, `UD-Q4_K_XL` plus a 262 K
   f16 KV cache fits, so the decode cliff reported in #27623 can be crossed without also
   changing KV precision to make room.
3. **A physical bandwidth step.** Phase R varies memory clock by +/-4 % with software offsets;
   the A6000 is a -18 % step at nearly the same core count. That is a four-times-larger lever on
   the same axis, but it moves power at the same time, so it is a **cross-check on Phase R, not
   a replacement for it**.

## What it does not do

- **No bf16 target for the 27B.** BF16 is 50 GB; it does not fit on 48 GB either. The bf16 anchor
  that #25618 rests on comes from `phase_qsmall` on the existing card instead, the 9B model whose
  BF16 is 18.4 GB, and whose Q4_K_M is the exact file used in llama.cpp #26750.
- **No architectural diversity.** Both cards are sm_86. The open question in #26750 concerns
  Blackwell; an Ada or Blackwell card would address it and a second Ampere will not.
- **No wider power sweep.** The 3090 already adjusts 100-450 W; the A6000's ceiling is lower.
- **No multi-GPU coverage.** Several open issues (#27366, #27577, #26339, and the `-sm tensor`
  reports in PR #27342) are multi-GPU; one more single card does not reach them.

## Code prepared for it

| file | purpose |
|---|---|
| `harness/devices.py` | device enumeration, per-device stock state, measured idle floor, VRAM capacity guard, ECC state, neighbour-GPU recording |
| `harness/matrices/phase_q.py` | 27B quantization ladder, rung chosen by `QWEN_Q_TARGET`, declares `REQUIRES_VRAM_GB` |
| `harness/matrices/phase_qsmall.py` | 9B ladder **including BF16**, runs on the existing 24 GB card, rung chosen by `QWEN_QS_TARGET` |
| `run_phase_q.sh` | disk-staged driver: auto-selects the rungs the card can hold, verifies completeness before deleting weights, never touches `models/target` |
| `harness/analyze_cross_device.py` | comparison restricted to dimensionless quantities, with the two identity controls |

`phase_a.py` needs no A6000 variant: it names no device, so mirroring Phase A is
`--gpu <n> --settle-floor`, and the device identity is recorded in the result file.

## Correctness work this preparation required

Adding a second device broke five things that were invisible with one card. All are fixed and
each fix was verified against the live hardware.

1. **`nvidia-settings` and `nvidia-smi` maintain independent GPU enumerations.** Clock offsets
   were being applied by nvidia-smi index. On one GPU the two trivially agree; on two they need
   not, and the failure mode is applying an offset to one card while measuring another, silently.
   Both tools report the same GPU UUID, so the mapping is now resolved by matching UUIDs
   (`gpustate.settings_index_for`) and **refuses instead of guessing**. There were two call
   sites; the second (`telemetry.overclock_state`) was missed on the first pass.
2. **"Stock" was a hard-coded 420 W**, this 3090's default and not a universal one. An A6000
   defaults to 300 W, so "restore stock" would have restored something that was never stock for
   it. Stock is now read from each device's own `power.default_limit`.
3. **The 60 C thermal gate is calibrated for this card in this chassis.** On another cooler it is
   either unreachable (a timeout every arm) or trivially met (a gate that does nothing).
   `--settle-floor` derives the target from the device's own measured idle floor, and bench.py
   now **refuses to use the fixed target on a card it was not calibrated for**.
4. **The idle floor was sampled, not waited for.** Six readings two seconds apart, taken right
   after a previous run, return a "floor" measured while the card is still shedding heat, and the
   gate silently becomes a no-op exactly when it is needed. It now waits for the reading to stop
   falling and reports if it never stabilises.
5. **The run lock did not actually lock.** `acquire_lock` overwrote the file, so two concurrent
   runs would each believe they held it. It now refuses while a live run holds it, and takes over
   a stale lock with a message. The lock stays **global, not per-device**, deliberately:
   concurrent runs in one chassis contaminate each other through the power supply, case airflow
   and PCIe regardless of which card each uses.

Two smaller ones: the device tag now includes the index (two identical cards would otherwise
overwrite each other's outputs), and the cross-device report labels columns with the filename as
well as the card name.

## Order of operations on arrival

```bash
# 0. confirm the card and the mapping before anything else
python3 -c "import sys;sys.path.insert(0,'harness');import devices as D
[print(d.describe()) for d in D.enumerate_devices()]"
python3 -c "import sys;sys.path.insert(0,'harness');import gpustate as G
print('settings index for smi 1 ->', G.settings_index_for(1))"   # must not raise

# 1. mirror Phase A on the new card: gives k0/c at a different bandwidth, and both controls
python3 -u harness/bench.py --matrix phase_a --passes 5 --gpu 1 --settle-floor \
    --port 18170 --out results/phase_a_a6000.json
python3 harness/analyze_cross_device.py results/phase_a.json results/phase_a_a6000.json

# 2. the quantization ladder, the thing 24 GB cannot do
GPU=1 PASSES=3 bash run_phase_q.sh

# 3. the bf16 anchor, on EITHER card (9B fits both)
for R in Q4_K_M Q6_K Q8_0 BF16; do
  QWEN_QS_TARGET=$R python3 -u harness/bench.py --matrix phase_qsmall --passes 3 \
      --gpu 0 --port 18175 --out results/phase_qsmall_$R.json
done
```

Step 1 is the gate: if CONTROL 1 (acceptance) or CONTROL 2 (fork positions) does not come back
identical between two sm_86 cards running the same GGUF at greedy, something other than the
device is varying and steps 2 and 3 should not be interpreted until that is understood.

## Disk

The 27B ladder is ~93 GB of weights and `run_phase_q.sh` stages one rung at a time, reusing the
`UD-Q4_K_XL` this repo already holds. The 9B ladder is 41.8 GB and fits at once, but it competes
with the 21 GB MoE target needed for the dense-vs-MoE phase, so those two cannot both be resident
on the current volume. Sequence them.

---

## Correction, 2026-08-25: the card is on a separate host, and the plan above assumed otherwise

The A6000 exists and has been surveyed. It is **not** a second card in this chassis. It is
`mailer.cirda.nycu.edu.tw`, reached over Tailscale, and that invalidates part of the plan above
rather than merely adding detail.

Everything in "Correctness work this preparation required" was written for two cards in one box.
On separate hosts, items 1 (resolving the nvidia-smi against nvidia-settings index mapping by GPU
UUID) and 5's justification (concurrent runs in one chassis contaminating each other through the
power supply, case airflow and PCIe) do not apply: each host has one card at index 0 and its own
power and airflow. The code is not wrong, it is simply not exercised by this arrangement. Items 2
(stock power read per device), 3 (thermal gate calibrated per device) and 4 (idle floor waited for,
not sampled) matter more than before, because the hosts differ in ways two cards in one box would
not.

What replaces the removed reasoning is the cross-host rule the fleet already imposes: **absolute
throughput does not pool across hosts.** The step-1 gate below still holds and is still the gate,
but it is a comparison of dimensionless quantities and fork positions, not of tok/s.

### Surveyed state, 2026-08-25

| | value | consequence |
|---|---|---|
| GPU | RTX A6000, 49140 MiB, sm_86 | as planned |
| driver | 580.95.05, DKMS-built for the running 6.1.0-39 kernel | **works; nothing to repair** |
| ECC | **Disabled** | matches both 3090s, so ECC is not a confound after all |
| power.default_limit | 300 W | as the table above assumed |
| persistence mode | Enabled | differs from host A; record it |
| CUDA | 12.9 at `/usr/local/cuda`, **`nvcc` not on `PATH`** | one `export` away |
| cmake | **absent** | the only package that must be installed; 3.25.1 is available and ggml-cuda needs 3.18 |
| gcc | 12.2 (Debian 12), glibc 2.36 | officially supported with CUDA 12.9; no host-compiler override needed |
| cores | 16 | builds take roughly twice as long as host B's 32 |
| disk | 549 GB free | the 93 GB ladder fits without staging, though staging is kept anyway |
| rsync | absent | transfers use `scp`, as with host B |

This makes **three** distinct toolchains across the fleet: CUDA 13.3 / gcc 14.2 / glibc 2.41 on
host A, CUDA 12.0 / gcc 13.3 / glibc 2.39 on host B, CUDA 12.9 / gcc 12.2 / glibc 2.36 here. Each
host therefore needs its own build and its own baseline, and no result crosses hosts except as a
comparison of dimensionless quantities.

### The blocker is occupancy, not configuration

The card is in use. A `ghcr.io/ggml-org/llama.cpp:server-cuda` container (`v32-gpu`) has been up
four days serving `gemma-4-26B-A4B-it-UD-Q4_K_M` with a LoRA on port 8083, holding 19 GB and
leaving 29491 MiB free, alongside roughly thirty other containers on the same box. The owner has
said this is other research in progress and that the card will be free later.

Nothing in this plan can run against a shared card. It is not only that a neighbour at 100 %
utilisation would make the timings meaningless; the harness pins clocks with `nvidia-smi -lgc`,
waits on a measured idle floor and gates on temperature, and every one of those actions would
reach into the neighbouring workload. **Phase Q on this card requires the card to itself.**

Note also that 29 GB free is not enough for the rung this card was wanted for: `Q8_0` needs
31.3 GB and `UD-Q6_K_XL` needs 27.5 GB. Even setting contamination aside, the top of the ladder
does not fit beside the running service.
