# Qwen3.8-27B speculative decoding on a single RTX 3090: a controlled study

> **Status, 2026-08-24.** Phase A is complete: 875 measurements, 0 incidents, nothing excluded.
> Phase R (resource response) is complete at 1125 measurements and Phase R2 is re-running its
> compute axis with the clock pinned instead of power-capped, which fixes three defects its own
> review found. Later phases are designed and not yet run; each says so where it appears.
>
> The hypotheses, the analysis plan and the prior-art scoping were committed **before** any
> measurement, in [`PREREGISTRATION.md`](PREREGISTRATION.md), which is append-only. Two of its
> own hypotheses have since been recorded as unsupported.

Successor to [`thc1006/qwen3.6-speculative-decoding-rtx3090`](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090).
That repo asked whether speculative decoding pays off for a 3B-active MoE on consumer Ampere,
and on llama.cpp the answer was no: every configuration tested came out net negative, which the
write-up explained by expert saturation, since a draft of K tokens well below the ~94-token
saturation threshold forces the verify pass to load the union of K positions' expert slices.
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) is dense-hybrid. No experts, no
routing, no union. So that mechanism cannot be what decides the answer here, and the question is
open again.

## What this repo is not claiming

The prior-art sweep for this study (2026-08-24, recorded in `PREREGISTRATION.md`) found that
several things a first draft of this README would have called "first" are already published:

- **Single RTX 3090 + DFlash2 + a Q4 target is already reported**, twice, in the comment thread
  of llama.cpp PR #27342, by `treo` (`UD-Q4_K_XL`, 32K ctx) and `ouening` (`UD-Q4_K_M`, 128K
  ctx, Windows). Priority is theirs and is cited, not re-claimed here.
- **llama.cpp `draft-mtp` on a 3090** is covered in depth by
  [`sudoingX/qwen38-mtp`](https://github.com/sudoingX/qwen38-mtp), across six 3090 rows plus
  quant, KV-type, power and DSpark sweeps.
- **Drafter-quantization comparison** is partly answered in that same PR thread, on a 32 GB card.
- **vLLM on a single 3090** for this model is covered by
  [`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090).
- **Losslessness of speculative decoding on consumer hardware** was studied in
  [arXiv 2607.17283](https://arxiv.org/html/2607.17283), though on Apple silicon, with Qwen2.5,
  and with classic two-model speculation rather than MTP or DFlash2.

So the throughput table is not the contribution. What is left, and what this repo went after, is
the protocol and the axes nobody ran: a paired interleaved design that puts an interval on every
number, a thermal gate at arm entry, per-request energy, byte-level output divergence, and a
mechanism test that separates two competing explanations for why deeper drafting stops paying.
Details below.

## What this answers

Written out as questions because that is how people arrive here, and the numbers behind each
answer are in the sections below.

**Is speculative decoding worth enabling for Qwen3.8-27B on an RTX 3090?**
Yes, and by a lot. The built-in MTP head at `--spec-draft-n-max 2` gives +59.8 % decode
throughput over no speculation, measured across 875 requests with an interval.

**What `--spec-draft-n-max` should I use?**
2 for `draft-mtp`, 4 for `draft-dflash`. Both are the interior maximum of
`speedup = mean_len / k` where `k` grows linearly with verification width, so this is derived
and not just the best cell in a table.

**Does DFlash2 (llama.cpp PR #27342) beat the built-in MTP head on consumer Ampere?**
No. At its own best depth DFlash2 gives +51.9 % against MTP's +59.8 %. It does draft longer
blocks, and its fixed per-step cost is actually lower, but its acceptance falls faster with
depth.

**Does speculative decoding save energy, or just time?**
Both. MTP at n-max 2 takes 2503 J to produce a 400-token answer against 3980 J without it, so
37 % less energy. Decode-only, with prefill measured separately and subtracted.

**Is llama.cpp speculative decoding lossless at temperature 0?**
Not in bytes. 76-80 % of greedy requests produce different text from the non-speculative
baseline, forking at a median 23 % of the way in. It is deterministic and reproduces exactly
across passes, and it corroborates llama.cpp issue #25618.

**Why does deeper drafting stop paying off?**
Because speculation turns a bandwidth-bound decode into a compute-bound verify, and each extra
verified position costs about 0.28 of a plain decode step. Measured directly by varying memory
clock and core clock independently: speculative arms are 0.14-0.21x as sensitive to memory
bandwidth as the baseline and 1.71-1.74x as sensitive to core clock.

**Does the MoE result from Qwen3.6-35B-A3B carry over?**
No. That model's expert-saturation argument does not apply to a dense-hybrid target, and the
sign flips: net loss there, large net win here.

**Which prompts benefit?**
Code and reasoning most, Chinese least. `dflash2-n7` is +22.6 % overall while being a net loss
on prose, chat and Chinese, so a single average is misleading for this model.

## Design

| | |
|---|---|
| target | `unsloth/Qwen3.8-27B-GGUF` · `Qwen3.8-27B-UD-Q4_K_XL.gguf` (17.56 GB) |
| architecture | `qwen35`, 64 layers, `full_attention_interval: 4` → **48 Gated DeltaNet + 16 full attention**, vocab 248320, native VL |
| MTP | embedded in the quant: `qwen35.nextn_predict_layers = 1`, `blk.64.nextn.*` present (verified by reading the GGUF, no separate module needed) |
| GPU | 1 × RTX 3090 24 GB, driver 610.43.02, **420 W default limit, reset to stock for the primary matrix**. The card was found overclocked (mem +400 MHz, core +100 MHz, 450 W) and the first Phase A run was discarded because of it; see [`docs/GPU_AS_FOUND.md`](docs/GPU_AS_FOUND.md) |
| host | Debian 13, kernel 6.12, i9-13900, 31 GB RAM |
| engine | llama.cpp built from source, CUDA 13.3, `CMAKE_CUDA_ARCHITECTURES=86` (sm_86), two trees with identical flags |
| trees | `master` @ `c060ca9` (build 200) · **PR #27342** (DFlash2, unmerged) @ `d1a522f` |
| prompts | 25, balanced 3×5 → **5 per class** over code / prose / reason / chat / zh; every prompt written to exceed the 400-token cap |
| sampling | greedy, full sampler chain pinned explicitly, `cache_prompt: false`, `--parallel 1`; `--spec-draft-p-min` left at its default, so no arm gates its own drafts on confidence and acceptance is a property of the drafter rather than of a threshold |

### The controls, and why each exists

Every one of these was added because something measurable went wrong without it.

| control | the failure it prevents |
|---|---|
| **arms interleaved within each pass, order rotated** | running arms sequentially confounds session drift with arm identity |
| **thermal gate at arm entry** | this card sits on its 450 W cap and loses **9.3 % of SM clock** (1950 → 1769 MHz) over one pass; that is larger than several effects under study, and it lands inside every paired comparison |
| **dual-tree baseline** | DFlash2 needs an unmerged branch; comparing it to a master-tree baseline would conflate the method with the branch |
| **`cache_prompt: false`, verified via `t_cache_n`** | prompts share a system message within a class; prefix caching also *interacts* with speculation ([vLLM #38182](https://github.com/vllm-project/vllm/issues/38182)) — a confound of this exact shape forced a retraction in the sibling repo |
| **drafter assertion, log evidence + `t_draft_n > 0`** | the predecessor repo shipped a table row whose draft model never attached; a flag can be accepted and ignored |
| **port-ownership guard** | a killed-but-unreaped server keeps answering `/health`; a contributor to another study published three rows measured against a zombie |
| **class-stratified primary endpoint** | the effect has **opposite signs** across classes; a raw mean reports the prompt mixture as if it were a result |
| **cluster bootstrap over prompts** | passes of one prompt are repeated measures, not independent samples |
| **degeneracy screen relative to baseline** | collapsed output is fast; [vLLM #52475](https://github.com/vllm-project/vllm/issues/52475) reports MTP repetition collapse on this model family |
| **stock-clock enforcement** | this card arrived overclocked (mem +400 MHz, core +100 MHz, 450 W vs a 420 W default) while the README said "stock"; the harness now reads the offsets and **refuses to run** unless they are zero or an overclock is declared |

A full audit of the predecessor repo's methodology is in
[`docs/METHODOLOGY_AUDIT.md`](docs/METHODOLOGY_AUDIT.md). It includes a re-analysis of that
repo's own committed data, which shows its headline effect was understated about two-fold by
prompt-class mixture.

## Results: Phase A

7 arms × 25 prompts × 5 passes = **875 measurements, 0 incidents, 0 excluded, 0 quality-flagged.**
Intervals are a paired cluster bootstrap over prompts (passes of one prompt are repeated
measures, not independent samples), on the class-stratified effect.

| arm | verify width | tok/s | vs own-tree baseline | tok/J | decode J per request |
|---|---:|---:|---|---:|---:|
| baseline @ master | — | 41.55 | — | 0.1005 | 3980 |
| baseline @ PR #27342 | — | **41.55** | — | 0.1005 | 3979 |
| **mtp-n2** | 3 | 66.39 | **+59.8 % [+57.0, +62.8]** | 0.1627 | **2503 (−37 %)** |
| mtp-n3 | 4 | 63.29 | +52.3 % [+48.5, +56.5] | 0.1549 | 2684 |
| dflash2-n4 | 5 | 63.13 | +51.9 % [+45.6, +58.2] | 0.1554 | 2835 |
| mtp-n5 | 6 | 54.89 | +32.1 % [+26.4, +37.8] | 0.1343 | 3228 |
| dflash2-n7 | 8 | 50.95 | +22.6 % [+14.7, +30.4] | 0.1251 | 3786 |

The two trees agree to 41.55 tok/s and produce **byte-identical output on 125/125 prompt-passes**,
so nothing in the DFlash2 numbers is attributable to the unmerged branch. Run-to-run CV within a
prompt is ≤ 0.3 %.

**MTP at n-max 2 cuts the decode energy for a 400-token answer by 37 %** (3980 → 2503 J). Both
energy columns are decode-only: the prefill is measured separately, in its own eight-repetition
calibration per prompt, and subtracted. Counting it, the same request goes 4050 → 2583 J, a 36.2 %
saving. The prefill is measured per arm rather than assumed constant, and it is not: 70.9 J for
the baseline against 83.2 J for `dflash2-n7`, because a speculative arm processes the prompt
through its drafter as well.

No study in the prior-art sweep publishes an energy figure for this model; `joule`, `tok/J` and
`watt` appear zero times in PR #27342's 60-comment thread.

### The headline number hides a sign change

| arm | code | reason | prose | chat | zh |
|---|---:|---:|---:|---:|---:|
| mtp-n2 | +92.0 % | +73.6 % | +48.6 % | +46.9 % | +37.7 % |
| dflash2-n4 | +116.8 % | +89.4 % | +23.0 % | +30.2 % | +0.3 % |
| mtp-n5 | +90.7 % | +54.9 % | +7.1 % | +7.1 % | +0.6 % |
| **dflash2-n7** | +91.8 % | +65.4 % | **−11.1 %** | **−4.3 %** | **−28.7 %** |

`dflash2-n7` is +22.6 % overall. It is also a net loss on three of five classes. That is the same
failure this repo documents in the predecessor's own headline, where a mixture statistic got
reported as an effect, except here it runs the other way: the average flatters an arm that hurts
most of the workload. Which is why the primary endpoint is class-stratified and why the per-class
table is not an appendix.

### A cost model, not a table

llama.cpp reports enough per request to recover the cost of one speculative verification step in
units of a plain decode step. With `mean_len = (predicted_n − 1) / (predicted_n − accepted − 1)`:

    speedup = mean_len / k       k(w) = k0 + c·(w − 1),   w = n_max + 1

| method | widths | k0 | **c** | r² |
|---|---|---:|---:|---:|
| `draft-mtp` | 3, 4, 6 | 0.8937 | **0.2829** | 0.9998 |
| `draft-dflash` | 5, 8 | 0.7825 | **0.2784** | (2 points, meaningless) |

The `− 1` in that formula was missing at first and the numbers it produced looked fine. The first
generated token comes out of the prompt-processing pass, not out of a decode forward, and leaving
it in the count inflated the forward count by one. Checking the derivation against the server's
own `mean len` log line on all 625 speculative requests is what found it. The correction moves
`c` by 0.8 % and changes nothing that is claimed here, but it is the reason
[`upstream/`](upstream/) carries a one-line patch to expose the verification-step count the
server already holds: a derivation that reproduces plausible numbers and is quietly wrong by a
percent is exactly what an exposed counter prevents. Around 30 % of requests need one further
step removed, which is what truncation at the token cap looks like, and the API cannot say which,
so the figures above are low by under 1 % and are reported as such.

`c` agrees to **1.6 %** between the target's own built-in nextn head and a structurally unrelated
1.1 GB block-diffusion drafter, while `k0` differs by 14 %. The marginal cost of verifying one
more position belongs to the verification path; the fixed cost belongs to the drafter, and
DFlash2's fixed cost is the *lower* of the two. Across five prompt classes whose acceptance rates
differ by nearly tenfold, the class means of `k` span **0.26 % to 0.94 %** of their own mean,
depending on the arm. An earlier version of this section quoted 0.35 to 0.54 % here, which was
the dispersion of `k` over individual requests rather than over class means. The record-level
figure is the smaller and better-looking of the two, and it is not the one this claim is about.

This predicts the optimum instead of tabulating it: `mean_len` saturates with depth while `k`
grows linearly, so the best n-max is interior. Measured: **2 for MTP, 4 for DFlash2.**

### What the cost model rules out

A state-rollback account charges the overhead to *rejection*: 48 of this model's 64 layers are
Gated DeltaNet and cannot roll back by truncating a KV suffix. Writing that as
`k = k_verify + r·n_max·(1 − acceptance)` makes `r` estimable from the slope of `k` against
acceptance. Across an acceptance range of **0.096–0.918**, every arm returns
**|r| ≤ 0.0028** decode-steps per rejected token, r² between 0.001 and 0.060.

The overhead is paid per position **verified**, not per draft **rejected**. That does not make
rollback free; it bounds how much of the measured cost rollback can account for, and the bound is
approximately none. The hypothesis was this repo's own, pre-registered, and is reported as
unsupported.

### Losslessness

Speculative arms are byte-identical to their baseline on only 25–30 of 125 prompt-passes: **76–80 %
of requests diverge**, forking at a median 23 % into the text. Every arm is nonetheless
**100/100 reproducible across passes**, so the divergence is deterministic rather than noise. Fork
positions partition the arms into exactly two stable groups by verification width
(`{3,4}` against `{5,6,8}`), identically in all five passes and shared across unrelated drafters.

This corroborates [llama.cpp #25618](https://github.com/ggml-org/llama.cpp/issues/25618) rather
than discovering anything: that thread already establishes the phenomenon, its
quantization-dependence, its drafter-independence, and a root cause on the Vulkan side. What is
still open is the **CUDA** boundary, and a width-localised boundary is what this repo can add.
[llama.cpp #26750](https://github.com/ggml-org/llama.cpp/issues/26750) asks the same question on
Blackwell; this card is sm_86 and cannot answer it.
See [`docs/UPSTREAM_CONTRIBUTIONS.md`](docs/UPSTREAM_CONTRIBUTIONS.md).

### Two things went wrong and are recorded, not smoothed over

1. **The card arrived overclocked** (memory +400 MHz, core +100 MHz, 450 W against a 420 W
   default) while this README described it as stock. The first Phase A run was **discarded**, the
   card reset, and the harness now refuses to start on a non-stock card. `docs/GPU_AS_FOUND.md`.
2. **The completed run's process crashed** with a glibc `double free or corruption` after writing
   its last record. The cause was a harness bug (`preexec_fn` used alongside a sampling thread, now
   `start_new_session=True`). All 875 measurements survived; the final pass's derived comparisons
   were recomputed from the recorded text. `PREREGISTRATION.md`, Correction 2.

## Results: later phases

**Phase R** (resource response: memory bandwidth × power budget × method) is complete at 1125
measurements. Its pre-flight confirmed the assumption the design rests on: lowering the power
limit to 250 W and 175 W leaves the memory clock at 9501 MHz, unchanged, so the compute and
bandwidth levers are genuinely separable on this card. Its own review then found that a power cap
is a poor compute lever, because the clock it produces is an outcome rather than a setting, and
**Phase R2** is re-running the compute axis with the SM clock pinned at 600, 1200 and 1700 MHz
instead.

The rest are designed, pre-registered and not yet measured. Each hypothesis was written down
before its data existed, in the addenda to [`PREREGISTRATION.md`](PREREGISTRATION.md).

| phase | question | status |
|---|---|---|
| **C** | does drafter quantization change the answer, and does the predecessor's v3.0 need an erratum? | queued |
| **L** | does the long-context decode collapse of [#27623](https://github.com/ggml-org/llama.cpp/issues/27623) reproduce on sm_86, and does speculation survive it? | designed, ladder to 96K |
| **M** | does `draft-mtp` at small n-max escape the MoE penalty that `draft-simple` at n-max 8 suffers? | designed, anchored on reproducing the predecessor's −44.6 % |
| **Q** | does the target quantization ladder move the marginal cost per verified position? | driver written, needs the 48 GB card above `UD-Q5_K_XL` |
| **V** | does the same comparison hold on vLLM rather than llama.cpp? | designed, [`docs/PHASE_V_DESIGN.md`](docs/PHASE_V_DESIGN.md); matched at K=1, so it waits on the n-max ladder |

## Reproduce

```bash
# toolchain (Debian 13; NVIDIA CUDA repo already configured)
sudo apt-get install -y cuda-toolkit-13-3 ninja-build ccache

# two trees, identical flags: DFlash2 is an unmerged PR, so there is no prebuilt for it,
# and mixing a prebuilt master with a self-built PR binary would reintroduce a build confound
git clone https://github.com/ggml-org/llama.cpp llamacpp-master
cp -r llamacpp-master llamacpp-dflash2
cd llamacpp-dflash2 && git fetch origin pull/27342/head:pr-27342 && git checkout pr-27342 && cd ..
for t in llamacpp-master llamacpp-dflash2; do
  CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B $t/build -S $t -GNinja \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CCACHE=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build $t/build -j --target llama-server
done

# models
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf --local-dir models/target
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF --local-dir models/dflash2

# run + analyse
python3 harness/bench.py --matrix phase_a --passes 5 --out results/phase_a.json
python3 harness/analyze.py results/phase_a.json
```

`harness/bench.py --prompts-per-class 1` runs a reduced dry run; reduced runs label themselves
in the output file so they can never be mistaken for a full result.

## Limitations, stated up front

- Single card, single host. Absolute tok/s are **not** comparable to the predecessor repo's
  numbers: that work used two other physical 3090s with 350 W caps.
- `-c 8192` for the primary matrix. The long-context behaviour: including whether speculation
  survives the ~25× decode collapse past ~80 K reported in
  [llama.cpp #27623](https://github.com/ggml-org/llama.cpp/issues/27623), is a separate phase.
- `draft-eagle3` is supported by this build but **no EAGLE3 drafter has been published for
  Qwen3.8-27B**, so the method cannot be evaluated.
- Multimodal input is not in the primary matrix; llama.cpp has historically refused speculation
  together with `--mmproj` ([#19712](https://github.com/ggml-org/llama.cpp/issues/19712)).
- Single-stream only in Phase A. Speculative decoding is a single-stream optimisation and its
  advantage is reported by others to vanish by `--parallel 4`; measuring that is a later phase.

## License

MIT for code. Results (JSON / CSV) under CC-0.

## Author

Hsiu-Chi Tsai · GitHub [`thc1006`](https://github.com/thc1006)
