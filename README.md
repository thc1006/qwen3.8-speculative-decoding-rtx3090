# Qwen3.8-27B speculative decoding on a single RTX 3090: a controlled study

> **STATUS: measurement in progress (started 2026-08-24). No result in this README is final.**
> Sections marked `RESULTS PENDING` are placeholders. The hypotheses, the design, the analysis
> plan and the prior-art scoping were all committed **before** any measurement was taken — see
> [`PREREGISTRATION.md`](PREREGISTRATION.md).

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
  of llama.cpp PR #27342 — by `treo` (`UD-Q4_K_XL`, 32K ctx) and `ouening` (`UD-Q4_K_M`, 128K
  ctx, Windows). Priority is theirs and is cited, not re-claimed here.
- **llama.cpp `draft-mtp` on a 3090** is covered in depth by
  [`sudoingX/qwen38-mtp`](https://github.com/sudoingX/qwen38-mtp), across six 3090 rows plus
  quant, KV-type, power and DSpark sweeps.
- **Drafter-quantization comparison** is partly answered in that same PR thread, on a 32 GB card.
- **vLLM on a single 3090** for this model is covered by
  [`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090).
- **Losslessness of speculative decoding on consumer hardware** was studied in
  [arXiv 2607.17283](https://arxiv.org/html/2607.17283) — though on Apple silicon, with Qwen2.5,
  and with classic two-model speculation rather than MTP or DFlash2.

So the throughput table is not the contribution. What is left, and what this repo went after, is
the protocol and the axes nobody ran: a paired interleaved design that puts an interval on every
number, a thermal gate at arm entry, per-request energy, byte-level output divergence, and a
mechanism test that separates two competing explanations for why deeper drafting stops paying.
Details below.

## Design

| | |
|---|---|
| target | `unsloth/Qwen3.8-27B-GGUF` · `Qwen3.8-27B-UD-Q4_K_XL.gguf` (17.56 GB) |
| architecture | `qwen35`, 64 layers, `full_attention_interval: 4` → **48 Gated DeltaNet + 16 full attention**, vocab 248320, native VL |
| MTP | embedded in the quant: `qwen35.nextn_predict_layers = 1`, `blk.64.nextn.*` present (verified by reading the GGUF, no separate module needed) |
| GPU | 1 × RTX 3090 24 GB, driver 610.43.02, **420 W default limit, reset to stock for the primary matrix** — the card was found overclocked (mem +400 MHz, core +100 MHz, 450 W) and the first Phase A run was discarded because of it; see [`docs/GPU_AS_FOUND.md`](docs/GPU_AS_FOUND.md) |
| host | Debian 13, kernel 6.12, i9-13900, 31 GB RAM |
| engine | llama.cpp built from source, CUDA 13.3, `CMAKE_CUDA_ARCHITECTURES=86`, two trees with identical flags |
| trees | `master` @ `c060ca9` (build 200) · **PR #27342** (DFlash2, unmerged) @ `d1a522f` |
| prompts | 25, balanced 3×5 → **5 per class** over code / prose / reason / chat / zh; every prompt written to exceed the 400-token cap |
| sampling | greedy, full sampler chain pinned explicitly, `cache_prompt: false`, `--parallel 1` |

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

A full audit of the predecessor repo's methodology — including a re-analysis of its own committed
data showing its headline effect was understated roughly two-fold by prompt-class mixture — is in
[`docs/METHODOLOGY_AUDIT.md`](docs/METHODOLOGY_AUDIT.md).

## Results: Phase A

7 arms × 25 prompts × 5 passes = **875 measurements, 0 incidents, 0 excluded, 0 quality-flagged.**
Intervals are a paired cluster bootstrap over prompts (passes of one prompt are repeated
measures, not independent samples), on the class-stratified effect.

| arm | verify width | tok/s | vs own-tree baseline | tok/J | J per request |
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

**MTP at n-max 2 cuts the energy to produce a 400-token answer by 37 %** (3980 → 2503 J). No study
in the prior-art sweep publishes an energy figure for this model; `joule`, `tok/J` and `watt`
appear zero times in PR #27342's 60-comment thread.

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
units of a plain decode step. With `mean_len = predicted_n / (predicted_n − accepted)`:

    speedup = mean_len / k       k(w) = k0 + c·(w − 1),   w = n_max + 1

| method | widths | k0 | **c** | r² |
|---|---|---:|---:|---:|
| `draft-mtp` | 3, 4, 6 | 0.8934 | **0.2806** | 0.9998 |
| `draft-dflash` | 5, 8 | 0.7831 | **0.2761** | (2 points — meaningless) |

`c` agrees to **1.6 %** between the target's own built-in nextn head and a structurally unrelated
1.1 GB block-diffusion drafter, while `k0` differs by 14 %. The marginal cost of verifying one
more position belongs to the verification path; the fixed cost belongs to the drafter — and
DFlash2's fixed cost is the *lower* of the two. `k` varies by only 0.35–0.54 % across five prompt
classes whose acceptance rates differ by nearly tenfold.

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
**100/100 reproducible across passes** — the divergence is deterministic, not noise. Fork
positions partition the arms into exactly two stable groups by verification width
(`{3,4}` against `{5,6,8}`), identically in all five passes and shared across unrelated drafters.

This corroborates [llama.cpp #25618](https://github.com/ggml-org/llama.cpp/issues/25618) rather
than discovering anything: that thread already establishes the phenomenon, its
quantization-dependence, its drafter-independence, and a root cause on the Vulkan side. What is
still open is the **CUDA** boundary, and a width-localised boundary is what this repo can add.
See [`docs/UPSTREAM_CONTRIBUTIONS.md`](docs/UPSTREAM_CONTRIBUTIONS.md).

### Two things went wrong and are recorded, not smoothed over

1. **The card arrived overclocked** (memory +400 MHz, core +100 MHz, 450 W against a 420 W
   default) while this README described it as stock. The first Phase A run was **discarded**, the
   card reset, and the harness now refuses to start on a non-stock card. `docs/GPU_AS_FOUND.md`.
2. **The completed run's process crashed** with a glibc `double free or corruption` after writing
   its last record — a harness bug (`preexec_fn` used alongside a sampling thread, now
   `start_new_session=True`). All 875 measurements survived; the final pass's derived comparisons
   were recomputed from the recorded text. `PREREGISTRATION.md`, Correction 2.

## Results: later phases

Phase R (resource response: memory bandwidth × power budget × method) is running. Its pre-flight
confirmed the assumption the design rests on: **lowering the power limit to 250 W and 175 W leaves
the memory clock at 9501 MHz, unchanged**, so the compute and bandwidth levers are genuinely
separable on this card.

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
  [llama.cpp #27623](https://github.com/ggml-org/llama.cpp/issues/27623) — is a separate phase.
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
