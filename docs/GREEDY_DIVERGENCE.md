# Greedy output divergence

Extracted from the README so the front page stays readable. Part of
[`thc1006/qwen3.8-speculative-decoding-rtx3090`](https://github.com/thc1006/qwen3.8-speculative-decoding-rtx3090).

## Losslessness

Speculative arms are byte-identical to their baseline on only 25-30 of 125 prompt-passes:
**76-80 % of requests diverge**, forking at a median 23 % into the text. Every arm is nonetheless
**100/100 reproducible across passes**, so the divergence is deterministic rather than noise.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../analysis/plot_width_partition_dark.png">
  <img alt="Five-by-five matrix giving the share of 25 prompts on which two speculative arms show the same first-divergence or censoring signature. Arms at verification widths 3 and 4 match on 100 percent, of which 24 points are prompts where neither arm diverged inside the 400-token window; widths 5, 6 and 8 match on 100 percent, of which 20 points are censored; across the two groups agreement falls to 44 percent, of which 16 points are censored. The blocks span both drafters." src="../analysis/plot_width_partition.png">
</picture>

Fork positions partition the arms into exactly two groups by verification width, `{3,4}` against
`{5,6,8}`, identically in all five passes. **The grouping crosses drafters**: width 5 and width 8
are DFlash2 while width 6 is the built-in MTP head, and all three agree with each other on every
prompt. So drafter identity does not predict the grouping and verification width does. That
boundary co-occurs with the CUDA `calc_nwarps` table switching `ncols_dst` from four warps to two.

Two things qualify that, and both were found by this repo looking for them.

**The intervention does not support the warp count as the cause.** Three builds of the same
revision, differing only in that table, were pre-registered with their outcomes written down
first. The forced-up build passes every validity gate - the greedy baseline is byte-identical
across builds on 25 of 25 prompts, the widths it did not touch are identical on 50 of 50, the
widths it did touch differ on 60 of 75, and disassembly confirms the edit changed exactly the
kernels it should and no others. But of the 18 prompts that can discriminate, the registered
prediction that widths 5, 6 and 8 adopt the `{3,4}` fork positions held on **3**. The forced-down
direction was void on its first attempt for a reason worth stating: its table row included width
1, and a drafter decodes one token at a time, so it perturbed every arm through its drafter.

That set has since been replaced by four builds from a single cmake configure, which is what the
first attempt lacked: its control had been built under a different configure, and a reconfigure
regenerates `flags.make` and recompiles every `ggml-base` source, which is enough to move a
width-1 greedy baseline that shares byte-identical kernel machine code. The fourth build is a
second stock one, and it is the control the first set never had. **control and control2 agree on
0 of 6202 SASS kernels differing and on 150 of 150 outputs byte for byte**, so the build is
deterministic and a difference from a forced build is the table.

The result is a null, and a clean one. Disassembly shows the edit reached the machine code and
only there: forced-up differs from control in 92 of 6202 kernels, all `mul_mat_vec_q`, at template
widths 5 to 8; forced-down2 in 46, at 3 and 4. `ggml_cuda_should_use_mmvq` falls through to
`ne11 <= MMVQ_MAX_BATCH_SIZE` on Ampere for every quantized type, so those kernels are the ones
that run. `test-backend-ops perf` puts the effect on the kernel at **+13.6 % at width 5 and
+26.7 % at width 8**, against a rebuild noise floor of 0.17 %. And the output does not move: **0
of 75 records differ for forced-up, 0 of 50 for forced-down2**.

So the warp count changes this kernel a great deal and changes no output byte. A fork position is
a property of the text, and a mechanism that cannot change the text cannot change where two texts
diverge. **The warp count is out as the cause**, on those grounds rather than on a measured
absence, and the co-occurrence above is reported as co-occurrence. What else changes at that width
is open.

**Every agreement is censored, not some of them.** An earlier version of this paragraph measured
the window in characters and reported 15 of 25 prompts censored, with the partition checked against
the 10 that were not. The design fixes the window in tokens, and characters per token span 1.36 to
6.17 across these prompts, so that split was an artefact of the wrong unit. Measured in tokens,
`harness/truncation_audit.py` gives 490 of 750 Phase A records diverged, 260 right-censored, and
**0 that reached EOS**. No record anywhere in this study stopped on its own, so no identity here is
exact: every one means "did not diverge within 400 tokens". There is no clean subset, because the
censoring is uniform, and the robustness check the earlier text claimed cannot be run on this data
at all. Forks resolve between token 6 and token 334, the latest at 83 % of the window. What settles
it is a larger budget, which is TODO.md item D2.

This corroborates [llama.cpp #25618](https://github.com/ggml-org/llama.cpp/issues/25618) rather
than discovering anything: that thread already establishes the phenomenon, its
quantization-dependence, its drafter-independence, and a root cause on the Vulkan side. What is
still open is the **CUDA** boundary, and a width-localised boundary is what this repo can add -
now with the intervention result attached, which points away from the mechanism the thread
proposes.
[llama.cpp #26750](https://github.com/ggml-org/llama.cpp/issues/26750) asks the same question on
Blackwell; this card is sm_86 and cannot answer it. See
[`UPSTREAM_CONTRIBUTIONS.md`](UPSTREAM_CONTRIBUTIONS.md).

