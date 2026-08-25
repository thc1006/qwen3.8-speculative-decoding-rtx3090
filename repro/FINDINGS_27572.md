# llama.cpp #27572 on CUDA: what was tested, and what could not be

Updated 2026-08-25. Host B, RTX 3090 (350 W SKU), CUDA 12.0, llama.cpp `c060ca9`, target
`Qwen3.8-27B-UD-Q4_K_XL`, `--spec-type draft-mtp --spec-draft-n-max 4`, `-ctk q4_0 -ctv q4_0`.

The report is on HIP/gfx1151: draft acceptance collapses to exactly 0.0 under `-np N` when
concurrent requests carry long prompts. The open question here was whether CUDA has it too.

**The cause is not settled, and this file no longer states one.** The issue title and its first
account named an async device-to-host copy of `t_h_nextn` racing a later graph over the same
buffer. The reporter's own probe refuted that: reading the device tensor at the spec-hook entry
found it already NaN on 647 of 652 probes, so the graph was producing NaN and the copy was
delivering it faithfully. That rules out the extract side, which is what the title assumed.

The account that replaced it is a write-after-read on the graph inputs: `set_inputs` for ubatch
k+1 writes into tensors ubatch k's graph is still reading, because `graph_compute_async` does not
wait and the guard that would is gated on `cparams.pipeline_parallel`, which needs more than one
device. That fits the measurements and is not proven either. Two reasons to keep it open. The
fix that removes the symptom is `ggml_backend_sched_synchronize`, a fence over every backend, so
its working does not single out the input write. And the reason offered for CUDA being safe, that
the legacy default stream synchronizes implicitly, does not describe this code: the copy uses
`cudaStreamPerThread`, which the CUDA docs say "does not synchronize with other streams (just
like explicitly created streams)", the compute streams are created `cudaStreamNonBlocking`, which
the legacy stream excludes anyway, and `cudaStreamLegacy` appears nowhere in `ggml-cuda.cu`. HIP
compiles the same `.cu` files. So a clean CUDA run bounds the timing; it does not show the
ordering is there.

What follows is therefore a non-reproduction on this hardware and software, stated as that.

## Tested, and no collapse in any of them

| configuration | requests | acceptance | exactly zero |
|---|---|---|---|
| `-np 4`, 256 / 1024 / 4096 / 8192 / 16384 tokens, concurrent | 4 each | 0.23-0.70 | 0 |
| `-np 4`, **19 000 tokens** (the reported length), concurrent | 4 | 0.34-0.44 | 0 |
| `-np 4`, 16 384 tokens, concurrent, **12 repetitions** | 48 | all non-zero | 0 |
| **`-np 8`**, 4 500 tokens, concurrent, `-c 40960` | 8 | 0.25-0.39 | 0 |
| `-np 1` sequential controls at every length | - | healthy | 0 |

## Not tested, and why

Two configurations the sweep appeared to cover turned out to measure nothing. Both are recorded
here because an earlier version of the reproducer counted them as clean runs.

**Prompts above about 20 000 tokens at `-np 4`.** `n_ctx` is divided across slots, so `-c 81920`
with `-np 4` gives each slot 20 480. The server refuses anything longer:

```
request (24645 tokens) exceeds the available context size (20480 tokens)
```

The 24 576 and 32 768 cases were rejected, not answered. The reproducer recorded them as empty
completions, which is half the reported symptom, so its own verdict line said "does not
reproduce" while three of five cases had never run.

**`-np 16`, at any context.** Compute buffers grow with the slot count and this model does not
start at all:

```
graph_reserve: failed to allocate compute buffers
llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers
```

Measured envelope on this card:

| | `-c 81920` | `-c 40960` | `-c 20480` | `-c 10240` |
|---|---|---|---|---|
| `-np 4` | starts, slot 20480 | - | - | - |
| `-np 8` | fails | starts, slot 5120 | starts, slot 2560 | - |
| `-np 16` | fails | fails | fails | fails |

So on 24 GB the concurrency axis for this model stops at `-np 8`, and reaching it costs context.

## What this supports

The reported acceptance collapse does not appear on CUDA in any configuration this card can hold,
including the reported prompt length at the reported slot count, and including twelve repetitions
of the longest case a single-shot run might have missed by timing.

It does not rule the collapse out above `-np 8`, or at prompts past a slot's context. Those need
a card with more memory, not more patience.
