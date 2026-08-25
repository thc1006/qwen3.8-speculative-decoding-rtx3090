# Does `embd_layer_inp` row `i` belong to batch token `i`?

The evidence behind the llama.cpp `output_reorder()` change. The headline numbers get quoted in
the PR notes, so the harness that produced them lives here rather than being deleted with the
scratch build it ran in.

## The question

`extract_layer_inputs()` writes rows in **ubatch** order:
`dst_offset = token_offset * row_floats`, called per ubatch with `token_offset = n_tokens_prev`.
`common/speculative.cpp:1115` reads them by **batch** index:
`layer + (i_batch_beg[seq_id] + offset + i) * n_embd_tgt`.

Those two orders are the same only when the splitter does not regroup. `split_equal` regroups
whenever a batch spans more than one ubatch or interleaves sequences.

## Why it is decidable without model arithmetic

For `LLM_ARCH_LLAMA`, `res->t_layer_inp[0]` is `inpL` straight out of `build_inp_embd`
(`src/models/llama.cpp:108` and `:127`) with nothing between them touching `inpL`. That is a pure
embedding lookup, so a layer-0 row depends on the token id and on nothing else: not the position,
not the sequence, not the batch shape. Two rows holding the same token are bit-identical, so the
check needs no tolerance, and a reference table can be built by decoding each id once in a single
sequence where nothing is permuted.

## Running it

`llamacpp-master/` and `llamacpp-dflash2/` are the measurement binaries for this study and must
not be rebuilt. Use a separate checkout.

```sh
git clone https://github.com/ggml-org/llama.cpp /tmp/probe && cd /tmp/probe
git checkout --detach fc62ba7
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release   # CPU is enough; the model is synthetic
# paste probe.cpp.inc above main() in tests/test-llama-archs.cpp, then in main():
#     if (getenv("LAYER_INP_PROBE")) { return probe_layer_inp_order(seed); }
cmake --build build --target test-llama-archs -j
LAYER_INP_PROBE=1 ./build/bin/test-llama-archs -s 4242
```

Then apply each variant to `src/` and rebuild between runs.

## What it printed

400 randomly generated batches: 1-4 sequences, 1-5 tokens each, `n_ubatch` drawn from
{1,2,3,4,8,16,32}, random output flags, sequences round-robined at random with each sequence's
positions kept ascending. A case counts as a failure if any row is not the layer-0 input of the
token at that batch position.

| variant | failures / 400 |
|---|---|
| upstream `fc62ba7` | 210 |
| remove the `embd_layer_inp` swap and nothing else | **231, worse than upstream** |
| permute by token index (the change) | **0** |

Seed 4242. The last row was rerun at seeds 1, 7, 99 and 12345: 2000 cases, 0 failures.

## The four fixed cases that name the mechanism

Token ids, batch order is `10 11 20 21`:

| case | upstream | remove-only | the change |
|---|---|---|---|
| contiguous, all outputs, `n_ubatch=8` | `10 11 20 21` | `10 11 20 21` | `10 11 20 21` |
| contiguous, all outputs, `n_ubatch=2` | `10 11 20 21` | `10 20 11 21` | `10 11 20 21` |
| seq-1-first, partial outputs, `n_ubatch=8` | `20 21 10 11` | `20 21 10 11` | `20 21 10 11` |
| contiguous, outputs on positions 1 and 2, `n_ubatch=2` | `20 10 11 21` | `10 20 11 21` | `10 11 20 21` |

Row 2 is why removal alone is not the fix: upstream is **correct** there. With
`n_outputs == n_tokens` the output permutation happens to equal the token permutation, so the
existing swap is load-bearing. Row 4 is where upstream breaks: with `n_outputs < n_tokens` the
swap indices address output rows while the buffer holds token rows, and the result is neither
ubatch nor batch order.

## Not covered

Unmasked `embd_nextn` takes the same treatment on structural grounds and is not measured here.
A synthetic model cannot carry nextn tensors: `llama_model_saver::add_tensors_from_model` copies
them from an already-loaded model, so a from-scratch gguf has none and the loader aborts on
`GGML_ASSERT(buft != nullptr)`. The grounds are that `output_reserve` sizes unmasked `embd_nextn`
`n_embd_out * n_batch` against masked's `n_embd_out * n_outputs_max`, the decode loop writes it at
`masked ? n_outputs_prev : n_tokens_prev`, and `common/speculative.cpp:607-609` reads it and
`embd_layer_inp` in one loop with the same `i * n_embd_tgt` stride.
