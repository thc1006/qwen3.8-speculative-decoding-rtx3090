# sglang #35822: what the code says, before any hardware test

Static analysis only. Nothing here has been run on a GPU yet, and the write-up says so wherever
it matters. Sources are `main` unless a version is named.

## What the issue is not

The obvious first guess is that this duplicates the known EAGLE tensor-parallel divergence, where
ranks accept different numbers of drafts and the next collective deadlocks. It does not.

That bug lives in the **greedy** branch of `eagle_sample`, which computed the accept set from a
per-rank `torch.argmax` without broadcasting. PR #31478 is the open fix, unmerged since
2026-07-17. Checking the tag the reporter actually ran:

| version | `tp_group.broadcast` calls | where |
|---|---:|---|
| v0.5.16 | 3 | sampling branch only |
| **v0.5.17** (the reporter's) | 3 | sampling branch only |
| main | 6 | greedy branch and sampling branch |

The reporter's py-spy stack is `tree_speculative_sampling_target_only`, which is the **sampling**
branch, and that branch already broadcasts on v0.5.17. So the missing greedy broadcast cannot be
what is hanging them. Reporting this as a duplicate would have sent the thread the wrong way.

## The two unbounded loops

### 1. `TreeSpeculativeSamplingTargetOnly`, the sibling walk

`python/sglang/kernels/aot/csrc/speculative/speculative_sampling.cuh`, lines 72 to 95:

```c
for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
  cur_index = retrive_next_token[bx * num_draft_tokens + cur_index];
  while (cur_index != -1) {                                   // only exit besides accept
    ...
    prob_acc += target_prob_single;
    if (coin <= prob_acc / threshold_acc || target_prob_single >= threshold_single) {
      prob_acc = 0.;  ...  break;                             // prob_acc reset lives here only
    } else {
      cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
    }
  }
```

`cur_index` is never bounds-checked and `prob_acc` is never checked for finiteness. The file
contains no `isnan`, no `isfinite`, and no comparison of `cur_index` against `num_draft_tokens`.

**Verified on an RTX 3090, sm_86, the architecture in the report.** `hang_repro.cu` transcribes
the loop verbatim, adds an iteration counter so a non-terminating walk reports itself instead of
wedging the device, and runs the two candidate causes factorially. Built with the CUDA 13.3 nvcc,
no SGLang or sgl-kernel needed, since the loop is self-contained.

| sibling chain | NaN in `target_probs` | accepts? | iterations | outcome |
|---|---|---|---:|---|
| acyclic | no | no | 2 | terminates |
| acyclic | no | yes | 2 | terminates |
| **acyclic** | **yes** | no | **2** | **terminates** |
| **cyclic** | no | **yes** | **1** | **terminates** |
| cyclic | no | no | 1000001 | hit the cap |
| cyclic | yes | no | 1000001 | hit the cap |
| self-referential | no | no | 1000001 | hit the cap |

A first version of this test put a cycle into the row labelled as the NaN case and would have
supported the wrong conclusion, which was written down before the factorial run corrected it. The
result that survives is narrower:

- **A cycle in `retrive_next_sibling` is necessary.** Every acyclic case terminates, NaN included.
- **A cycle is not sufficient.** If any node on the cycle is accepted, `break` leaves the loop, and
  the row above shows a cyclic chain terminating in one iteration.
- **NaN is neither necessary nor sufficient on its own.** What it does is guarantee that nothing is
  ever accepted, because every comparison against NaN is false and the accept branch is the only
  place `prob_acc` is reset. That removes the accidental escape and turns a survivable cycle into
  a certain hang.

So the kernel is not the origin of the malformed data; it is the place where malformed data stops
being recoverable. The origin question points at the tree builder, which is loop 2 below.

Data-dependent non-termination is also not an architecture property, which is consistent with the
same symptom appearing on Ampere (#35822), Hopper (#33549) and AMD (#29347).

The reporter runs `--kv-cache-dtype fp8_e5m2` on sm_86, which has no native FP8, and reports that
short requests pass while longer ones hang. That is a plausible NaN source and nothing between it
and this loop checks, but on this evidence NaN alone would slow the walk rather than stop it.

### 2. `build_tree_kernel`, the parent walk

`python/sglang/kernels/aot/csrc/speculative/eagle_utils.cu`, lines 105 to 121:

```c
int cur_position = tid - 1;
while (true) {
  tree_mask[token_tree_idx + cur_position] = true;            // write
  int parent_tb_idx = selected_index[... + cur_position] / topk;
  if (parent_tb_idx == 0) break;                              // only exit
  int token_idx = parent_list[... + parent_tb_idx];
  for (cur_position = 0; cur_position < draft_token_num; ++cur_position) {
    if (selected_index[... + cur_position] == token_idx) break;
  }
  // no handling for the search finding nothing
}
```

**The same search appears twice in this file and only one copy handles failure.** The `tid == 0`
branch runs it and then checks:

```c
if (parent_position == draft_token_num) {
  printf("WARNING: invalid eagle tree!!! Detected a token with no parent token selected. "
         "Please check if the logprob has nan. The token will be ignored to keep proceeding.\n");
  continue;
}
```

The `tid != 0` branch above runs the same search with no such check, and feeds the result straight
back into the next iteration of `while (true)`.

Shapes, from `build_tree_kernel_efficient`. In QLEN_ONLY mode `tree_mask` holds
`N * bs * N` bools with `token_tree_idx = N*N*bid + N*tid + 1`, so offsets from `-1` to `N-2` are
this thread's row. `selected_index` is `top_scores_index`, indexed as
`selected_index[bid * (N - 1) + cur_position]`, so it is `N - 1` wide per batch item.

- The search bound is `cur_position < draft_token_num`, one wider than `selected_index`, so its
  last iteration reads past the row even when the search succeeds.
- A successful match at the last position leaves `cur_position = N - 1`, and the next iteration
  writes `tree_mask` at the first element of the following thread's row.
- A failed search leaves `cur_position = N`, one further again, and for the last `(bid, tid)`
  that is past the end of the allocation.
- With no `parent_tb_idx == 0` ever reached, `while (true)` does not exit.

**Reachability of a cycle, checked and not supported.** The obvious theory was that the retrieve
arrays carry stale pointers between steps. They do not: `build_tree_kernel_efficient` allocates
`torch.full((3, bs, num_verify_tokens), -1, ...)` fresh on every call, so a token skipped by the
`continue` above keeps `-1`. The builder also inserts children at the head of the list while `i`
counts downward, which makes `sibling[i] > i` for every link, so the chains it produces are
strictly increasing and cannot contain a cycle. A cycle therefore has to come from somewhere
other than this builder's normal path, and the out-of-bounds write above is the candidate.

There is a `# TODO: make them torch.empty` on that allocation. If it is ever taken, the stale
pointer theory becomes live.

## Proposed fix for loop 1, and why it is safe

Bound the sibling walk by `num_draft_tokens`. A sibling chain sits inside one level of a tree with
`num_draft_tokens` nodes in total, so a longer walk is malformed by definition.

Checked in `loop_fix_check.py`: all three non-terminating cases terminate, both well-formed cases
give identical accept sets, and **4000 randomly generated valid trees produce zero differences**,
so the bound is invisible on any input the tree builder should produce.

A NaN guard on `prob_acc` and a bounds check on `cur_index` are worth discussing separately. The
bound alone restores liveness, which is the part that matters: a slower answer beats a wedged
server, and today the watchdog kills the process at 300 s.

## What is not done

Nothing has been run on a GPU. The next step is a direct call to
`tree_speculative_sampling_target_only` with a cyclic tree and with a NaN probability, on real
hardware, to confirm the kernel hangs and that the bound releases it. That test needs a GPU that
is not busy, because a hanging kernel holds one at 100 %.

The existing unit test at `python/sglang/kernels/aot/tests/speculative/test_speculative_sampling.py`
covers one well-formed tree and has no liveness, malformed-input or NaN case.

## Hardware results, RTX 3090 sm_86, CUDA 13.3

Two standalone reproductions, neither needing SGLang, sgl-kernel, a model or a second GPU. Both
were run alongside an unrelated benchmark on the same card and did not disturb it.

### Sibling walk, `TreeSpeculativeSamplingTargetOnly`

`hang_repro.cu`. Iteration cap 1000000; a valid walk needs at most `num_draft_tokens`.

| sibling chain | NaN | accepts | iterations | bounded |
|---|---|---|---:|---:|
| acyclic | no | no | 2 | 2 |
| acyclic | no | yes | 2 | 2 |
| acyclic | yes | no | 2 | 2 |
| cyclic | no | yes | 1 | 1 |
| cyclic | no | no | 1000001 | 4 |
| cyclic | yes | no | 1000001 | 4 |
| self-referential | no | no | 1000001 | 4 |

### Parent walk, `build_tree_kernel`, the `tid != 0` branch

`builder_repro.cu`, QLEN_ONLY layout, `draft_token_num` 4, `topk` 2, `depth` 3.

| case | iterations | max offset written / row end |
|---|---:|---|
| parent present in `selected_index` | 3 | 2 / 2 |
| parent present, guarded | 3 | 2 / 2 |
| **parent absent** | 2 | **4 / 2** |
| parent absent, guarded | 1 | 2 / 2 |

Under `compute-sanitizer`:

```
memcheck,  as shipped    : 3 errors   (Invalid __global__ read of size 8 bytes,
                                       threads (1,0,0) and (2,0,0))
memcheck,  with the guard: 0 errors
synccheck, as shipped    : 0 errors
```

The 8-byte reads are `selected_index`, which is `draft_token_num - 1` wide per batch item while
the search loop runs to `draft_token_num`.

**synccheck finding nothing matters.** The starting theory for this issue was a divergent
cooperative-group barrier on Ampere. The sampling kernel's accept loop contains no barrier at
all, uses only `blockIdx.x`, and is executed identically by every thread in the block, so its
control flow is uniform by construction. When it does not terminate, no thread reaches the
`__syncthreads()` that follows, which looks like a stuck barrier from outside and is not one.

### Whether the guard changes results

The `parent present` rows are identical with and without it, 3 iterations and the same offsets.
For the sibling walk, 4000 randomly generated valid trees give identical accept sets.

## What could be sent upstream, and what could not

Verified enough to propose:

1. `build_tree_kernel`, `tid != 0` branch: handle the parent-not-found case the way the
   `tid == 0` branch already does, and stop the search at `draft_token_num - 1`, which is the
   width `selected_index` actually has. Sanitizer evidence above, and the branch being fixed is
   the only one of the two that lacks the handling.
2. `TreeSpeculativeSamplingTargetOnly`: bound the sibling walk by `num_draft_tokens`. Defensive
   rather than a proven bug fix, and it should be described that way.

Not established, and so not claimable in a PR title or description:

- That either change fixes #35822. There is no server-level reproduction here, no TP=2, and the
  reporter's exact model was never loaded.
- That a cycle in `retrive_next_sibling` occurs in practice. The CUDA builder cannot produce one
  on its normal path.
