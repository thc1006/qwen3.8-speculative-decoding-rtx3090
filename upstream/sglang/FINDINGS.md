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

The NaN path is the one that needs no malformed input at all. A single NaN reaching `prob_acc`
makes both accept comparisons false, because every comparison against NaN is false. The accept
branch is the only place `prob_acc` is reset, so it stays NaN, so no candidate is ever accepted,
so the walk runs until the sibling chain reaches -1. If it does not, the kernel spins forever,
which is what 100 % utilisation at 25 W looks like.

That also explains why the same symptom appears on Ampere (#35822), Hopper (#33549) and AMD
(#29347). Data-dependent non-termination is not an architecture property.

Transcribed to CPU in `loop_termination.py`, which needs no GPU because the claim is about
control flow:

| input | result |
|---|---|
| well formed tree | terminates in 2 steps |
| sibling cycle 1 to 3 to 1 | does not terminate |
| self-referential sibling | does not terminate |
| **well formed tree, one NaN in `target_probs`** | **does not terminate** |

The reporter runs `--kv-cache-dtype fp8_e5m2` on sm_86, which has no native FP8, and reports that
short requests pass while longer ones hang. A low-range float format holding a growing cache is a
plausible NaN source, and nothing between it and this loop checks.

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

The inner search has no not-found case. When it falls through, `cur_position` equals
`draft_token_num`, and the next iteration uses that to **write** `tree_mask` one past the end.
An out-of-bounds write here can corrupt the very arrays the sampling kernel then walks, so the
two loops are not independent: this one is a candidate source of the malformed tree the other one
cannot survive.

Reachability is not established. The claim here is what the code does when the search fails, not
that it does fail in the reporter's run.

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
