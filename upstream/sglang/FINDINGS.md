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

## Run on hardware, 2026-08-25: both loops confirmed on sm_86

An idle RTX A6000 became available. It is **compute capability 8.6, the same as the 2x A2 in the
report**, so this is the reporting architecture rather than an approximation. Raw output in
[`hardware_sm86.txt`](hardware_sm86.txt); nvcc 12.9, `-arch=sm_86`.

### Loop 1, the sibling walk

| case | iterations as-is | hit the cap | bounded |
|---|---:|---|---:|
| acyclic, no NaN, nothing accepted | 2 | no | 2 |
| acyclic, no NaN, accepts | 2 | no | 2 |
| acyclic, **with NaN** | 2 | **no** | 2 |
| **cycle 1<->3, no NaN, no accept** | **1 000 001** | **yes** | 4 |
| cycle 1<->3, no NaN, accepts | 1 | no | 1 |
| **cycle 1<->3, with NaN** | **1 000 001** | **yes** | 4 |
| **self sibling 1 -> 1** | **1 000 001** | **yes** | 4 |

Three readings, and the factorial is what separates them. **A cycle alone is sufficient**: row 4
has no NaN anywhere and still does not terminate. **A NaN alone is not**: row 3 is acyclic, carries
a NaN, and exits in two steps. **A cycle is not sufficient either if anything in it is accepted**:
row 5 exits at the first iteration. The failing condition is a cycle with no acceptance inside it.

This matters for the thread, because the report's own hypothesis leans on NaN probabilities. NaN
is neither necessary nor sufficient here; the cycle is what wedges the walk, and a NaN only
matters because it guarantees nothing is ever accepted.

The bound releases every non-terminating case at four iterations and leaves every terminating case
**unchanged**, which is the same result `loop_fix_check.py` reached in simulation over 4 000
random valid trees.

### Loop 2, the parent walk that PR #36201 bounds

`ancestor not selected` runs 200 001 iterations, hits the cap, and reaches **offset 4 against a
row that ends at 2** — the write past the row is real, not inferred, and the runaway counter leaks
into `positions` as `200007`. `ancestor chain loops` behaves the same. With the guard both cases
terminate in one and three iterations, offsets stay inside the row, and **the two valid cases are
byte-identical with and without it**.

### The sampler walk, run without the escape hatch

`cap` is a runtime argument, so the same kernel text runs unbounded by passing `ULLONG_MAX`.
[`hang_live.cu`](hang_live.cu) extracts the kernel from `hang_repro.cu` rather than retyping it
and launches the minimal sufficient case, the 1 <-> 3 cycle with nothing accepted.

| | utilisation | power | SM clock | memory |
|---|---:|---:|---:|---:|
| idle | 0 % | **24.1 W** | 210 MHz | 15 MiB |
| **spinning** | **100 %** | **97-100 W** | **1950 MHz** | 287 MiB |
| after the process is killed | 0 % | 25 W | 210 MHz | 15 MiB |

The kernel never returned; it was still running when the process was killed, and the device
recovered fully. **One block of one thread is enough** to pin utilisation at 100 % and drive the
SM clock to its ceiling.

On the power figure, be careful. The report describes "100 % GPU utilization with very low power
draw (~25 W)" on an A2, and this is ~98 W on an A6000, so the numbers do not match and should not
be presented as if they did. What matches is the shape, and the fractions are comparable: 25 W is
about 42 % of the A2's 60 W board power, 98 W is about 33 % of the A6000's 300 W. The signature is
a device that reads as fully utilised, with its clock at maximum, drawing far less than real work
would. That is what a spin looks like and it is not what a slow kernel or a deadlock looks like.

### The sampler walk also reads out of bounds, and one case needs no malformed tree

The walk's only exit test is `cur_index != -1`; it never checks that `cur_index` indexes the row,
and `draft_token_id` is read from `candidates` and used to index `target_probs` unvalidated.
[`oob_repro.cu`](oob_repro.cu) runs each case in its own process, because a launch failure
poisons the context and later cases would otherwise never run. Under `compute-sanitizer
--tool memcheck` on sm_86:

| case | result |
|---|---|
| sibling chain entirely in range | **no error** |
| sibling entry 99, row holds 4 | invalid read, 153 bytes **after** a 128-byte allocation |
| sibling entry -7, which is not the -1 sentinel | invalid read, 56 bytes **before** a 32-byte allocation |
| **candidate token id 4096 against d = 8** | invalid read, **13 821 bytes after** a 4-byte allocation |

The control is clean and the three malformed inputs each abort the launch, so this is a tool's
verdict rather than a reading of the source.

The last row is a **separate defect with a lower precondition**. Its sibling chain is entirely
valid; all it takes is one token id in `candidates` outside the vocabulary dimension. A bounded
walk would not prevent it.

The `else` branch writes `draft_probs` at the same computed offset it reads `target_probs` from,
into a buffer of the same size, so that write is out of bounds on the same inputs. It is not
separately observed here, because the kernel dies at the read that precedes it, and it is not
claimed as observed.

### What this does and does not establish

It establishes that both loops are non-terminating on the reporting architecture, that the
proposed bounds release them, and that neither bound changes a valid input.

It does not establish that either fix resolves #35822. The chain that would connect them is:
the builder's out-of-row write corrupts a neighbouring row, which produces a malformed sibling
chain, which wedges the sampler. Every link in that chain is now measured except the middle one —
that the out-of-row write specifically turns `retrive_next_sibling` into a cycle.

That middle link is more plausible than it was. The builder's runaway walk does not overrun its
row by a little: `builder_repro` reaches offset 200 007 against a row that ends at 2, which leaves
the tensor entirely and lands wherever the caching allocator placed the next block. The same
function allocates and returns `retrive_next_sibling`. So the mechanism is not "a neighbouring
element is corrupted" but "an arbitrary distance into the arena is written", and the sibling array
is one of the things in that arena. Demonstrating it would take a run with the real allocator, not
these standalone reproducers.

Until that is shown, the two are separate defects that happen to sit in the same feature, and the
open PR still does not claim otherwise.

There is still no server-level reproduction, no TP=2, and no AWQ Qwen3.8 on this host.

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

## The fix already exists in this repository, on the CPU backend

`python/sglang/kernels/aot/csrc/cpu/spec.cpp` implements the same tree build. Its helper is
introduced with the comment "mirroring the CUDA kernel's `invalid eagle tree` printf", so it was
written against the CUDA original, and it carries guards the original does not.

| | CUDA `eagle_utils.cu` | CPU `spec.cpp` |
|---|---|---|
| width of the parent search | `cur_position < draft_token_num` | `i < sel_stride`, and `sel_stride = draft_token_num - 1` |
| not-found result | no path for it in the `tid != 0` branch | `find_parent_node` returns `-1` |
| caller handling | none in the `tid != 0` branch | `if (found < 0) { TORCH_WARN(...); break; }` |
| ancestor walk | `while (true)` | `while (position < depth)` |

The bound carries its own reason in the source:

```cpp
// A valid root-ward walk has at most `depth` steps; the bound turns a
// malformed (cyclic) tree into a warning instead of a scheduler hang.
while (position < depth) {
```

So the project already documents that a malformed cyclic tree causes a scheduler hang here, and
already bounds the walk to prevent it, on one backend. #35822 reports a scheduler hang, on the
other.

The CPU file guards the ancestor walk in both of its mask-layout branches, QLEN_ONLY and
FULL_MASK. The CUDA kernel guards neither, and guards only the separate parent lookup in its
`tid == 0` branch.

This also settles the search width independently. `sel_stride = draft_token_num - 1` is the CPU
backend's own statement of how wide `selected_index` is, which is the array the CUDA search runs
one element past.

**What this makes the PR.** Not a new theory and not a new mechanism: porting guards that one
backend in this repository already has to another that does not, with the comment explaining why
they were needed already written by whoever added them. The hardware and sanitizer results above
are then evidence of what the missing guards cost on sm_86, rather than the argument for adding
them.

## Submitted: sgl-project/sglang#36201

Opened 2026-08-24. Bounds the ancestor walk in `build_tree_efficient` by `depth`, searches only
the `draft_token_num - 1` entries `selected_index` holds, and stops when the ancestor was not
selected, which is what the CPU and Triton builders already do. Adds
`test/registered/spec/utils/test_build_eagle_tree_malformed.py`, which pins the kernel to a
reference walk over two requests and three trees.

The PR does not claim to fix #35822. It says the report is worth reading alongside, notes that
its stack is in the other kernel whose sibling walk is also unbounded, and states that a cyclic
sibling chain is not reachable through the CUDA builder's normal path.

`call-gate / pr-gate` shows red on `Require run-ci label (optional)`. That is the CI gate waiting
for a maintainer to apply `run-ci`; #31478 sits in the same state and #35872 runs only because it
has the label. Not a defect in the change.

**A mistake caught in the last check before opening.** The second request's `PARENT_VALID` was
copied from the first without accounting for its `selected_index` being `[2, 4, 0]` rather than
`[4, 2, 0]`, so its ancestor resolved to itself and the "valid" case was a third looping tree.
`test_valid_chain` was not testing a valid chain, and the PR body would have claimed a
before-equals-after result that did not hold. Fixed and re-measured before the PR was opened.

That run also showed the same absent-ancestor tree terminating in two iterations with one request
and running away with two. Whether the unpatched walk stops depends on what the out-of-bounds
read returns, which is stated in the PR because it is why this is awkward to reproduce.

## Still open here

The sibling walk in `TreeSpeculativeSamplingTargetOnly` is unbounded in the same way, verified
non-terminating on sm_86 on a cyclic chain. It is deliberately not in #36201: #35771 is already
open against that kernel's accept condition, and a second change to the same lines would collide.
