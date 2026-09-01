# The two sibling walks: what is proven and what is not

`VerifyTreeGreedy` in `eagle_utils.cu` and `TreeSpeculativeSamplingTargetOnly` in
`speculative_sampling.cuh` both walk `retrive_next_sibling` with `while (cur_index != -1)` and no
other exit, and neither checks that `cur_index` is a position in the request's row before using it
as one. The sampling kernel additionally uses the candidate token id to index a
vocabulary-strided array.

Patch: `0002-bound-sibling-walks.patch`. Both get a hop bound and a range check; the sampler gets
the vocabulary check as well. It runs the walk on all 1024 threads of the block, and
`compute-sanitizer` counts two errors per malformed case and prints one, saying so itself in
`hardware_sm86_oob.txt`: `1 errors were not printed. Use --print-limit option`. The number
of messages is the tool's print limit, not a count of threads.

## Proven

**Non-termination on a cycle, on the real `VerifyTreeGreedy`.** Both kernels extracted from the
two source files into one program and run on the same trees. On a two-node sibling cycle with
nothing acceptable, the unfixed kernel does not return and the process has to be killed at a
45-second timeout; the fixed kernel returns and reports. Host B, RTX 3090, sm_86, CUDA 12.0.

**No behaviour change on well-formed trees.** 20 000 randomly generated trees, built the way the
builder builds them so the chains are strictly increasing and acyclic, comparing `predicts`,
`accept_index` and `accept_token_num`: **0 differing outputs**.

**The sampler walk, separately.** `hang_repro.cu` runs its loop uncapped on an RTX A6000: 100 %
utilisation, 1935 MHz, 119 W against a 25 W idle, never returns. `oob_repro.cu` under
`compute-sanitizer` reports invalid reads for an out-of-range sibling (153 bytes past a 128-byte
allocation), for a negative sibling that is not the -1 sentinel (56 bytes before a 32-byte
allocation), and for an out-of-vocabulary candidate id (13 821 bytes past a 4-byte allocation),
with a clean control.

## Not proven, and why

**The out-of-range read in `VerifyTreeGreedy` itself.** The eight-byte read lands past the
allocation but inside the same page, so it does not fault, and host B's `compute-sanitizer` is a
distro stub at `/usr/bin` without its injection library. Running it plainly returns
`sync: no error`, which is not evidence either way. The same read is demonstrated for the
sampling kernel, which does the identical thing to the identical array, so the mechanism is
established; this particular kernel's instance of it is not, on this host.

Queued for host C, whose CUDA 12.9 toolkit has a working sanitizer, once the forced-warp
intervention releases the card.

## Not claimed

That either walk causes any reported hang. #35822's stack is in
`TreeSpeculativeSamplingTargetOnly`, and the signature it describes matches what an unbounded spin
looks like, but nothing here shows the request that hung carried a malformed sibling chain.
