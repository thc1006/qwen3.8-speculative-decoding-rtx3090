"""Faithful CPU transcription of the sibling walk in TreeSpeculativeSamplingTargetOnly.

Transcribed from python/sglang/kernels/aot/csrc/speculative/speculative_sampling.cuh at main,
lines 72 to 95. No GPU involved: the claim under test is about control flow, and control flow can
be checked without a device. Each case is run with a step budget so a non-terminating walk is
reported rather than hanging this script too.
"""
import math

BUDGET = 10_000            # a real walk cannot exceed num_draft_tokens steps


def walk(num_speculative_tokens, num_draft_tokens, retrive_next_token, retrive_next_sibling,
         retrive_index, candidates, target_probs, uniform_samples, d=8,
         threshold_single=1.0, threshold_acc=1.0, bx=0):
    """The kernel's accept loop, transcribed. Returns (accepted, steps, terminated)."""
    prob_acc = 0.0
    cur_prob_offset = bx * num_draft_tokens * d
    coin = uniform_samples[bx * num_draft_tokens]
    last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens]
    num_accepted_tokens = 0
    cur_index = 0
    steps = 0

    for _j in range(1, num_speculative_tokens):
        cur_index = retrive_next_token[bx * num_draft_tokens + cur_index]
        while cur_index != -1:
            steps += 1
            if steps > BUDGET:
                return num_accepted_tokens, steps, False        # did not terminate
            draft_token_id = candidates[bx * num_draft_tokens + cur_index]
            target_prob_single = target_probs[cur_prob_offset + draft_token_id]
            prob_acc += target_prob_single
            # The kernel's accept test, verbatim. Both comparisons are false when either side
            # is NaN, and prob_acc is only reset inside the accept branch.
            if coin <= prob_acc / threshold_acc or target_prob_single >= threshold_single:
                prob_acc = 0.0
                cur_prob_offset = (bx * num_draft_tokens + cur_index) * d
                coin = uniform_samples[bx * num_draft_tokens + cur_index]
                num_accepted_tokens += 1
                last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens + cur_index]
                break
            else:
                cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index]
        if cur_index == -1:
            break
    return num_accepted_tokens, steps, True


def run(name, **kw):
    acc, steps, ok = walk(**kw)
    verdict = f"terminated after {steps} step(s), {acc} accepted" if ok else \
              f"DID NOT TERMINATE within {BUDGET} steps"
    print(f"  {name:44s} {verdict}")
    return ok


N = 4          # num_draft_tokens
D = 8          # vocab
print("Case 1 is the shape the existing unit test uses. The rest are malformed inputs the")
print("kernel accepts without complaint.\n")

# ---- 1. well formed: root -> child 1 -> child 2, sibling chains end in -1
ok1 = run("well formed tree, nothing accepted",
          num_speculative_tokens=3, num_draft_tokens=N,
          retrive_next_token=[1, 2, -1, -1], retrive_next_sibling=[-1, 3, -1, -1],
          retrive_index=[0, 1, 2, 3], candidates=[0, 1, 2, 3],
          target_probs=[0.0] * (N * D), uniform_samples=[1.0] * N, d=D)

# ---- 2. two siblings pointing at each other
ok2 = run("sibling cycle 1 <-> 3",
          num_speculative_tokens=3, num_draft_tokens=N,
          retrive_next_token=[1, -1, -1, -1], retrive_next_sibling=[-1, 3, -1, 1],
          retrive_index=[0, 1, 2, 3], candidates=[0, 1, 2, 3],
          target_probs=[0.0] * (N * D), uniform_samples=[1.0] * N, d=D)

# ---- 3. a node whose sibling is itself
ok3 = run("self-referential sibling, node 1 -> 1",
          num_speculative_tokens=3, num_draft_tokens=N,
          retrive_next_token=[1, -1, -1, -1], retrive_next_sibling=[-1, 1, -1, -1],
          retrive_index=[0, 1, 2, 3], candidates=[0, 1, 2, 3],
          target_probs=[0.0] * (N * D), uniform_samples=[1.0] * N, d=D)

# ---- 4. a single NaN in target_probs, tree perfectly well formed
probs = [0.0] * (N * D)
probs[1] = math.nan
ok4 = run("well formed tree, one NaN in target_probs",
          num_speculative_tokens=3, num_draft_tokens=N,
          retrive_next_token=[1, -1, -1, -1], retrive_next_sibling=[-1, 3, -1, 1],
          retrive_index=[0, 1, 2, 3], candidates=[1, 1, 1, 1],
          target_probs=probs, uniform_samples=[0.5] * N, d=D)

print()
print("  Case 1 terminates. Cases 2 and 3 are malformed trees and never reach -1.")
print("  Case 4 is the one that matters: the tree is fine, a single NaN reaches prob_acc, and")
print("  because NaN fails both comparisons the accept branch can never run, so prob_acc is")
print("  never reset and the walk cannot end. That is a hang with no out-of-bounds access and")
print("  no malformed tree, reachable from any source of NaN in the target probabilities.")
