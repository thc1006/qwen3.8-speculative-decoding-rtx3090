"""Does bounding the sibling walk fix the hang without changing well-formed results?

The bound is num_draft_tokens. A sibling chain lives inside one level of a tree that has
num_draft_tokens nodes in total, so it cannot legitimately be longer than that, and a walk that
exceeds it is malformed by definition rather than merely unusual.
"""
import math, itertools, random

def walk(bounded, num_speculative_tokens, num_draft_tokens, retrive_next_token,
         retrive_next_sibling, retrive_index, candidates, target_probs, uniform_samples,
         d=8, threshold_single=1.0, threshold_acc=1.0, bx=0, budget=10_000):
    prob_acc = 0.0
    cur_prob_offset = bx * num_draft_tokens * d
    coin = uniform_samples[bx * num_draft_tokens]
    num_accepted_tokens = 0
    accepted = []
    cur_index = 0
    total = 0
    for _j in range(1, num_speculative_tokens):
        cur_index = retrive_next_token[bx * num_draft_tokens + cur_index]
        steps = 0
        while cur_index != -1:
            if bounded and steps >= num_draft_tokens:     # <-- the proposed guard
                cur_index = -1
                break
            steps += 1; total += 1
            if total > budget:
                return None, total          # non-terminating
            tok = candidates[bx * num_draft_tokens + cur_index]
            p = target_probs[cur_prob_offset + tok]
            prob_acc += p
            if coin <= prob_acc / threshold_acc or p >= threshold_single:
                prob_acc = 0.0
                cur_prob_offset = (bx * num_draft_tokens + cur_index) * d
                coin = uniform_samples[bx * num_draft_tokens + cur_index]
                num_accepted_tokens += 1
                accepted.append(retrive_index[bx * num_draft_tokens + cur_index])
                break
            else:
                cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index]
        if cur_index == -1:
            break
    return (num_accepted_tokens, tuple(accepted)), total

N, D = 4, 8
bad = [0.0]*(N*D); bad[1] = math.nan
CASES = {
 "well formed, nothing accepted": dict(num_speculative_tokens=3, num_draft_tokens=N,
   retrive_next_token=[1,2,-1,-1], retrive_next_sibling=[-1,3,-1,-1],
   retrive_index=[0,1,2,3], candidates=[0,1,2,3], target_probs=[0.0]*(N*D),
   uniform_samples=[1.0]*N, d=D),
 "well formed, everything accepted": dict(num_speculative_tokens=3, num_draft_tokens=N,
   retrive_next_token=[1,2,-1,-1], retrive_next_sibling=[-1,3,-1,-1],
   retrive_index=[0,1,2,3], candidates=[0,1,2,3], target_probs=[1.0]*(N*D),
   uniform_samples=[0.0]*N, d=D),
 "sibling cycle 1 <-> 3": dict(num_speculative_tokens=3, num_draft_tokens=N,
   retrive_next_token=[1,-1,-1,-1], retrive_next_sibling=[-1,3,-1,1],
   retrive_index=[0,1,2,3], candidates=[0,1,2,3], target_probs=[0.0]*(N*D),
   uniform_samples=[1.0]*N, d=D),
 "self sibling 1 -> 1": dict(num_speculative_tokens=3, num_draft_tokens=N,
   retrive_next_token=[1,-1,-1,-1], retrive_next_sibling=[-1,1,-1,-1],
   retrive_index=[0,1,2,3], candidates=[0,1,2,3], target_probs=[0.0]*(N*D),
   uniform_samples=[1.0]*N, d=D),
 "well formed tree, one NaN prob": dict(num_speculative_tokens=3, num_draft_tokens=N,
   retrive_next_token=[1,-1,-1,-1], retrive_next_sibling=[-1,3,-1,1],
   retrive_index=[0,1,2,3], candidates=[1,1,1,1], target_probs=bad,
   uniform_samples=[0.5]*N, d=D),
}
print(f"  {'case':36s} {'current':>22s} {'with the bound':>22s}")
for name,kw in CASES.items():
    a,_ = walk(False, **kw); b,_ = walk(True, **kw)
    f = lambda r: "HANGS" if r is None else f"{r[0]} accepted {r[1]}"
    print(f"  {name:36s} {f(a):>22s} {f(b):>22s}")

print("\n  randomised equivalence on well-formed trees (the bound must be invisible there)")
random.seed(11)
diff = 0; tested = 0
for trial in range(4000):
    n = random.choice([2,4,8,16])
    # build a valid tree: every node's parent index is lower, siblings chain forward, ends at -1
    nxt = [-1]*n; sib = [-1]*n
    children = {i: [] for i in range(n)}
    for node in range(1, n):
        children[random.randrange(node)].append(node)
    for parent, kids in children.items():
        if kids:
            nxt[parent] = kids[0]
            for a_, b_ in zip(kids, kids[1:]):
                sib[a_] = b_
    kw = dict(num_speculative_tokens=random.randint(2, min(5, n)), num_draft_tokens=n,
              retrive_next_token=nxt, retrive_next_sibling=sib,
              retrive_index=list(range(n)), candidates=[random.randrange(D) for _ in range(n)],
              target_probs=[random.random() for _ in range(n*D)],
              uniform_samples=[random.random() for _ in range(n)], d=D,
              threshold_single=random.uniform(0.5,1.0), threshold_acc=random.uniform(0.5,1.0))
    a,_ = walk(False, **kw); b,_ = walk(True, **kw)
    tested += 1
    if a != b: diff += 1
print(f"    {tested} random well-formed trees, results differing: {diff}")
print("    " + ("the bound never changes a well-formed result" if diff==0 else "THE BOUND CHANGES BEHAVIOUR"))
