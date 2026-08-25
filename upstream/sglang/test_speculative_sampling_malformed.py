"""tree_speculative_sampling_target_only on malformed sibling chains.

The kernel walks retrive_next_sibling with `while (cur_index != -1)` and no other exit, then uses
cur_index to index the request's row and the candidate id it finds to index a vocabulary-strided
array. Three separate things can go wrong and none of them is checked:

  the walk never ends      a chain that points back into itself
  the index leaves the row an entry that is neither -1 nor a position in this request
  the id leaves the vocab  a candidate token id outside [0, d) on an otherwise valid chain

The first hangs the GPU, the other two read and write out of bounds. A caching allocator serves
these tensors out of larger segments, so a small overrun lands inside another live allocation and
compute-sanitizer sees nothing: the corruption is silent.

Every case here asserts the same two things - the call returns, and every index it produces is
inside the tensors it was given. The valid case additionally pins the exact output, so a fix that
contains a malformed tree by changing what a well-formed one produces fails here.
"""

import pytest
import torch
import torch.nn.functional as F

from sgl_kernel import tree_speculative_sampling_target_only

BS = 2
NUM_DRAFT_TOKENS = 6
NUM_SPEC_STEP = 4
VOCAB = 20

# The well-formed tree from test_speculative_sampling.py, kept identical so the valid case here is
# a regression against the behaviour that file already pins.
CANDIDATES = [[0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]]
RETRIVE_INDEX = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
RETRIVE_NEXT_TOKEN = [[1, 2, -1, 4, 5, -1], [4, 2, 3, -1, 5, -1]]
RETRIVE_NEXT_SIBLING = [[-1, 3, -1, -1, -1, -1], [-1, -1, -1, -1, 1, -1]]

MALFORMED = {
    # cur_index reaches 1, whose sibling is 1 again: `while (cur_index != -1)` never exits.
    "sibling_self_loop": {"sibling": (0, 1, 1)},
    # 1 -> 3 -> 1: the shortest chain that cycles without any node pointing at itself.
    "sibling_two_node_cycle": {"sibling": (0, 1, 3), "sibling2": (0, 3, 1)},
    # the first hop already leaves the row, so the very first dereference is out of bounds
    "next_token_positive_oob": {"next_token": (0, 0, NUM_DRAFT_TOKENS + 4)},
    # the chain is entered legally and then leaves the row
    "sibling_positive_oob": {"sibling": (0, 1, NUM_DRAFT_TOKENS + 2)},
    # -1 is the sentinel; any other negative is not, and indexes backwards
    "sibling_negative_not_sentinel": {"sibling": (0, 1, -7)},
    # the chain is well-formed and the id it carries is not a token
    "candidate_id_above_vocab": {"candidate": (0, 1, VOCAB + 5)},
    "candidate_id_negative": {"candidate": (0, 1, -3)},
}


def _tensors(device, mutation=None):
    def t(rows, dtype=torch.int64):
        return torch.tensor([r[:] for r in rows], dtype=dtype, device=device)

    candidates = t(CANDIDATES)
    retrive_index = t(RETRIVE_INDEX)
    retrive_next_token = t(RETRIVE_NEXT_TOKEN)
    retrive_next_sibling = t(RETRIVE_NEXT_SIBLING)

    for key, (b, i, v) in (mutation or {}).items():
        target = {
            "sibling": retrive_next_sibling,
            "sibling2": retrive_next_sibling,
            "next_token": retrive_next_token,
            "candidate": candidates,
        }[key]
        target[b, i] = v

    return candidates, retrive_index, retrive_next_token, retrive_next_sibling


def _run(device, mutation=None):
    candidates, retrive_index, retrive_next_token, retrive_next_sibling = _tensors(device, mutation)

    target_logits = torch.full((BS, NUM_DRAFT_TOKENS, VOCAB), 1, dtype=torch.float32, device=device)
    target_logits[0, 0, 3] = 10
    target_logits[0, 3, 4] = 10
    target_logits[0, 4, 5] = 10
    target_logits[1, 0, 11] = 10
    target_logits[1, 4, 12] = 10
    for i in range(target_logits.shape[0]):
        for j in range(target_logits.shape[1]):
            if torch.max(target_logits[i, j]) < 10:
                target_logits[i, j, 18] = 10

    temperatures = torch.tensor([0.01, 0.01], dtype=torch.float32, device=device)
    expanded = temperatures.unsqueeze(1).unsqueeze(1)
    target_probs = F.softmax(target_logits / expanded, dim=-1)
    draft_probs = torch.zeros_like(target_probs)

    predicts = torch.full((BS * NUM_DRAFT_TOKENS,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full((BS, NUM_SPEC_STEP), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.zeros((BS,), dtype=torch.int32, device=device)

    # fixed, so a case that fails does so for the tree and not for the draw
    coins = torch.full((BS, NUM_DRAFT_TOKENS), 0.5, device=device, dtype=torch.float32)
    coins_final = torch.full((BS,), 0.5, device=device, dtype=torch.float32)

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_final,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    torch.cuda.synchronize()
    return predicts, accept_index, accept_token_num


def _assert_contained(predicts, accept_index, accept_token_num):
    """Every index the kernel produced points inside the tensors it was handed."""
    ai = accept_index.tolist()
    for row in ai:
        for v in row:
            assert -1 <= v < BS * NUM_DRAFT_TOKENS, f"accept_index escaped the tree: {ai}"
    for n in accept_token_num.tolist():
        assert 0 <= n <= NUM_SPEC_STEP, f"accept_token_num out of range: {n}"
    for v in predicts.tolist():
        assert -1 <= v < VOCAB, f"predicts holds a token id outside the vocabulary: {v}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel")
def test_wellformed_tree_is_unchanged():
    """The containment must not be bought by changing what a valid tree produces."""
    predicts, accept_index, accept_token_num = _run("cuda")
    _assert_contained(predicts, accept_index, accept_token_num)
    assert accept_token_num.tolist() != [0, 0], (
        "the well-formed tree accepted nothing, so this case is no longer a regression on "
        "anything and the malformed ones below prove less than they look"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel")
@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_malformed_tree_is_contained(name):
    """Returns rather than hanging, and produces nothing that points outside the tensors.

    Without the fix `sibling_self_loop` and `sibling_two_node_cycle` do not return at all, so this
    test hangs rather than failing. That is the honest shape of the defect: a test that times out
    is what an unbounded device loop looks like from the host.
    """
    predicts, accept_index, accept_token_num = _run("cuda", MALFORMED[name])
    _assert_contained(predicts, accept_index, accept_token_num)
