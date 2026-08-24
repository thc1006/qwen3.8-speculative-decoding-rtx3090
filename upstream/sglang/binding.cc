// Registers the real build_tree_kernel_efficient from eagle_utils.cu as a torch op, so a memory
// checker sees the shipped kernel through the shipped host wrapper rather than a transcription.
// The .cu file is copied unmodified from the tree under test.
#include <torch/library.h>
#include <ATen/ATen.h>

void build_tree_kernel_efficient(
    at::Tensor parent_list, at::Tensor selected_index, at::Tensor verified_seq_len,
    at::Tensor tree_mask, at::Tensor positions, at::Tensor retrive_index,
    at::Tensor retrive_next_token, at::Tensor retrive_next_sibling,
    int64_t topk, int64_t depth, int64_t draft_token_num, int64_t tree_mask_mode);

TORCH_LIBRARY(eagle_probe, m) {
  m.def("build_tree_kernel_efficient(Tensor parent_list, Tensor selected_index, "
        "Tensor verified_seq_len, Tensor! tree_mask, Tensor! positions, Tensor! retrive_index, "
        "Tensor! retrive_next_token, Tensor! retrive_next_sibling, int topk, int depth, "
        "int draft_token_num, int tree_mask_mode) -> ()");
}
TORCH_LIBRARY_IMPL(eagle_probe, CUDA, m) {
  m.impl("build_tree_kernel_efficient", &build_tree_kernel_efficient);
}
