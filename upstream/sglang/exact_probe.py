"""Make the tid == 0 row overrun visible through PyTorch tensors.

compute-sanitizer reports reads outside an *allocation*, not outside a *tensor*. PyTorch's
caching allocator rounds small blocks up to 512 bytes, so a 48-byte selected_index sits in a
512-byte block and an 8-byte overrun lands inside it, unreported. Sizing selected_index to fill
a block exactly puts the overrun past the block end, where the checker can see it.

selected_index holds bs*(draft_token_num-1) int64. 64 elements is 512 bytes, so bs=16 with
draft_token_num=5 fills one block exactly.
"""
import enum, sys, types, torch
from torch.utils.cpp_extension import load
variant = sys.argv[1]
BS, N, TOPK, DEPTH = 16, 5, 2, 3
PSTRIDE = TOPK * (DEPTH - 1) + 1

class TreeMaskMode(enum.IntEnum):
    FULL_MASK = 0; QLEN_ONLY = 1; QLEN_ONLY_BITPACKING = 2

load(name=f"eagle_probe_{variant}",
     sources=[f"/tmp/eagle_probe/eagle_{variant}.cu", "/tmp/eagle_probe/binding.cc"],
     extra_cuda_cflags=["-O3","-arch=sm_86","-std=c++17"], extra_cflags=["-O3","-std=c++17"],
     extra_include_paths=["/tmp/eagle_probe/include"],
     build_directory=f"/tmp/eagle_probe/build_{variant}", verbose=False)

dev = "cuda"
sel = torch.full((BS, N - 1), 2, dtype=torch.int64, device=dev)   # 64 int64 = 512 bytes exactly
sel[:, -1] = 0                                                    # an entry below topk ends a walk
par = torch.zeros((BS, PSTRIDE), dtype=torch.int64, device=dev)
par[:, 1] = 99                                                    # 99 is never in selected_index
seq = torch.full((BS,), 3, dtype=torch.int64, device=dev)
mask = torch.zeros(BS * N * N, dtype=torch.bool, device=dev)
pos = torch.zeros(BS * N, dtype=torch.int64, device=dev)
buf = torch.full((3, BS, N), -1, dtype=torch.int64, device=dev)

print(f"### {variant}: selected_index is {sel.numel()} int64 = {sel.numel()*8} bytes "
      f"(a 512-byte block holds exactly {512//8})", flush=True)
print(f"    element_size*numel = {sel.element_size()*sel.numel()}  "
      f"storage bytes = {sel.untyped_storage().nbytes()}", flush=True)
torch.ops.eagle_probe.build_tree_kernel_efficient(
    par, sel, seq, mask, pos, buf[0], buf[1], buf[2],
    TOPK, DEPTH, N, int(TreeMaskMode.QLEN_ONLY))
torch.cuda.synchronize()
print("    kernel returned", flush=True)
