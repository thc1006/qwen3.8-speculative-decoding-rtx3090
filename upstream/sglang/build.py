import os, sys, torch
from torch.utils.cpp_extension import load
variant = sys.argv[1] if len(sys.argv) > 1 else "after"
mod = load(
    name=f"eagle_probe_{variant}",
    sources=[f"/tmp/eagle_probe/eagle_{variant}.cu", "/tmp/eagle_probe/binding.cc"],
    extra_cuda_cflags=["-O3", "-arch=sm_86", "-std=c++17"],
    extra_cflags=["-O3", "-std=c++17"],
    extra_include_paths=["/tmp/eagle_probe/include"],
    build_directory=f"/tmp/eagle_probe/build_{variant}",
    verbose=False,
)
print(f"  built {variant}; op present:",
      hasattr(torch.ops.eagle_probe, "build_tree_kernel_efficient"))
