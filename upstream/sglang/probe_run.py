"""Run the registered malformed-tree test against a chosen build of eagle_utils.cu.

Stubs the two sglang imports the test file needs so it can run without the package, then points
it at the probe extension's op instead of torch.ops.sgl_kernel.
"""
import enum, sys, types, unittest, torch
from torch.utils.cpp_extension import load

variant = sys.argv[1] if len(sys.argv) > 1 else "after"
only = sys.argv[2] if len(sys.argv) > 2 else None

class TreeMaskMode(enum.IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2

eu = types.ModuleType("sglang.srt.speculative.eagle_utils"); eu.TreeMaskMode = TreeMaskMode
ci = types.ModuleType("sglang.test.ci.ci_register")
ci.register_cuda_ci = lambda **k: None
ci.register_amd_ci = lambda **k: None
for name, mod in [("sglang", types.ModuleType("sglang")),
                  ("sglang.srt", types.ModuleType("sglang.srt")),
                  ("sglang.srt.speculative", types.ModuleType("sglang.srt.speculative")),
                  ("sglang.srt.speculative.eagle_utils", eu),
                  ("sglang.test", types.ModuleType("sglang.test")),
                  ("sglang.test.ci", types.ModuleType("sglang.test.ci")),
                  ("sglang.test.ci.ci_register", ci)]:
    sys.modules[name] = mod

load(name=f"eagle_probe_{variant}",
     sources=[f"/tmp/eagle_probe/eagle_{variant}.cu", "/tmp/eagle_probe/binding.cc"],
     extra_cuda_cflags=["-O3", "-arch=sm_86", "-std=c++17"],
     extra_cflags=["-O3", "-std=c++17"],
     extra_include_paths=["/tmp/eagle_probe/include"],
     build_directory=f"/tmp/eagle_probe/build_{variant}", verbose=False)

sys.path.insert(0, "/tmp/eagle_probe")
import test_build_eagle_tree_malformed as T
_orig = T._run
T._run = lambda p, s, sl, m, op=None: _orig(
    p, s, sl, m, torch.ops.eagle_probe.build_tree_kernel_efficient)

print(f"### variant={variant}  op=torch.ops.eagle_probe")
suite = unittest.TestLoader().loadTestsFromName(only, T) if only else \
        unittest.TestLoader().loadTestsFromModule(T)
res = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if res.wasSuccessful() else 1)
