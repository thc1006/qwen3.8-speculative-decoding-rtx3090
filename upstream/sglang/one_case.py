"""Run one (case, mask mode) pair against one build, so a hanging combination can be isolated."""
import enum, sys, types, torch
from torch.utils.cpp_extension import load
variant, case_name, mode_name = sys.argv[1], sys.argv[2], sys.argv[3]

class TreeMaskMode(enum.IntEnum):
    FULL_MASK = 0; QLEN_ONLY = 1; QLEN_ONLY_BITPACKING = 2
eu = types.ModuleType("m"); eu.TreeMaskMode = TreeMaskMode
ci = types.ModuleType("c"); ci.register_cuda_ci = ci.register_amd_ci = lambda **k: None
for n, m in [("sglang", types.ModuleType("s")), ("sglang.srt", types.ModuleType("s")),
             ("sglang.srt.speculative", types.ModuleType("s")),
             ("sglang.srt.speculative.eagle_utils", eu), ("sglang.test", types.ModuleType("s")),
             ("sglang.test.ci", types.ModuleType("s")), ("sglang.test.ci.ci_register", ci)]:
    sys.modules[n] = m
load(name=f"eagle_probe_{variant}",
     sources=[f"/tmp/eagle_probe/eagle_{variant}.cu", "/tmp/eagle_probe/binding.cc"],
     extra_cuda_cflags=["-O3","-arch=sm_86","-std=c++17"], extra_cflags=["-O3","-std=c++17"],
     extra_include_paths=["/tmp/eagle_probe/include"],
     build_directory=f"/tmp/eagle_probe/build_{variant}", verbose=False)
sys.path.insert(0, "/tmp/eagle_probe")
import test_build_eagle_tree_malformed as T
mode = getattr(TreeMaskMode, mode_name)
par, sel = (T.CROSS_ROW_PARENT, T.CROSS_ROW_SELECTED) if case_name == "cross_row" \
           else (T.CASES[case_name], T.SELECTED)
print(f"### {variant} / {case_name} / {mode_name}", flush=True)
got = T._run(par, sel, T.SEQ_LENS, mode, torch.ops.eagle_probe.build_tree_kernel_efficient)
want = T._reference(par, sel, T.SEQ_LENS)
gm = T._decode_mask(got[1], want[1].shape, T.SEQ_LENS, mode)
ok = all(torch.equal(a, b) for a, b in
         ((got[0], want[0]), (gm, want[1]), (got[2], want[2]), (got[3], want[3]), (got[4], want[4])))
print("  outputs match reference:", ok)
