"""Phase L: does speculative decoding survive the long-context decode cliff?

llama.cpp issue #27623, open 2026-08-23 with zero comments, reports that decode throughput on
this model collapses roughly 25x once the KV position passes about 80 K, from 33 tok/s at 68 K
to 1.4 tok/s at 91 K, while prompt processing stays fast at ~1300 tok/s. Reproduced there on an
RTX 4080 SUPER (sm_89) across three quants. **Its author withdrew that 25x on 2026-08-26**,
after re-measuring with eval-only rather than wall-clock timings; this phase ran before that,
so read its non-reproduction as consistent with the withdrawal rather than independent of it.

Nothing in the 2026-08-24 sweep reports a reproduction on another architecture, or the follow-up
that matters here: does speculation survive the cliff, ride it down, or make it worse? The sweep
sees what was posted, not what was run. DFlash2's advertised advantage is precisely long-context retention,
so the answer is not guessable.

Depth ladder and why it stops at 96 K. KV on this model costs ~34 KB/token at q8_0, since only
16 of its 64 layers hold KV (`full_attention_interval: 4`); the 48 Gated DeltaNet layers carry a
fixed ~75 MB of recurrent state that does not grow. Against a 17.56 GB target and the ~1.9 GB
compute buffer measured on this host:

    8 K    17.8 + 1.9 = 19.7 GB
    32 K   18.7 + 1.9 = 20.6 GB
    64 K   19.7 + 1.9 = 21.6 GB
    80 K   20.3 + 1.9 = 22.2 GB
    96 K   20.8 + 1.9 = 22.7 GB     95 % of the card
    128 K  21.9 + 1.9 = 23.8 GB     99 %, not attempted

96 K clears the reported ~80 K cliff, which is what the phase is for. Going deeper would need
q4_0 KV, and changing KV precision to reach depth would confound the two.

Two departures from the other phases, both deliberate and both declared.

`CACHE_PROMPT` is True. Every request in an arm shares the same filler, and re-prefilling 96 K
tokens per request would take about 75 s each and dominate the run. The KV that decode sees is
identical whether it was cached or recomputed, and a shared long prefix is how long context is
actually served. The energy accounting knows about this: with the cache on, the measured
request skips the prefill, so its energy is decode energy and nothing is subtracted.

`PROMPTS_PER_CLASS` is 3. Depth is the variable here, not prompt class, so the full 25-prompt
set is more than this phase needs. It cannot go lower than 3, though: the cluster bootstrap
resamples prompts within each class, and a class holding one prompt returns that same prompt
every time, contributing zero variance. At one prompt per class every interval collapses onto
its point estimate and reads as perfect precision when it actually means precision was never
estimated. Fifteen prompts keep the intervals real; `stats.Interval.width_understated` flags the
case if a class ever ends up short anyway.

Filler is real non-repeating public-domain prose cut to an exact token count by the server's own
tokenizer. Filling with a repeated paragraph would hand the drafter that repetition's
predictability and inflate acceptance for a reason unrelated to depth.

Depth is chosen by environment variable and recorded in the result:

    QWEN_L_DEPTH=65536 python3 harness/bench.py --matrix phase_l ...
"""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import Arm  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent

# depth -> (context size to allocate, approximate VRAM need in GB)
DEPTHS = {
    8192:  (12288,  19.7),
    32768: (36864,  20.6),
    65536: (69632,  21.6),
    81920: (86016,  22.2),
    98304: (102400, 22.7),
}

CONTEXT_FILLER_TOKENS = int(os.environ.get("QWEN_L_DEPTH", "8192"))
if CONTEXT_FILLER_TOKENS not in DEPTHS:
    raise RuntimeError(f"QWEN_L_DEPTH={CONTEXT_FILLER_TOKENS} is not one of {sorted(DEPTHS)}")
_ctx, REQUIRES_VRAM_GB = DEPTHS[CONTEXT_FILLER_TOKENS]

CACHE_PROMPT = True
PROMPTS_PER_CLASS = 3
MAX_TOKENS = 160        # at the reported 1.4 tok/s past the cliff this is ~114 s per request

TREES = {
    "master":  REPO / "llamacpp-master",
    "pr27342": REPO / "llamacpp-dflash2",
}
BINARIES = {k: v / "build/bin/llama-server" for k, v in TREES.items()}
MODEL = REPO / "models/target/Qwen3.8-27B-UD-Q4_K_XL.gguf"
DFLASH2_Q4 = REPO / "models/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"

COMMON_ARGS = [
    "-ngl", "999", "-c", str(_ctx), "-fa", "on",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-webui", "--parallel", "1", "--jinja", "--fit", "off",
]

_d = CONTEXT_FILLER_TOKENS // 1024
ARMS = [
    Arm(f"baseline@{_d}k", [], tree="master", note=f"no spec at {_d}K KV depth"),
    Arm(f"baseline@{_d}k-pr", [], tree="pr27342", note="same, on the DFlash2 branch"),
    Arm(f"mtp-n2@{_d}k", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
        tree="master", expects_drafter=True, note="best depth for MTP from Phase A"),
    Arm(f"dflash2-n4@{_d}k", ["--spec-type", "draft-dflash", "--spec-draft-n-max", "4",
                              "-md", str(DFLASH2_Q4)],
        tree="pr27342", expects_drafter=True, note="best depth for DFlash2 from Phase A"),
]

BASELINE_MAP = {a.name: (f"baseline@{_d}k-pr" if a.tree == "pr27342" else f"baseline@{_d}k")
                for a in ARMS}
