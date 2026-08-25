"""Phase M: dense-hybrid against MoE, same harness, same card, same prompts.

The predecessor study concluded that no llama.cpp speculative-decoding configuration is a net win
for Qwen3.6-35B-A3B on a consumer 3090. Its headline -44.6 % is DFlash with a BF16 drafter, which
this phase does not run. The arm this phase actually replicates is `draft-q35-08b-max8`, the 0.8B
draft-then-verify one, at -10.8 % raw and -21.5 % on the class-stratified estimand used here
(docs/METHODOLOGY_AUDIT.md). Anchoring on -44.6 % would fail a correct replication.

Two further things have to be said before this phase is read as being about MoE at all. Phase C
ran the same 0.8B drafter at n-max 8 against the DENSE target and measured -29.8 %
[-33.1, -26.4]: worse than the MoE arm, with no experts in the model. And the predecessor's
"acceptance near 100 %" is a counter that divided accepted by accepted; the adjacent log line
gives 115 of 214, or 53.7 %. Both remove load from the expert-saturation account below.

The predecessor's explanation was MoE expert loading:
a verification pass over K draft positions must load the union of those positions' expert sets,
and at K well below the threshold where that union saturates, the pass pays for far more expert
traffic than it saves in decode steps. Acceptance was near 100 % in every config, so the loss was
not a drafting failure.

That study could not test the one configuration its own explanation points at. Its arms were all
draft-then-verify with K from 8 to 64. MTP is self-speculation with K equal to n_max, and n_max
of 1 or 2 sits far below anything it measured. The same explanation predicts small K should be
where the MoE penalty is least. Its sibling vLLM study is the indirect evidence: MTP at k=1 on
the same 3090 came out 27.5 % faster, which was attributed to vLLM's structurally smaller K and
lighter verify path rather than to anything about MoE. Whether llama.cpp's MTP path also escapes
the penalty is the missing cell, and this phase fills it.

What makes the contrast clean here is something the GGUF metadata settles rather than an
assumption. Both models are hybrid attention with `full_attention_interval: 4`: the 27B has 64
blocks including its nextn one, the MoE 41 on the same counting. Both carry
`nextn_predict_layers: 1` and real `blk.N.nextn.*` tensors, so
both can run MTP. Both use the same 248320-token vocabulary and the same `qwen35` pre-tokenizer,
which is also what makes one 0.8B drafter vocab-compatible with both. The models therefore differ
in MoE routing and in size, not in attention design, and routing is what H4a is about.

The K ladder is the point of the arm list. MTP is sampled at n_max 1, 2, 3, 5 and 7, and
draft-then-verify at 2, 4, 6, 8 and 16, so the two paths overlap between 4 and 7 rather than
sitting in separate regimes. n_max 2 and 6 exist because widths 3 and 7 are inside the MMVQ
dispatch path while 9 and 17 are not: without them the draft-then-verify side has one usable
width and no `c` can be fitted for it at all.

Three arms run the DENSE target in the same session, against their own baseline. Until they were
added every arm here ran the MoE file, so the phase could only compare against a dense number
measured on another day, and its name was wrong. With them a pair differs in the model and in
nothing else: same harness, same card, same prompts, same build, same hour. `draft08b-n8` reproduces the predecessor's losing configuration exactly, down to the
0.8B Q4_K_M drafter and `-ngld 99`. It is there as an anchor: if this harness cannot reproduce
the loss the predecessor reported, nothing else measured here about the MoE should be believed,
and that check runs before any claim about MTP.

The cost model is what turns this into a number rather than a story. Fitting k(w) = k0 + c(w-1)
on both models gives c, the marginal cost of each additional verified position. If expert loading
is the mechanism, the MoE's c should exceed the dense model's, and by roughly the factor that
separates their net yields.

Two departures from the predecessor's setup, both forced and both declared. Context is 8192, not
its 16384, because every dense measurement in this study is at 8192 and the comparison has to be
matched on this side rather than on theirs; the MoE's KV is about 10 KB/token at q8_0 (10 of 41
layers hold KV), so this direction only frees memory. And `--draft-max` no longer exists in
llama.cpp: it was removed in favour of `--spec-draft-n-max`, which is what the anchor arm passes.

VRAM is tight. The MoE is 21.28 GiB, KV at 8192 adds about 0.08, the compute buffer about 1.9,
and the 0.8B drafter another 0.50 on the draft arms, so the worst arm sits near 23.8 of 24.0.
The predecessor ran this same file with the same drafter at 16384 context, which is more KV than
this asks for, so it fits; the capacity guard will refuse rather than OOM partway if it does not.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import Arm  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent

TREES = {"master": REPO / "llamacpp-master"}
BINARIES = {k: v / "build/bin/llama-server" for k, v in TREES.items()}

MODEL = REPO / "models/moe/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
DENSE = REPO / "models/target/Qwen3.8-27B-UD-Q4_K_XL.gguf"
DRAFT_08B = REPO / "models/draft08b/Qwen3.5-0.8B-Q4_K_M.gguf"

REQUIRES_VRAM_GB = 23.8

COMMON_ARGS = [
    "-ngl", "999", "-c", "8192", "-fa", "on",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "--no-webui", "--parallel", "1", "--jinja", "--fit", "off",
]

ARMS = [
    Arm("baseline-moe", [], tree="master", note="Qwen3.6-35B-A3B, no speculation"),

    # MTP: K = n_max, and this is the range the predecessor never reached.
    Arm("moe-mtp-n1", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"],
        tree="master", expects_drafter=True,
        note="smallest possible K; the vLLM result that flipped positive was k=1"),
    Arm("moe-mtp-n2", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
        tree="master", expects_drafter=True, note="best depth for MTP on the dense model"),
    Arm("moe-mtp-n3", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
        tree="master", expects_drafter=True),
    Arm("moe-mtp-n5", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "5"],
        tree="master", expects_drafter=True),
    Arm("moe-mtp-n7", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "7"],
        tree="master", expects_drafter=True,
        note="overlaps the draft-then-verify ladder, so the two paths meet"),

    # Draft-then-verify: the predecessor's path. n8 is its exact losing configuration.
    Arm("moe-draft08b-n4", ["--spec-type", "draft-simple", "--spec-draft-n-max", "4",
                            "-md", str(DRAFT_08B), "-ngld", "99"],
        tree="master", expects_drafter=True),
    Arm("moe-draft08b-n8", ["--spec-type", "draft-simple", "--spec-draft-n-max", "8",
                            "--spec-draft-n-min", "4", "-md", str(DRAFT_08B), "-ngld", "99"],
        tree="master", expects_drafter=True,
        note="REPLICATION ANCHOR: the predecessor's 0.8B draft-then-verify arm, -10.8 % raw "
                    "and -21.5 % class-stratified. Not its -44.6 %, which is a different method"),
    Arm("moe-draft08b-n16", ["--spec-type", "draft-simple", "--spec-draft-n-max", "16",
                             "--spec-draft-n-min", "4", "-md", str(DRAFT_08B), "-ngld", "99"],
        tree="master", expects_drafter=True,
        note="the predecessor also tested 16; larger K, closer to expert-set saturation"),

    # n4, n8 and n16 are verification widths 5, 9 and 17, and only 5 is inside MMVQ, so a fit
    # over this path had exactly one usable point and no `c` at all. n2 and n6 add widths 3 and 7.
    Arm("moe-draft08b-n2", ["--spec-type", "draft-simple", "--spec-draft-n-max", "2",
                            "-md", str(DRAFT_08B), "-ngld", "99"],
        tree="master", expects_drafter=True, note="width 3, on the MMVQ path"),
    Arm("moe-draft08b-n6", ["--spec-type", "draft-simple", "--spec-draft-n-max", "6",
                            "-md", str(DRAFT_08B), "-ngld", "99"],
        tree="master", expects_drafter=True, note="width 7, on the MMVQ path"),

    # The dense side, measured in this run rather than read out of a file from another day. Every
    # arm above runs the MoE target; these three run the dense one, so the architecture is the
    # only thing that differs between a pair. Without them this phase is a replication, not a
    # comparison, whatever its name says.
    Arm("baseline-dense", [], tree="master", model=DENSE,
        note="Qwen3.8-27B dense-hybrid, no speculation; the matched control for baseline-moe"),
    # H6b compares the marginal cost per verified position between the two models, so the dense
    # side needs the same width ladder the MoE side has, not one point. With a single width the
    # slope has nothing to fit and c_dense would have to be borrowed from phase_nmax, measured in
    # another session, which is the comparison these arms exist to avoid.
    Arm("dense-mtp-n1", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"],
        tree="master", expects_drafter=True, model=DENSE,
        note="matched against moe-mtp-n1"),
    Arm("dense-mtp-n2", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
        tree="master", expects_drafter=True, model=DENSE,
        note="matched against moe-mtp-n2"),
    Arm("dense-mtp-n3", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
        tree="master", expects_drafter=True, model=DENSE,
        note="matched against moe-mtp-n3"),
    Arm("dense-mtp-n5", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "5"],
        tree="master", expects_drafter=True, model=DENSE,
        note="matched against moe-mtp-n5"),
    Arm("dense-mtp-n7", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "7"],
        tree="master", expects_drafter=True, model=DENSE,
        note="matched against moe-mtp-n7"),
    Arm("dense-draft08b-n8", ["--spec-type", "draft-simple", "--spec-draft-n-max", "8",
                              "--spec-draft-n-min", "4", "-md", str(DRAFT_08B), "-ngld", "99"],
        tree="master", expects_drafter=True, model=DENSE,
        note="matched against the replication anchor. Phase C measured this configuration at "
             "-29.8 % on a separate day; this arm puts it in the same session as the MoE one"),

    # Width 9 is off the MMVQ path, so the anchor arm above cannot carry a slope. The
    # predecessor's loss lives on this path, which makes it the one where a difference in c
    # between the two models would say the most, so it gets the same 3, 5, 7 the MoE side has.
    Arm("dense-draft08b-n2", ["--spec-type", "draft-simple", "--spec-draft-n-max", "2",
                              "-md", str(DRAFT_08B), "-ngld", "99"],
        tree="master", expects_drafter=True, model=DENSE,
        note="width 3, matched against moe-draft08b-n2"),
    Arm("dense-draft08b-n4", ["--spec-type", "draft-simple", "--spec-draft-n-max", "4",
                              "-md", str(DRAFT_08B), "-ngld", "99"],
        tree="master", expects_drafter=True, model=DENSE,
        note="width 5, matched against moe-draft08b-n4"),
    Arm("dense-draft08b-n6", ["--spec-type", "draft-simple", "--spec-draft-n-max", "6",
                              "-md", str(DRAFT_08B), "-ngld", "99"],
        tree="master", expects_drafter=True, model=DENSE,
        note="width 7, matched against moe-draft08b-n6"),
]

# Each arm is scored against the baseline that ran the same model. Mapping everything to
# baseline-moe would have compared the dense arms against an MoE baseline, which is not a
# speculation effect at all: it is the difference between two models.
BASELINE_MAP = {a.name: ("baseline-dense" if a.model else "baseline-moe")
                for a in ARMS if not a.name.startswith("baseline")}
