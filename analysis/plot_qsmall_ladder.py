"""The Phase Q-small figure: a quantization ladder that reaches bf16.

llama.cpp #25618 scopes its finding as "quantized targets diverge, bf16 does not". Qwen3.8-27B
cannot test the second half -- its bf16 is 50 GB and fits on neither card here -- so the anchor
the whole claim rests on was unobtainable until this ladder swapped the model rather than the
hardware. Qwen3.5-9B-MTP ships the full ladder and its bf16 is 17.14 GiB.

Three panels, stacked, in the order the argument needs them and not in order of interest:

  1. acceptance, which must be FLAT. The MTP head is inside the target gguf, so quantizing the
     target quantizes the drafter. If acceptance moves with bits per weight, panel 2 is a
     mixture of verification cost and drafting behaviour and cannot be read as a cost.
  2. `c`, the marginal cost of a verified position, in the arm's own decode steps. The
     millisecond figure is printed beside each point rather than plotted, because `c` is
     denominated in each rung's own step and on Phase Q the two denominations disagreed in SIGN.
  3. share with NO DIVERGENCE OBSERVED against the non-speculative baseline through the token
     cap. Not "byte-identical": every request stops at the cap and none reaches EOS, so a
     match inside the window is right-censored rather than identity to the end of an answer.
     This is what H9 turns on, and
     the bf16 rung is the only place #25618's own control exists.

The x axis is measured, not labelled: bits per weight, file size over parameter count, with the
parameter count derived from the bf16 rung as size/2. Labels sit beside their points so the axis
stays a quantity.

Panels stack rather than sitting side by side, following plot.py's rule: GitHub scales anything
wider than its column, and a side-by-side row arrives unreadable.

    .venv/bin/python analysis/plot_qsmall_ladder.py results/phase_qsmall_*.json
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "harness"))

import matplotlib.pyplot as plt  # noqa: E402
import cost_model as CM  # noqa: E402
import ladder_trend as LT  # noqa: E402
import plot as P  # noqa: E402

PROVENANCE = ("Phase Q-small | Qwen3.5-9B-MTP, four rungs Q4_K_M to BF16 | RTX 3090 24 GB | "
              "greedy, --parallel 1, 8192 ctx q8_0 KV on every rung, fresh server per arm-pass, "
              "arms interleaved within a pass | hypotheses in PREREGISTRATION.md Correction 21")


def _panel_points(ax, xs, ys, label, color, marker):
    ax.plot(xs, ys, marker=marker, color=color, lw=1.6, ms=6, label=label, zorder=3)


def build(rungs, shared, bpw):
    families = sorted(set.intersection(
        *[{r["arm"].split("@")[0] for r in v["rows"]} for v in rungs]))
    prompt_class = {r["prompt"]: r["class"] for r in rungs[0]["rows"]}
    x = [bpw[v["label"]] for v in rungs]

    fig, axes = plt.subplots(3, 1, figsize=(P.FIG_W, 10.5), sharex=True)
    colors = [P.WONG[k] for k in ("blue", "vermillion", "green", "orange", "purple")]
    marks = "osD^v"

    # ---- 1. acceptance, the precondition
    ax = axes[0]
    for i, fam in enumerate(families):
        per = [LT._by_prompt(v, fam, "acceptance") for v in rungs]
        ys = [sum(sum(d[p]) for p in d if d[p]) / max(1, sum(len(d[p]) for p in d))
              for d in per]
        _panel_points(ax, x, ys, fam, colors[i % len(colors)], marks[i % len(marks)])
    ax.set_ylabel("acceptance\n(accepted / drafted)")
    # States what the panel shows. "The drafter must hold still, or panel 2 is not a cost" was
    # the design argument, and it claims more than acceptance can deliver: the MTP head is
    # embedded in the target gguf and is quantized with it, so stable acceptance shows the
    # drafter proposing the same way, not its forward pass costing the same.
    ax.set_title("1.  Acceptance holds still, so panel 2 is not the drafter proposing differently",
                 loc="left", fontsize=10.5, color=P.C("fg"))
    # Headroom first, legend second. A one-row legend at "upper left" sat exactly on the
    # highest series -- mtp-n2 runs flat across the top of this panel -- so the line passed
    # through the legend text. Reserving the space is what keeps them apart at any data range.
    _lo, _hi = ax.get_ylim()
    ax.set_ylim(_lo, _hi + (_hi - _lo) * 0.22)
    ax.legend(frameon=False, fontsize=8.5, ncol=len(families), loc="upper left")
    P._despine(ax)
    ax.grid(axis="y", color=P.C("grid"), lw=0.6, zorder=0)

    # ---- 2. c, with the millisecond value annotated
    ax = axes[1]
    cs, ms = [], []
    for v in rungs:
        ci = CM.fit_ci(v["rows"], shared)
        c = ci["c"].point if ci else float("nan")
        step = 1000.0 / v["baseline_tok_s"]
        cs.append(c)
        ms.append(c * step)
        if ci:
            ax.errorbar([bpw[v["label"]]], [c],
                        yerr=[[c - ci["c"].lo], [ci["c"].hi - c]],
                        fmt="none", ecolor=P.C("mut"), elinewidth=1.1, capsize=3, zorder=2)
    _panel_points(ax, x, cs, "c over shared widths " + str(shared), P.WONG["blue"], "o")
    # Labels sit ABOVE their point and the last one sits to its left. This series descends
    # left to right, so a label placed below and to the right lands on the segment leaving the
    # point, and the rightmost one ran into the frame. Above-and-right clears the descending
    # segment; flipping the last one keeps it inside the axes without widening them.
    _last = len(x) - 1
    for i, (xi, c, m, v) in enumerate(zip(x, cs, ms, rungs)):
        right = i != _last
        ax.annotate(f"{v['label']}\n{m:.2f} ms", (xi, c), textcoords="offset points",
                    xytext=(7 if right else -7, 8), ha="left" if right else "right",
                    va="bottom", fontsize=8.2, color=P.C("mut"), linespacing=1.15)
    _lo2, _hi2 = ax.get_ylim()
    ax.set_ylim(_lo2, _hi2 + (_hi2 - _lo2) * 0.18)
    ax.set_ylabel("c  (decode steps of this\nrung's own baseline)")
    ax.set_title("2.  Cost chord per verified position. Wall-time value beside each point --"
                 " the two can disagree in sign",
                 loc="left", fontsize=10.5, color=P.C("fg"))
    P._despine(ax)
    ax.grid(axis="y", color=P.C("grid"), lw=0.6, zorder=0)

    # ---- 3. byte-identical, H9
    ax = axes[2]
    for i, fam in enumerate(families):
        per = [LT._identical_by_prompt(v, fam) for v in rungs]
        if not all(per):
            continue
        ys = [100.0 * sum(sum(d[p]) for p in d) / max(1, sum(len(d[p]) for p in d)) for d in per]
        _panel_points(ax, x, ys, fam, colors[i % len(colors)], marks[i % len(marks)])
    ax.set_ylabel("no divergence observed\nthrough the token cap (%)")
    ax.set_xlabel("bits per weight  (file size / parameter count, measured)")
    # "the only ladder that can ask" is a claim about the world; the reason is the one to state.
    ax.set_title("3.  llama.cpp #25618 says bf16 preserves parity. The 27B cannot hold bf16, "
                 "so this 9B ladder is where the question can be asked",
                 loc="left", fontsize=10.5, color=P.C("fg"))
    ax.set_ylim(-3, 103)
    P._despine(ax)
    ax.grid(axis="y", color=P.C("grid"), lw=0.6, zorder=0)
    for a in axes:
        for xi, v in zip(x, rungs):
            a.axvline(xi, color=P.C("grid"), lw=0.5, ls=":", zorder=0)

    note = ("A binary outcome over 25 prompts resolves very little: Phase Q's pairwise intervals "
            "spanned 32 percentage points, so panel 3 shows an interval covering zero as "
            "UNMEASURED rather than as absent.")
    P._save(fig, "plot_qsmall_ladder", note=note, bottom=0.17, provenance=PROVENANCE,
             top_in=0.28)


def main() -> int:
    paths = sys.argv[1:]
    if len(paths) < 3:
        print(__doc__)
        print("error: give at least three rung result files", file=sys.stderr)
        return 2
    rungs = [LT.load_rung(p) for p in paths]
    rungs.sort(key=lambda v: v["size_bytes"] or 0)
    bad = LT.guards(rungs)
    if bad:
        print("REFUSED, the figure would misrepresent these rungs:")
        for x in bad:
            print(f"  - {x}")
        return 1
    shared = sorted(set.intersection(*[set(v["on_path"]) for v in rungs]))
    bpw, how = LT.bits_per_weight(rungs)
    if bpw is None:
        print(f"No bf16 rung: {how}. The x axis this figure needs does not exist.")
        return 1
    print(f"x axis: {how}")
    for mode in ("light", "dark"):
        with P.theme(mode):
            build(rungs, shared, bpw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
