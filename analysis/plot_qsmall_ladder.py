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
  3. byte-identical output against the non-speculative baseline. This is what H9 turns on, and
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
    ax.set_title("1.  The drafter must hold still, or panel 2 is not a cost",
                 loc="left", fontsize=10.5, color=P.C("fg"))
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
    for xi, c, m, v in zip(x, cs, ms, rungs):
        ax.annotate(f"{v['label']}\n{m:.2f} ms", (xi, c), textcoords="offset points",
                    xytext=(7, -2), fontsize=8.2, color=P.C("mut"), linespacing=1.15)
    ax.set_ylabel("c  (decode steps of this\nrung's own baseline)")
    ax.set_title("2.  Marginal cost of a verified position. Wall-time value beside each point --"
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
    ax.set_ylabel("byte-identical to the\nnon-speculative baseline (%)")
    ax.set_xlabel("bits per weight  (file size / parameter count, measured)")
    ax.set_title("3.  llama.cpp #25618 says bf16 preserves parity. This is the only ladder that "
                 "can ask", loc="left", fontsize=10.5, color=P.C("fg"))
    ax.set_ylim(-3, 103)
    P._despine(ax)
    ax.grid(axis="y", color=P.C("grid"), lw=0.6, zorder=0)
    for a in axes:
        for xi, v in zip(x, rungs):
            a.axvline(xi, color=P.C("grid"), lw=0.5, ls=":", zorder=0)

    note = ("A binary outcome over 25 prompts resolves very little: Phase Q's pairwise intervals "
            "spanned 32 percentage points, so panel 3 shows an interval covering zero as "
            "UNMEASURED rather than as absent.")
    P._save(fig, "plot_qsmall_ladder", note=note, bottom=0.17, provenance=PROVENANCE)


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
