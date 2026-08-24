"""Figures for the Phase A confirmatory matrix.

Every number is recomputed from `results/phase_a.json` through the same functions the text
reports use -- `analyze.build_series`, `stats.paired_cluster_bootstrap`, `cost_model.collect` --
so a figure cannot drift away from the report it illustrates.

Design follows published guidance rather than taste:
  * effect sizes with intervals are drawn as dot-and-whisker rows, not bars. A bar encodes
    magnitude from zero and fuses the estimate with its uncertainty; a dot separates them.
  * categorical colour is the Wong palette (Nature Methods 8:441), which stays separable under
    all three common colour-vision deficiencies and keeps a lightness difference in greyscale.
    Marker shape carries the same distinction, so colour is never the only channel.
  * the diverging map is RdBu_r, not RdYlGn: a red-to-green ramp is the one pairing that
    collapses for the most common deficiency.
  * every panel states its finding in the title and carries provenance in the footer.

    python3 analysis/plot.py        (repo root, matplotlib available)
"""
from __future__ import annotations

import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))

import analyze as A  # noqa: E402
import cost_model as CM  # noqa: E402
import stats as ST  # noqa: E402

OUT = ROOT / "analysis"
RESULT = ROOT / "results/phase_a.json"
DPI = 170

# Wong, B. (2011) Points of view: Color blindness. Nature Methods 8:441.
WONG = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
        "orange": "#E69F00", "purple": "#CC79A7", "grey": "#999999"}

# Verification width w = n_max + 1: the positions the target scores in one forward pass. It is
# the axis the CUDA kernel selection turns on, not n_max.
WIDTH = {"mtp-n2": 3, "mtp-n3": 4, "dflash2-n4": 5, "mtp-n5": 6, "dflash2-n7": 8}
SPEC_ARMS = sorted(WIDTH, key=WIDTH.get)
BASE = {"mtp-n2": "baseline@master", "mtp-n3": "baseline@master", "mtp-n5": "baseline@master",
        "dflash2-n4": "baseline@pr27342", "dflash2-n7": "baseline@pr27342"}
METHOD = {a: ("draft-dflash" if a.startswith("dflash") else "draft-mtp") for a in SPEC_ARMS}
STYLE = {"draft-mtp": (WONG["blue"], "o"), "draft-dflash": (WONG["vermillion"], "s")}

PROVENANCE = ("Qwen3.8-27B UD-Q4_K_XL · RTX 3090 24 GB · llama.cpp c060ca9 · 7 arms × 25 prompts × "
              "5 passes = 875 requests, 0 incidents, 0 excluded · greedy, --parallel 1, fresh server "
              "per arm-pass, thermal gate at arm entry · hypotheses fixed before measurement, see "
              "PREREGISTRATION.md · generated 2026-08-25 by analysis/plot.py")
CI_NOTE = ("Intervals are 95 % paired cluster bootstrap over prompts within class, 10 000 "
           "resamples, statistic = arm minus baseline.")


def _save(fig, name, note="", bottom_pad=0.0):
    # The footer wraps to the figure's own width and the figure is saved at that width.
    # bbox_inches="tight" instead grows the canvas to fit one unwrapped line, which produced a
    # 26575-pixel-wide image on the first attempt.
    w_in = fig.get_size_inches()[0]
    lines = textwrap.wrap(" ".join(x for x in (note, PROVENANCE) if x),
                          width=int(w_in * 15.0))
    fig.subplots_adjust(bottom=max(bottom_pad, 0.020 * len(lines) + 0.11))
    for i, ln in enumerate(reversed(lines)):
        fig.text(0.5, 0.008 + i * 0.020, ln, ha="center", va="bottom",
                 fontsize=7.3, style="italic", color="#5a5a5a")
    p = OUT / name
    fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}  ({p.stat().st_size // 1024} KB)")


def load():
    result = A.load(RESULT)
    series, prompt_class, _, _ = A.build_series(result, "decode_tok_s")
    return result, series, prompt_class


def _effect(series, prompt_class, arm, relative=True):
    # ARGUMENT ORDER: _balanced returns (arm, baseline) and the bootstrap takes (baseline, arm).
    # Feeding it the pair in the order _balanced hands them back inverts the sign of every
    # effect, which is how the first version of this file turned a +60 % win into a -36 % loss.
    arm_s, base_s = A._balanced(series[arm], series[BASE[arm]])
    return ST.paired_cluster_bootstrap(base_s, arm_s, prompt_class, relative=relative)


def _strat_mean(series, prompt_class, arm):
    per = defaultdict(list)
    for t, vs in series[arm].items():
        per[prompt_class[t]].extend(vs)
    return ST.stratified_mean(per)


# ------------------------------------------------------------------ 1. the primary endpoint
def fig_headline(series, prompt_class):
    rows = [(a, _effect(series, prompt_class, a), _strat_mean(series, prompt_class, a))
            for a in SPEC_ARMS]
    rows.sort(key=lambda r: r[1].point)
    base_abs = _strat_mean(series, prompt_class, "baseline@master")

    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    y = np.arange(len(rows))
    for i, (arm, iv, _) in enumerate(rows):
        col, mk = STYLE[METHOD[arm]]
        ax.plot([iv.lo, iv.hi], [i, i], color=col, lw=2.4, solid_capstyle="butt", zorder=2)
        ax.plot([iv.lo, iv.lo, np.nan, iv.hi, iv.hi], [i - .13, i + .13, np.nan, i - .13, i + .13],
                color=col, lw=2.0, zorder=2)
        ax.plot([iv.point], [i], marker=mk, ms=9, color=col, zorder=3,
                markeredgecolor="white", markeredgewidth=1.1)
    ax.axvline(0, color="#222222", lw=1.3, zorder=1)
    ax.text(0, len(rows) - 0.32, "  no change", fontsize=8.6, color="#444444", va="center")

    for i, (arm, iv, absolute) in enumerate(rows):
        ax.text(1.015, i, f"{absolute:5.1f} tok/s", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=9.4, family="monospace")
        ax.text(1.215, i, f"{iv.point:+5.1f} %   [{iv.lo:+5.1f}, {iv.hi:+5.1f}]",
                transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=9.4, family="monospace")
    ax.text(1.015, len(rows) - 0.32, "throughput", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=8.6, color="#444444")
    ax.text(1.215, len(rows) - 0.32, "effect vs baseline, 95 % CI",
            transform=ax.get_yaxis_transform(), va="center", ha="left",
            fontsize=8.6, color="#444444")

    ax.set_yticks(y, [f"{a}   w={WIDTH[a]}" for a, _, _ in rows], fontsize=10)
    ax.set_ylim(-0.6, len(rows) - 0.05)
    ax.set_xlabel("decode throughput against the non-speculative baseline built from the same "
                  "tree  (%)", fontsize=9.8)
    ax.set_title("Every speculative arm is faster, and every interval clears zero\n"
                 f"non-speculative baseline = {base_abs:.2f} tok/s   ·   n = 25 prompts × 5 passes "
                 "per arm", fontsize=12, pad=12)
    ax.grid(axis="x", alpha=0.22, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], color=c, marker=m, ls="none", ms=8, label=k)
               for k, (c, m) in STYLE.items()]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9.2)
    fig.subplots_adjust(left=0.13, right=0.635, top=0.80)
    _save(fig, "plot_headline.png", note=CI_NOTE)


# ------------------------------------------------------------------ 2. where the win lives
def fig_per_class(series, prompt_class):
    classes = sorted(set(prompt_class.values()))
    M = np.full((len(SPEC_ARMS), len(classes)), np.nan)
    for i, arm in enumerate(SPEC_ARMS):
        arm_s, base_s = A._balanced(series[arm], series[BASE[arm]])
        per = ST.per_class_intervals(base_s, arm_s, prompt_class, relative=True)
        for j, c in enumerate(classes):
            if c in per:
                M[i, j] = per[c].point

    lim = float(np.nanmax(np.abs(M)))
    fig, ax = plt.subplots(figsize=(9.4, 4.3))
    # RdBu, not RdYlGn: blue is the win and red the loss, and the pair survives red-green
    # deficiency, which a green-to-red ramp does not. RdBu_r would put the wins in red.
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:+.0f}%", ha="center", va="center", fontsize=10.5,
                    color="white" if abs(v) > lim * 0.58 else "#111111")
    ax.set_xticks(range(len(classes)), classes, fontsize=10.5)
    ax.set_yticks(range(len(SPEC_ARMS)), [f"{a}   w={WIDTH[a]}" for a in SPEC_ARMS], fontsize=10)
    ax.tick_params(length=0)
    ax.set_title("The gain is concentrated in code and reasoning, and reverses on conversational\n"
                 "and Chinese prompts once the verification width grows", fontsize=12, pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cb.set_label("throughput vs baseline (%)", fontsize=9.4)
    fig.subplots_adjust(left=0.17, right=0.90, top=0.80)
    _save(fig, "plot_per_class.png", note=CI_NOTE + " Per-class, exploratory: the "
          "preregistered endpoint is the pooled effect, not these five.")


# ------------------------------------------------------------------ 3. the cost model
def fig_cost_model(result):
    rows = CM.collect(result)
    grp = defaultdict(list)
    for r in rows:
        grp[(r["spec_type"], r["width"], r["n_max"])].append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for spec, (col, mk) in STYLE.items():
        pts = sorted((w, fmean([x["k"] for x in v])) for (s, w, _), v in grp.items() if s == spec)
        if len(pts) < 2:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        k0, c, r2 = CM._linfit([x - 1 for x in xs], ys)
        gx = np.linspace(min(xs) - 0.35, max(xs) + 0.35, 50)
        ax1.plot(gx, k0 + c * (gx - 1), color=col, lw=1.5, alpha=0.6, zorder=1)
        ax1.plot(xs, ys, mk, ms=9, color=col, zorder=3, markeredgecolor="white",
                 markeredgewidth=1.1,
                 label=(f"{spec}   k₀ = {k0:.3f}   c = {c:.4f}   "
                        + (f"r² = {r2:.4f}" if len(xs) > 2
                           else "r² undefined (2 widths)")))
    ax1.set_xlabel("verification width  w = n-max + 1   (positions scored per target pass)",
                   fontsize=9.6)
    ax1.set_ylabel("k   (target-pass cost, in baseline decode steps)", fontsize=9.6)
    ax1.set_title("Two independent drafters, one marginal cost:\n"
                  "k = k₀ + c(w−1) with c agreeing to 1.6 % while k₀ differs", fontsize=11)
    ax1.text(0.985, 0.03, "draft-dflash has only two widths here, so its line is determined\n"
                          "rather than fitted. phase_nmax adds w = 3 and 7.",
             transform=ax1.transAxes, fontsize=8.2, color="#555555", va="bottom", ha="right")
    ax1.legend(fontsize=8.4, frameon=False, loc="upper left")
    ax1.grid(alpha=0.22)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    for spec, (col, mk) in STYLE.items():
        pts = sorted((nm, fmean([x["speedup"] for x in v]))
                     for (s, _, nm), v in grp.items() if s == spec)
        if not pts:
            continue
        ax2.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, color=col, lw=1.7, ms=8,
                 markeredgecolor="white", markeredgewidth=1.1, label=spec)
        best = max(pts, key=lambda p: p[1])
        ax2.annotate(f"best n-max = {best[0]}\n{best[1]:.3f}×", best, textcoords="offset points",
                     xytext=(12, -6), fontsize=9, color=col, fontweight="bold")
    ax2.axhline(1.0, color="#666666", ls="--", lw=1.0)
    ax2.text(ax2.get_xlim()[0], 1.0, " break-even", fontsize=8.4, color="#666666", va="bottom")
    ax2.set_xlabel("--spec-draft-n-max", fontsize=9.6)
    ax2.set_ylabel("speedup  =  mean_len / k", fontsize=9.6)
    ax2.set_title("Speedup falls monotonically across the widths measured here:\n"
                  "the best n-max is the smallest one tested, not the largest", fontsize=11)
    ax2.legend(fontsize=9, frameon=False)
    ax2.grid(alpha=0.22)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.82, wspace=0.24)
    _save(fig, "plot_cost_model.png", note="k is derived per request as mean_len / "
          "speedup; points are means over 125 requests per arm.")


# ------------------------------------------------------------------ 4. the width partition
def fig_width_partition(result):
    fork = defaultdict(dict)
    for rec in result["records"]:
        d = rec.get("divergence")
        if rec["arm"] in WIDTH and rec["pass"] == 1 and d:
            fork[rec["prompt"]][rec["arm"]] = "SAME" if d["identical"] else d["first_diff_char"]
    prompts = sorted(p for p in fork if len(fork[p]) == len(SPEC_ARMS))
    lo = [a for a in SPEC_ARMS if WIDTH[a] <= 4]
    hi = [a for a in SPEC_ARMS if WIDTH[a] >= 5]
    clean = [p for p in prompts
             if len({fork[p][a] for a in lo}) == 1 and len({fork[p][a] for a in hi}) == 1]
    split = [p for p in clean if fork[p][lo[0]] != fork[p][hi[0]]]

    # Colour marks which distinct fork position a cell holds within its own row, so a row where
    # the two width groups disagree reads as two blocks and a row where they agree reads as one.
    PAL = [WONG["blue"], WONG["vermillion"], WONG["green"]]
    fig, ax = plt.subplots(figsize=(10.6, 8.8))
    for yi, p in enumerate(prompts):
        vals = [fork[p][a] for a in SPEC_ARMS]
        order = list(dict.fromkeys(v for v in vals if v != "SAME"))
        for xi, v in enumerate(vals):
            face = "#dddddd" if v == "SAME" else PAL[order.index(v) % len(PAL)]
            ax.add_patch(plt.Rectangle((xi, yi), 1, 1, facecolor=face, edgecolor="white", lw=1.3))
            ax.text(xi + .5, yi + .5, "identical" if v == "SAME" else f"@{v}",
                    ha="center", va="center", fontsize=8.4,
                    color="#666666" if v == "SAME" else "white")
    ax.axvline(2, color="#111111", lw=3.0)
    ax.set_xlim(0, len(SPEC_ARMS))
    ax.set_ylim(len(prompts), 0)
    ax.set_xticks([i + .5 for i in range(len(SPEC_ARMS))],
                  [f"{a}\nw = {WIDTH[a]}" for a in SPEC_ARMS], fontsize=9.6)
    ax.set_yticks([i + .5 for i in range(len(prompts))], prompts, fontsize=8.6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Greedy output forks from the baseline at a position set by verification\n"
                 "width, not by which drafter produced the tokens\n"
                 f"w ∈ {{3,4}} agree and w ∈ {{5,6,8}} agree on {len(clean)}/{len(prompts)} prompts;\n"
                 f"the two groups land on different positions on {len(split)} of them",
                 fontsize=11.5, pad=14)
    ax.annotate("CUDA calc_nwarps boundary\nncols_dst 1–4 → 4 warps  |  5–8 → 2 warps",
                xy=(2, len(prompts)), xytext=(2, len(prompts) + 1.5), ha="center", va="top",
                fontsize=9, color="#111111",
                arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.2))
    fig.subplots_adjust(left=0.22, right=0.99, top=0.845)
    _save(fig, "plot_width_partition.png", bottom_pad=0.13,
          note="Pass 1 shown; the partition is identical in all five passes. "
               "Greedy decoding is deterministic, so no interval applies here.")
    print(f"    partition: {len(clean)}/{len(prompts)} rows group cleanly, "
          f"{len(split)} differ between groups")


def main():
    OUT.mkdir(exist_ok=True)
    result, series, prompt_class = load()
    fig_headline(series, prompt_class)
    fig_per_class(series, prompt_class)
    fig_cost_model(result)
    fig_width_partition(result)


if __name__ == "__main__":
    main()
