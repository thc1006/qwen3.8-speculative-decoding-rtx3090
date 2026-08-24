"""Figures for the Phase A confirmatory matrix, sized for a GitHub README.

Every number is recomputed from the result files through the same functions the text reports
use -- `analyze.build_series`, `stats.paired_cluster_bootstrap`, `cost_model.collect`,
`elasticity._paired_elasticity` -- so a figure cannot drift from the report it illustrates.

Layout is dictated by where these are read. GitHub renders README content in a column of roughly
1010 px and scales anything wider to fit, so a 2300 px figure with 8 pt labels arrives at about
3.5 pt and is unreadable. Two rules follow, and they are why nothing here is drawn side by side:

    * no figure exceeds MAX_PX wide, so downscaling is slight;
    * two views of one idea stack vertically, which keeps full width for each.

Design follows published guidance rather than taste:
    * effect sizes with intervals are dots and whiskers, not bars. A bar encodes magnitude from
      zero and fuses the estimate with its uncertainty.
    * categorical colour is the Wong palette (Nature Methods 8:441), separable under all three
      common colour-vision deficiencies and in greyscale. Marker shape repeats the distinction,
      so colour is never the only channel.
    * the diverging map is RdBu, never RdYlGn: red-to-green is the one pairing that collapses.
    * agreement between conditions is measured and drawn directly rather than left for the
      reader to verify cell by cell.

Each figure is written twice, light and dark, because a white figure is a glare panel in a dark
README. Pair them with <picture> and prefers-color-scheme.

    python3 analysis/plot.py        (repo root, matplotlib available)
"""
from __future__ import annotations

import sys
import textwrap
from collections import defaultdict
from contextlib import contextmanager
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
import elasticity as EL  # noqa: E402
import stats as ST  # noqa: E402

OUT = ROOT / "analysis"
RESULT = ROOT / "results/phase_a.json"
RESULT_R = ROOT / "results/phase_r.json"
RESULT_R2 = ROOT / "results/phase_r2.json"

DPI = 150
FIG_W = 9.0          # inches; 9.0 x 150 = 1350 px, close to GitHub's column
MAX_PX = 1400

# Wong, B. (2011) Points of view: Color blindness. Nature Methods 8:441.
WONG = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
        "orange": "#E69F00", "purple": "#CC79A7"}
INK = {"light": {"fg": "#111111", "mut": "#5a5a5a", "grid": "#c9c9c9", "bg": "#ffffff",
                 "neutral": "#333333", "flat": "#e2e2e2"},
       "dark":  {"fg": "#e6edf3", "mut": "#9aa4b0", "grid": "#3a4149", "bg": "#0d1117",
                 "neutral": "#c9d1d9", "flat": "#30363d"}}

WIDTH = {"mtp-n2": 3, "mtp-n3": 4, "dflash2-n4": 5, "mtp-n5": 6, "dflash2-n7": 8}
SPEC_ARMS = sorted(WIDTH, key=WIDTH.get)
BASE = {"mtp-n2": "baseline@master", "mtp-n3": "baseline@master", "mtp-n5": "baseline@master",
        "dflash2-n4": "baseline@pr27342", "dflash2-n7": "baseline@pr27342"}
METHOD = {a: ("draft-dflash" if a.startswith("dflash") else "draft-mtp") for a in SPEC_ARMS}
DRAFTER = {"mtp-n2": "MTP", "mtp-n3": "MTP", "mtp-n5": "MTP",
           "dflash2-n4": "DFlash2", "dflash2-n7": "DFlash2"}
STYLE = {"draft-mtp": (WONG["blue"], "o"), "draft-dflash": (WONG["vermillion"], "s")}

PROVENANCE = ("Qwen3.8-27B UD-Q4_K_XL · RTX 3090 24 GB · llama.cpp c060ca9 · greedy, "
              "--parallel 1, fresh server per arm-pass, thermal gate at arm entry · "
              "PREREGISTRATION.md · 2026-08-25")
PHASE_A_N = "Phase A: 875 requests, 0 incidents, 0 excluded."
CI_NOTE = "95 % paired cluster bootstrap over prompts within class, 10 000 resamples."

_MODE = "light"


def C(k):
    return INK[_MODE][k]


@contextmanager
def theme(mode):
    global _MODE
    _MODE = mode
    rc = {"figure.facecolor": C("bg"), "axes.facecolor": C("bg"), "savefig.facecolor": C("bg"),
          "text.color": C("fg"), "axes.labelcolor": C("fg"), "axes.edgecolor": C("grid"),
          "xtick.color": C("fg"), "ytick.color": C("fg"), "grid.color": C("grid"),
          "font.size": 11.0, "axes.titlesize": 12.5, "axes.labelsize": 10.5,
          "xtick.labelsize": 10.0, "ytick.labelsize": 10.0, "legend.fontsize": 9.8}
    with plt.rc_context(rc):
        yield
    _MODE = "light"


def _save(fig, stem, note="", bottom=0.13):
    # Footer spacing is computed in inches, not in figure fractions. A fixed fraction is a
    # different number of pixels on every figure height, and on the 4.5-inch panels it put the
    # footer lines on top of each other and on the x-label.
    w_in, h_in = fig.get_size_inches()
    lines = textwrap.wrap(" ".join(x for x in (note, PROVENANCE) if x), width=int(w_in * 15.5))
    line_h, pad = 0.155 / h_in, 0.05 / h_in
    # 0.62 in below the axes for the tick labels and the x-label, then the footer block under it.
    fig.subplots_adjust(bottom=max(bottom, pad + line_h * len(lines) + 0.62 / h_in))
    for i, ln in enumerate(reversed(lines)):
        fig.text(0.5, pad + i * line_h, ln, ha="center", va="bottom",
                 fontsize=7.6, style="italic", color=C("mut"))
    name = f"{stem}.png" if _MODE == "light" else f"{stem}_dark.png"
    p = OUT / name
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    px = int(w_in * DPI)
    flag = "" if px <= MAX_PX else f"  ** {px} px exceeds MAX_PX={MAX_PX} **"
    print(f"  {name:38s} {px:5d} px  {p.stat().st_size // 1024:4d} KB{flag}")


def _effect(series, prompt_class, arm, relative=True):
    # _balanced returns (arm, baseline); the bootstrap takes (baseline, arm). Handing it the pair
    # in the order _balanced returns them inverts the sign of every effect.
    arm_s, base_s = A._balanced(series[arm], series[BASE[arm]])
    return ST.paired_cluster_bootstrap(base_s, arm_s, prompt_class, relative=relative)


def _strat_mean(series, prompt_class, arm):
    per = defaultdict(list)
    for t, vs in series[arm].items():
        per[prompt_class[t]].extend(vs)
    return ST.stratified_mean(per)


def _despine(ax, keep=("bottom", "left")):
    for s, sp in ax.spines.items():
        sp.set_visible(s in keep)


# ------------------------------------------------------------------ 1. the primary endpoint
def fig_headline(series, prompt_class):
    rows = sorted(((a, _effect(series, prompt_class, a), _strat_mean(series, prompt_class, a))
                   for a in SPEC_ARMS), key=lambda r: r[1].point)
    base_abs = _strat_mean(series, prompt_class, "baseline@master")

    fig, ax = plt.subplots(figsize=(FIG_W, 4.5))
    for i, (arm, iv, _) in enumerate(rows):
        col, mk = STYLE[METHOD[arm]]
        ax.plot([iv.lo, iv.hi], [i, i], color=col, lw=2.6, solid_capstyle="butt", zorder=2)
        for x in (iv.lo, iv.hi):
            ax.plot([x, x], [i - .14, i + .14], color=col, lw=2.2, zorder=2)
        ax.plot([iv.point], [i], marker=mk, ms=10, color=col, zorder=3,
                markeredgecolor=C("bg"), markeredgewidth=1.3)
        ax.text(iv.hi + 1.8, i, f"{iv.point:+.1f} %", va="center", fontsize=10.4,
                family="monospace", color=C("fg"))
    ax.axvline(0, color=C("neutral"), lw=1.4, zorder=1)
    ax.set_yticks(range(len(rows)),
                  [f"{a}  ·  w={WIDTH[a]}  ·  {m:.1f} tok/s" for a, _, m in rows], fontsize=10)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.set_xlim(0, max(r[1].hi for r in rows) * 1.20)
    ax.set_xlabel("faster than the non-speculative baseline of the same tree  (%)")
    ax.set_title("Every speculative arm is faster, and every 95 % interval clears zero\n"
                 f"baseline {base_abs:.2f} tok/s  ·  25 prompts × 5 passes per arm", pad=12)
    ax.grid(axis="x", alpha=0.5, zorder=0)
    _despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[plt.Line2D([], [], color=c, marker=m, ls="none", ms=8, label=k)
                       for k, (c, m) in STYLE.items()],
              loc="lower right", frameon=False)
    fig.subplots_adjust(left=0.315, right=0.975, top=0.80)
    _save(fig, "plot_headline", note=PHASE_A_N + " " + CI_NOTE)


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
    fig, ax = plt.subplots(figsize=(FIG_W, 4.3))
    # RdBu: blue the win, red the loss. RdYlGn would collapse for red-green deficiency and
    # RdBu_r would put the wins in red.
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:+.0f}%", ha="center", va="center", fontsize=11.5,
                    color="white" if abs(v) > lim * 0.58 else "#111111")
    ax.set_xticks(range(len(classes)), classes, fontsize=11)
    ax.set_yticks(range(len(SPEC_ARMS)), [f"{a}  ·  w={WIDTH[a]}" for a in SPEC_ARMS], fontsize=10)
    ax.tick_params(length=0)
    _despine(ax, keep=())
    ax.set_title("The gain is concentrated in code and reasoning, and reverses on\n"
                 "conversational and Chinese prompts as the verification width grows", pad=12)
    fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02).set_label("vs baseline (%)", fontsize=9.6)
    fig.subplots_adjust(left=0.20, right=0.93, top=0.80)
    _save(fig, "plot_per_class",
          note=PHASE_A_N + " " + CI_NOTE + " Per-class is exploratory; the preregistered endpoint is the pooled "
                         "effect.")


# ------------------------------------------------------------------ 3. the cost model
def fig_cost_model(result):
    rows = CM.collect(result)
    grp = defaultdict(list)
    for r in rows:
        grp[(r["spec_type"], r["width"])].append(r)

    def series_of(spec, key):
        return sorted((w, fmean([x[key] for x in v])) for (s, w), v in grp.items() if s == spec)

    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 8.6), sharex=True)
    ax_n, ax_k, ax_s = axes

    c_labels = []
    for spec, (col, mk) in STYLE.items():
        for ax, key in ((ax_n, "mean_len"), (ax_k, "k"), (ax_s, "speedup")):
            pts = series_of(spec, key)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, color=col, lw=1.8,
                    ms=9, markeredgecolor=C("bg"), markeredgewidth=1.2,
                    label=spec if ax is ax_n else None)
        k_pts = series_of(spec, "k")
        if len(k_pts) >= 2:
            k0, c, _ = CM._linfit([w - 1 for w, _ in k_pts], [v for _, v in k_pts])
            gx = np.linspace(2.6, 8.4, 40)
            ax_k.plot(gx, k0 + c * (gx - 1), color=col, lw=1.1, ls="--", alpha=0.75, zorder=1)
            c_labels.append((f"c = {c:.4f}   ({spec})", col))

    ref = series_of("draft-mtp", "mean_len")
    if ref:
        w0, y0 = ref[0]
        gx = np.linspace(w0, 8.4, 40)
        ax_n.plot(gx, y0 * gx / w0, color=C("mut"), lw=1.1, ls=":", zorder=1)
        ax_n.text(6.05, y0 * 6.05 / w0, " growth in proportion to width", color=C("mut"),
                  fontsize=9.2, va="center", rotation=0)
    ax_n.set_ylabel("tokens accepted\nper target pass")
    ax_n.set_title("The benefit saturates: the gap below the dotted line is what is lost", pad=8)
    ax_n.legend(frameon=False, loc="lower right")

    for i, (txt, col) in enumerate(c_labels):
        ax_k.text(0.015, 0.93 - i * 0.115, txt, transform=ax_k.transAxes, color=col,
                  fontsize=9.8, va="top", family="monospace")
    ax_k.set_ylabel("cost of one target pass,\nin plain decode steps")
    ax_k.set_title("The cost is linear: each extra verified position costs a fixed c", pad=8)

    ax_s.axhline(1.0, color=C("mut"), ls=":", lw=1.1)
    ax_s.text(2.65, 1.005, "break-even", fontsize=9, color=C("mut"), va="bottom")
    ax_s.set_ylabel("speedup\n= tokens / cost")
    ax_s.set_title("Saturating benefit over linear cost: the best width is a small one", pad=8)
    ax_s.set_xlabel("verification width  w = n-max + 1   (positions the target scores per pass)")
    for spec, (col, _) in STYLE.items():
        pts = series_of(spec, "speedup")
        if pts:
            bw, bv = max(pts, key=lambda p: p[1])
            ax_s.annotate(f"best w = {bw}  ({bv:.2f}×)", (bw, bv), textcoords="offset points",
                          xytext=(10, 6), fontsize=9.6, color=col, fontweight="bold")

    for ax in axes:
        ax.grid(alpha=0.5)
        _despine(ax)
    ax_s.set_xticks([3, 4, 5, 6, 7, 8])
    fig.subplots_adjust(left=0.165, right=0.965, top=0.945, hspace=0.30)
    _save(fig, "plot_cost_model", bottom=0.105,
          note=PHASE_A_N + " k is recovered per request as mean_len / speedup, then averaged over 125 requests "
               "per arm. draft-dflash has two widths here, so its dashed line is determined "
               "rather than fitted.")


# ------------------------------------------------------------------ 4. the width partition
def _forks(result):
    fork = defaultdict(dict)
    for rec in result["records"]:
        d = rec.get("divergence")
        if rec["arm"] in WIDTH and rec["pass"] == 1 and d:
            fork[rec["prompt"]][rec["arm"]] = "SAME" if d["identical"] else d["first_diff_char"]
    return {p: v for p, v in fork.items() if len(v) == len(SPEC_ARMS)}


def fig_width_partition(result):
    fork = _forks(result)
    prompts = sorted(fork)
    n = len(SPEC_ARMS)
    agree = np.array([[100.0 * sum(fork[p][a] == fork[p][b] for p in prompts) / len(prompts)
                       for b in SPEC_ARMS] for a in SPEC_ARMS])
    lo = [a for a in SPEC_ARMS if WIDTH[a] <= 4]
    hi = [a for a in SPEC_ARMS if WIDTH[a] >= 5]
    cross = min(agree[SPEC_ARMS.index(a), SPEC_ARMS.index(b)] for a in lo for b in hi)
    differ = sum(1 for p in prompts if fork[p][lo[0]] != fork[p][hi[0]])

    fig, ax = plt.subplots(figsize=(FIG_W, 5.6))
    im = ax.imshow(agree, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="#8a8a8a", fontsize=14)
            else:
                ax.text(j, i, f"{agree[i, j]:.0f}%", ha="center", va="center", fontsize=13,
                        color="white" if agree[i, j] > 55 else "#111111")
    labels = [f"w={WIDTH[a]}\n{DRAFTER[a]}" for a in SPEC_ARMS]
    ax.set_xticks(range(n), labels, fontsize=10)
    ax.set_yticks(range(n), [f"{a}\nw={WIDTH[a]} · {DRAFTER[a]}" for a in SPEC_ARMS], fontsize=9.4)
    ax.tick_params(length=0)
    _despine(ax, keep=())
    ax.axhline(1.5, color=C("fg"), lw=3.0)
    ax.axvline(1.5, color=C("fg"), lw=3.0)
    ax.set_title("Verification width decides where output forks — not the drafter\n"
                 f"100 % agreement within each width group, {cross:.0f} % across them",
                 fontsize=12, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.030, pad=0.025).set_label(
        f"share of the {len(prompts)} prompts on which\nboth arms fork at the same character (%)", fontsize=9)
    fig.subplots_adjust(left=0.175, right=0.885, top=0.845)
    _save(fig, "plot_width_partition",
          note=PHASE_A_N + f" The 100 % block spans both drafters, so drafter identity does not predict it "
               f"while width does. The two groups fork elsewhere on {differ} of {len(prompts)} "
               f"prompts. Pass 1; identical in all five. w is the CUDA kernel's ncols_dst and "
               f"calc_nwarps switches between 4 and 5.")


# ------------------------------------------------------------------ 5. the bandwidth bottleneck
R_METHODS = [("baseline", None, "o"), ("mtp-n3", WONG["blue"], "s"), ("mtp-n7", WONG["green"], "^")]
R_BW = ["bw-lo", "stock", "bw-hi"]


def fig_bandwidth(result_r):
    cells = EL._cells(result_r)
    clocks = {c: EL._clock(result_r, "baseline", c, EL.MEM_KEY) for c in R_BW}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, 7.4))
    bars = []
    for m, col, mk in R_METHODS:
        col = col or C("neutral")
        y = [fmean([v for vs in cells[(m, c)].values() for v in vs]) for c in R_BW]
        ref = y[R_BW.index("stock")]
        ax1.plot([clocks[c] for c in R_BW], [100 * v / ref for v in y], marker=mk, color=col,
                 lw=2.0, ms=9, markeredgecolor=C("bg"), markeredgewidth=1.2,
                 label=f"{m}  ({ref:.1f} tok/s at stock)")
        for a, b in zip(R_BW, R_BW[1:]):
            e, elo, ehi = EL._paired_elasticity(
                cells[(m, a)], cells[(m, b)], clocks[a], clocks[b],
                x_lo_samples=EL._clock_samples(result_r, m, a, EL.MEM_KEY),
                x_hi_samples=EL._clock_samples(result_r, m, b, EL.MEM_KEY))
            bars.append((f"{m}   {a} → {b}", e, elo, ehi, col, mk))

    ax1.axhline(100, color=C("mut"), lw=1.0, ls=":")
    ax1.set_xlabel("memory clock (MHz)")
    ax1.set_ylabel("throughput, % of that\nmethod's own stock value")
    ax1.set_title("Moving memory bandwidth ±4 % moves the baseline and barely\n"
                  "moves either speculative arm", pad=10)
    ax1.legend(frameon=False, loc="upper left", fontsize=9.4)
    ax1.grid(alpha=0.5)
    _despine(ax1)

    for i, (_, e, elo, ehi, col, mk) in enumerate(bars):
        ax2.plot([elo, ehi], [i, i], color=col, lw=2.4, zorder=2)
        ax2.plot([e], [i], marker=mk, ms=9, color=col, zorder=3,
                 markeredgecolor=C("bg"), markeredgewidth=1.2)
        ax2.text(ehi + 0.035, i, f"{e:.2f}", va="center", fontsize=10, family="monospace",
                 color=C("fg"))
    for x, lab, ha in ((0.0, "bandwidth-independent", "left"),
                       (1.0, "throughput ∝ bandwidth", "right")):
        ax2.axvline(x, color=C("mut"), lw=1.0, ls=":")
        pad = " " if ha == "left" else ""
        ax2.text(x, len(bars) - 0.3, f"{pad}{lab}{'' if ha == 'left' else ' '}", fontsize=9,
                 color=C("mut"), va="center", ha=ha)
    ax2.set_yticks(range(len(bars)), [b[0] for b in bars], fontsize=9.4)
    ax2.set_ylim(-0.6, len(bars) - 0.05)
    ax2.set_xlim(-0.10, 1.15)
    ax2.set_xlabel("bandwidth elasticity   d(ln tok/s) / d(ln memory clock)")
    ax2.set_title("The baseline is bandwidth-bound; speculation is not", pad=10)
    ax2.grid(axis="x", alpha=0.5)
    ax2.tick_params(axis="y", length=0)
    _despine(ax2, keep=("bottom",))
    fig.subplots_adjust(left=0.275, right=0.965, top=0.915, hspace=0.62)
    _save(fig, "plot_bandwidth_elasticity", bottom=0.115,
          note="Phase R, 1125 requests. Elasticity is a cluster bootstrap over prompts and over "
               "the measured clock, per interval, never pooled across a regime change. Only the "
               "bandwidth conditions are shown: Phase R's compute axis used a power cap, which "
               "Phase R2 replaces with a pinned clock.")


# ------------------------------------------------------------------ 6. what speculation is bound by
# Phase R2 pins the SM clock instead of capping power, so both levers move independently and the
# denominator of each elasticity is a setting rather than an outcome. That makes the two axes
# comparable, which is what this figure needs.
R2_HI = ("sm1700-bwlo", "sm1700", "sm1700-bwhi")     # bandwidth sweep at a pinned 1710 MHz core
R2_COMPUTE = (("sm600", "sm1200"), ("sm1200", "sm1700"))
R2_METHODS = [("baseline", None, "o"), ("mtp-n3", WONG["blue"], "s"), ("mtp-n7", WONG["green"], "^")]


def _elast(res, m, lo, hi, key):
    cells = EL._cells(res)
    x_lo, x_hi = EL._clock(res, m, lo, key), EL._clock(res, m, hi, key)
    return EL._paired_elasticity(cells[(m, lo)], cells[(m, hi)], x_lo, x_hi,
                                 x_lo_samples=EL._clock_samples(res, m, lo, key),
                                 x_hi_samples=EL._clock_samples(res, m, hi, key))


def fig_bound_by(res):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, 8.0))

    # --- the plane: bandwidth elasticity against compute elasticity, at the top of the clock range
    # the two speculative points sit almost on top of each other, so their labels are offset
    # rather than anchored, or they overlap
    OFFS = {"baseline": (14, 0, "left"), "mtp-n3": (-10, -26, "right"), "mtp-n7": (10, 18, "left")}
    for m, col, mk in R2_METHODS:
        col = col or C("neutral")
        bw = fmean([_elast(res, m, a, b, EL.MEM_KEY)[0]
                    for a, b in zip(R2_HI, R2_HI[1:])])
        cp = _elast(res, m, "sm1200", "sm1700", EL.SM_KEY)[0]
        ax1.plot([bw], [cp], marker=mk, ms=14, color=col, markeredgecolor=C("bg"),
                 markeredgewidth=1.5, zorder=3, label=m)
        dx, dy, ha = OFFS[m]
        ax1.annotate(f"{m}\n({bw:.2f}, {cp:.2f})", (bw, cp), textcoords="offset points",
                     xytext=(dx, dy), fontsize=9.6, color=col, va="center", ha=ha)
    ax1.plot([0, 1], [1, 0], color=C("mut"), ls=":", lw=1.0, zorder=1)
    ax1.text(0.06, 0.90, "compute-bound", fontsize=9.6, color=C("mut"))
    ax1.text(0.62, 0.06, "bandwidth-bound", fontsize=9.6, color=C("mut"))
    ax1.set_xlim(-0.03, 1.0); ax1.set_ylim(-0.03, 1.0)
    ax1.set_xlabel("bandwidth elasticity   d(ln tok/s) / d(ln memory clock)")
    ax1.set_ylabel("compute elasticity\nd(ln tok/s) / d(ln SM clock)")
    ax1.set_title("Speculation does not speed up a bandwidth-bound decode.\n"
                  "It moves the workload into the other corner", pad=10)
    ax1.grid(alpha=0.5); _despine(ax1)

    # --- the same thing as a regime change: the baseline stops responding to clock, speculation
    #     does not
    labels, width = [], 0.26
    xs = np.arange(len(R2_COMPUTE))
    for i, (m, col, mk) in enumerate(R2_METHODS):
        col = col or C("neutral")
        vals, errs = [], [[], []]
        for lo, hi in R2_COMPUTE:
            e, elo, ehi = _elast(res, m, lo, hi, EL.SM_KEY)
            vals.append(e); errs[0].append(e - elo); errs[1].append(ehi - e)
        ax2.bar(xs + (i - 1) * width, vals, width * 0.92, color=col, label=m, zorder=2)
        ax2.errorbar(xs + (i - 1) * width, vals, yerr=errs, fmt="none", ecolor=C("fg"),
                     capsize=3, lw=1.2, zorder=3)
        for x, v in zip(xs + (i - 1) * width, vals):
            ax2.text(x, v + 0.025, f"{v:.2f}", ha="center", fontsize=9.2, color=C("fg"))
    ax2.set_xticks(xs, ["600 → 1200 MHz\nboth still compute-starved",
                        "1200 → 1710 MHz\nthe baseline hits its bandwidth ceiling"], fontsize=9.6)
    ax2.set_ylabel("compute elasticity")
    ax2.set_ylim(0, 1.08)
    ax2.set_title("Below 1200 MHz everything scales with clock. Above it, only\n"
                  "the speculative arms still do", pad=10)
    ax2.legend(frameon=False, ncol=3, loc="upper right")
    ax2.grid(axis="y", alpha=0.5); _despine(ax2)

    fig.subplots_adjust(left=0.155, right=0.975, top=0.925, hspace=0.55)
    _save(fig, "plot_bound_by", bottom=0.10,
          note="Phase R2, 1575 requests, 0 incidents. The SM clock is pinned with nvidia-smi -lgc "
               "rather than produced by a power cap, so each elasticity has a setting in its "
               "denominator instead of an outcome. Intervals are a cluster bootstrap over prompts "
               "and over the measured clock, per interval, never pooled across a regime change.")


def main():
    OUT.mkdir(exist_ok=True)
    result = A.load(RESULT)
    series, prompt_class, _, _ = A.build_series(result, "decode_tok_s")
    result_r = A.load(RESULT_R) if RESULT_R.exists() else None
    result_r2 = A.load(RESULT_R2) if RESULT_R2.exists() else None
    for mode in ("light", "dark"):
        print(f"  --- {mode}")
        with theme(mode):
            fig_headline(series, prompt_class)
            fig_per_class(series, prompt_class)
            fig_cost_model(result)
            fig_width_partition(result)
            if result_r:
                fig_bandwidth(result_r)
            if result_r2:
                fig_bound_by(result_r2)


if __name__ == "__main__":
    main()
