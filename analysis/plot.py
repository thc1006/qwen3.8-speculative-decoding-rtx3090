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
import matplotlib.patheffects as pe
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))

import analyze as A  # noqa: E402
import quality  # noqa: E402
import cost_model as CM  # noqa: E402
import elasticity as EL  # noqa: E402
import stats as ST  # noqa: E402

OUT = ROOT / "analysis"
RESULT = ROOT / "results/phase_a.json"
RESULT_NMAX = ROOT / "results/phase_nmax.json"
RESULT_R2 = ROOT / "results/phase_r2.json"

DPI = 150
FIG_W = 9.0          # inches; 9.0 x 150 = 1350 px, close to GitHub's column
MAX_PX = 1400

# Wong, B. (2011) Points of view: Color blindness. Nature Methods 8:441.
# Wong's palette is specified for print on white and is not a dark-theme palette. Measured
# against this repo's dark background #0d1117: blue #0072B2 is 3.65:1, below WCAG AA for text,
# and it is what carries the fit coefficients at 14-15 px on plot_dispatch_boundary and
# plot_cost_model. Its paired vermillion is 4.89:1, so two annotations that are peers read at
# visibly unequal weight. The dark values below put blue, vermillion and green in a 6.6-8.4 band
# together with the orange that was already there.
#
# This has to be per theme rather than one palette for both: Wong's own sky blue #56B4E9 reaches
# 8.20:1 on the dark background and 2.31:1 on white, so swapping the palette outright would just
# move the failure to the light figures. The light values are unchanged.
#
# The light values are unchanged DESPITE failing the same test, and that is a measured decision
# rather than an oversight. On white, orange is 2.25:1, below the 3.0 floor a data marker is
# supposed to clear; vermillion 3.87, green 3.42 and purple 3.06 all sit under the 4.5 a small
# label needs. Lifting them looks obvious and is not: darkening the four to 4.5 collapses the
# minimum pairwise separation under simulated protanopia from 51.4 to 6.5, and vermillion and
# orange become one colour. Lifting orange ALONE to 3.01:1 puts orange and purple 2.9 apart
# under tritanopia, against 46.7 as published. The reason is structural. Under protan and
# deutan simulation the red-green hue difference is most of what these pairs have, and what
# survives it is the LIGHTNESS difference; equalising lightness to buy contrast spends the one
# channel that was still separating them. Wong's orange is pale because it is doing that work.
#
# So the light figures keep a contrast the guidance would fail, and the compensating property
# is the one this module already commits to: shape repeats every colour distinction, so colour
# is never the only channel. Checked figure by figure -- both series that carry orange are a
# triangle and a diamond against circles and squares, both heatmaps print the number in every
# cell, and the one panel where colour IS the only channel, the elasticity bars, separates at
# 57.6 or better in all three simulations. If a future figure encodes something in orange with
# no second channel, this trade stops holding and the palette is not what to change.
_WONG_LIGHT = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
               "orange": "#E69F00", "purple": "#CC79A7"}
_WONG_DARK = {"blue": "#4DA6E0", "vermillion": "#EE7733", "green": "#1FB894",
              "orange": "#E69F00", "purple": "#E58FC2"}


class _ThemedColours:
    """`WONG["blue"]` resolving against whichever theme is being drawn.

    A plain dict here would have been read once at import, which is before `theme()` runs, so
    every dark figure would have carried the light palette. Kept subscriptable so no call site
    changes: three modules index it directly.
    """

    def __getitem__(self, k):
        return (_WONG_DARK if _MODE == "dark" else _WONG_LIGHT)[k]

    def __iter__(self):
        return iter(_WONG_LIGHT)

    def keys(self):
        return _WONG_LIGHT.keys()


WONG = _ThemedColours()
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
class _ThemedStyle:
    """(colour, marker) per method, with the colour resolved per theme. Same reason as WONG."""

    _COL = {"draft-mtp": "blue", "draft-dflash": "vermillion"}
    _MARK = {"draft-mtp": "o", "draft-dflash": "s"}

    def __getitem__(self, k):
        return (WONG[self._COL[k]], self._MARK[k])

    def items(self):
        return [(k, self[k]) for k in self._MARK]

    def __iter__(self):
        return iter(self._MARK)


STYLE = _ThemedStyle()

PROVENANCE = ("Qwen3.8-27B UD-Q4_K_XL | RTX 3090 24 GB | llama.cpp c060ca9 | greedy, "
              "--parallel 1, fresh server per arm-pass, thermal gate at arm entry | "
              "PREREGISTRATION.md")


def captured_on(result):
    """The date the drawn result was taken, read from the file rather than typed once.

    The footer carried a single hard-coded `2026-08-25` on every figure in this module. The six
    figures draw three phases taken on two different days: `phase_a` and `phase_r2` on
    2026-08-24, `phase_nmax` on 2026-08-25. So five of the six asserted a date one day after
    their own data. A provenance line is the last place a date should come from memory.
    """
    ts = ((result or {}).get("env") or {}).get("captured_at") or ""
    return ts.split("T")[0] or "capture date not recorded"
PHASE_A_N = "Phase A: 875 requests, 0 incidents, 0 excluded."
CI_NOTE = ("Nominal 95 % paired cluster bootstrap over prompts within class, 10 000 "
           "resamples. The inferential unit is the prompt, n = 25; the passes are repeated "
           "measurements. Simulation in this repo puts the percentile interval's actual "
           "coverage at 87.5-92.0 % at that size, so these widths are optimistic.")

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


def _halo(lw=2.2):
    """Outline text in the background colour instead of putting a box behind it.

    The box came first, and it over-corrected. Its job was to stop the OTHER method's line reading
    as a strikethrough across a series label, and it did -- by masking a rectangle. A rectangle is
    wider and taller than the letters in it, so on plot_cost_model the blue `draft-mtp` label was
    erasing 51 to 120 px of the orange draft-dflash LINE in all three panels: measured, no orange
    pixel at all between x 730 and 850 in the first. A reader sees a line that stops and restarts,
    which is what missing data looks like.

    A stroke around the glyphs masks only the glyphs. The line stays continuous between the
    letters, the text stays legible, and nothing that is data gets covered. Kept as a function
    rather than a constant because `C("bg")` resolves per theme at call time.
    """
    return [pe.withStroke(linewidth=lw, foreground=C("bg"))]


def _save(fig, stem, note="", bottom=0.13, provenance=None, top_in=None,
          captured=None, after_layout=None):
    # Footer spacing is computed in inches, not in figure fractions. A fixed fraction is a
    # different number of pixels on every figure height, and on the 4.5-inch panels it put the
    # footer lines on top of each other and on the x-label.
    w_in, h_in = fig.get_size_inches()
    # `provenance` is overridable because the Phase M figures cover two targets and the
    # module-level line names one. A figure that carries the wrong target in its footer is worse
    # than one with no footer.
    prov = provenance or PROVENANCE
    if captured:
        prov = f"{prov} | {captured}"
    lines = textwrap.wrap(" ".join(x for x in (note, prov) if x), width=int(w_in * 15.5))
    line_h, pad = 0.155 / h_in, 0.05 / h_in
    # 0.62 in below the axes for the tick labels and the x-label, then the footer block under it.
    fig.subplots_adjust(bottom=max(bottom, pad + line_h * len(lines) + 0.62 / h_in))
    # `top_in` is the headroom above the topmost panel, in INCHES, for the same reason the footer
    # is: matplotlib's default top fraction of 0.88 is 0.5 in on a 4.5-inch figure and 1.3 in on a
    # 10.5-inch one, so a tall multi-panel figure with no suptitle opens with a band of nothing.
    # Measured on the committed set: plot_qsmall_ladder carried 159 px of blank at the top, 10 %
    # of its height, against 5 to 58 px everywhere else.
    if top_in is not None:
        fig.subplots_adjust(top=1.0 - top_in / h_in)
    for i, ln in enumerate(reversed(lines)):
        fig.text(0.5, pad + i * line_h, ln, ha="center", va="bottom",
                 fontsize=7.6, style="italic", color=C("mut"))
    # Anything that has to MEASURE the layout runs here, not at draw time. A label rotated to
    # lie along a line is the case: the line's angle on screen depends on the axes box, and the
    # axes box is not settled until the two subplots_adjust calls above have run.
    if after_layout is not None:
        fig.canvas.draw()
        after_layout(fig)
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
def fig_headline(series, prompt_class, captured=None):
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
                  [f"{a}  |  w={WIDTH[a]}  |  {m:.1f} tok/s" for a, _, m in rows], fontsize=10)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.set_xlim(0, max(r[1].hi for r in rows) * 1.20)
    ax.set_xlabel("faster than the non-speculative baseline of the same tree  (%)")
    ax.set_title("Every speculative arm is faster, and every interval clears zero\n"
                 f"nominal 95 %  |  baseline {base_abs:.2f} tok/s  |  25 prompts x 5 passes per arm",
                 pad=12)
    ax.grid(axis="x", alpha=0.5, zorder=0)
    _despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[plt.Line2D([], [], color=c, marker=m, ls="none", ms=8, label=k)
                       for k, (c, m) in STYLE.items()],
              loc="lower right", frameon=False)
    fig.subplots_adjust(left=0.315, right=0.975, top=0.80)
    _save(fig, "plot_headline", note=PHASE_A_N + " " + CI_NOTE, captured=captured)


# ------------------------------------------------------------------ 2. where the win lives
def fig_per_class(series, prompt_class, captured=None):
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
    ax.set_yticks(range(len(SPEC_ARMS)), [f"{a}  |  w={WIDTH[a]}" for a in SPEC_ARMS], fontsize=10)
    ax.tick_params(length=0)
    _despine(ax, keep=())
    ax.set_title("The gain is concentrated in code and reasoning, and reverses on\n"
                 "conversational and Chinese prompts as the verification width grows", pad=12)
    fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02).set_label("vs baseline (%)", fontsize=9.6)
    fig.subplots_adjust(left=0.20, right=0.93, top=0.80)
    _save(fig, "plot_per_class", captured=captured,
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
            # only across the widths that were fitted. Extending it past the last point would
            # both collide with the series label and draw a claim the data does not carry.
            gx = np.linspace(k_pts[0][0], k_pts[-1][0], 40)
            ax_k.plot(gx, k0 + c * (gx - 1), color=col, lw=1.1, ls="--", alpha=0.75, zorder=1)
            c_labels.append((f"c = {c:.4f}   ({spec}, Phase A only)", col))

    ref = series_of("draft-mtp", "mean_len")
    if ref:
        w0, y0 = ref[0]
        gx = np.linspace(w0, 8.4, 40)
        ax_n.plot(gx, y0 * gx / w0, color=C("mut"), lw=1.1, ls=":", zorder=1)
        ax_n.plot([0.045, 0.085], [0.90, 0.90], transform=ax_n.transAxes, color=C("mut"),
                  ls=":", lw=1.1, clip_on=False)
        ax_n.text(0.095, 0.90, "growth in proportion to width", transform=ax_n.transAxes,
                  color=C("mut"), fontsize=9.2, va="center")
    ax_n.set_ylabel("tokens accepted\nper target pass")
    ax_n.set_title("The benefit saturates: the gap below the dotted line is what is lost", pad=8)
    # direct labels at the line ends. A legend box has to sit somewhere, and on this panel every
    # somewhere is on top of a line.
    for spec, (col, _) in STYLE.items():
        pts = series_of(spec, "mean_len")
        if pts:
            # Offset down as well as right. The stroke stops the label being struck through, but
            # the letters still sit ON the other method's line and interrupt it: 119 px of the
            # orange line vanished behind the old opaque box, and 60 px still went behind the
            # glyphs. Panel 3's "best tested here" labels already clear their curves this way.
            ax_n.annotate(spec, pts[-1], textcoords="offset points", xytext=(9, -11),
                          color=col, fontsize=10, va="center", fontweight="bold",
                          # The other method's line runs on past this label and through it: at w=6
                          # the dflash line crosses "draft-mtp" and it read as a strikethrough, in
                          # both themes. Stroked, not boxed -- see _halo; the box that used to be
                          # here cut a 120 px hole in the line it was protecting the text from.
                          path_effects=_halo())

    for i, (txt, col) in enumerate(c_labels):
        # The y = 2.50 gridline runs through this text, and the vertical rules cross it too.
        ax_k.text(0.015, 0.93 - i * 0.115, txt, transform=ax_k.transAxes, color=col,
                  fontsize=9.8, va="top", family="monospace", path_effects=_halo())
    for spec, (col, _) in STYLE.items():
        pts = series_of(spec, "k")
        if pts:
            ax_k.annotate(spec, pts[-1], textcoords="offset points", xytext=(9, 0),
                          color=col, fontsize=10, va="center", fontweight="bold",
                          path_effects=_halo())
    ax_k.set_ylabel("cost of one target pass,\nin plain decode steps")
    # "linear ... fixed c" was the old title. The completed ladder shows curvature -- k(w) is
    # concave for draft-mtp on the dense target and the residuals are several times
    # prompt-to-prompt scatter -- so c is a chord across the widths fitted, not a constant
    # marginal cost. See cost_model.py and PREREGISTRATION.md.
    ax_k.set_title("Cost rises with width: c is the average chord over the widths fitted",
                   pad=8)

    ax_s.set_ylim(0.96, 1.92)   # headroom for the labels that sit above the best points
    ax_s.axhline(1.0, color=C("mut"), ls=":", lw=1.1)
    ax_s.text(2.65, 1.005, "break-even", fontsize=9, color=C("mut"), va="bottom")
    ax_s.set_ylabel("speedup\n= tokens / cost")
    ax_s.set_title("Saturating benefit over rising cost: the best TESTED width is a small one",
                   pad=8)
    ax_s.set_xlabel("verification width  w = n-max + 1 as configured\n"
                    "(a drafter that does not fill its budget verifies fewer columns than this)")
    # below and to the right of the best point, with a leader, so the text never crosses a curve
    for spec, (col, _) in STYLE.items():
        pts = series_of(spec, "speedup")
        if pts:
            bw, bv = max(pts, key=lambda p: p[1])
            # anchor the leftmost label left, or it runs under the y-axis title
            lha = "left" if bw <= 3 else "center"
            ax_s.annotate(f"best tested here: w = {bw}   {bv:.2f}x", (bw, bv),
                          textcoords="offset points",
                          xytext=(-6 if lha == "left" else 0, 13), ha=lha, fontsize=10,
                          color=col, fontweight="bold")
        # Stroked, not boxed. This was the one box `_halo` left behind: measured on the light
        # figure, it took 52 px out of the orange draft-dflash line between x 799 and 850,
        # which is the defect the stroke exists to prevent and not a smaller version of it.
        ax_s.annotate(spec, pts[-1], textcoords="offset points", xytext=(9, 0),
                      color=col, fontsize=10, va="center", fontweight="bold",
                      path_effects=_halo())

    for ax in axes:
        ax.grid(alpha=0.5)
        _despine(ax)
    ax_s.set_xticks([3, 4, 5, 6, 7, 8])
    for _ax in axes:
        _ax.set_xlim(2.6, 9.4)   # room for the end labels
    fig.subplots_adjust(left=0.165, right=0.895, top=0.945, hspace=0.30)
    _save(fig, "plot_cost_model", bottom=0.105, captured=captured_on(result),
          note=PHASE_A_N + " k is recovered per request as mean_len / speedup, then averaged over 125 requests "
               "per arm. draft-dflash has two widths here, so its line is determined "
               "rather than fitted. Phase A tests draft-dflash at w = 5 and 8 only, so its best "
               "point here is the better of two. The completed n-max ladder supersedes both "
               "coefficients and both optima: it fits c = 0.2904 for draft-mtp over widths 2-8 and "
               "c = 0.2481 for draft-dflash over 3, 5 and 7, and puts the best width at 3 for both "
               "methods. The dispatch-boundary figure shows that fit. The two series also run on "
               "different llama.cpp trees -- every draft-mtp arm on master, every draft-dflash arm "
               "on PR #27342 -- so drafter and source tree vary together and any gap between them "
               "is a difference between two pinned configurations, not a drafter-specific cost.")


# ------------------------------------------------------------------ 4. the width partition
def _forks(result):
    fork = defaultdict(dict)
    for rec in result["records"]:
        d = rec.get("divergence")
        if rec["arm"] in WIDTH and rec["pass"] == 1 and d:
            fork[rec["prompt"]][rec["arm"]] = quality.fork_cell(d)
    return {p: v for p, v in fork.items() if len(v) == len(SPEC_ARMS)}



def fig_dispatch_boundary(result_nmax):
    """k against verification width across the full ladder, with the MMVQ dispatch limit marked.

    The cost-model figure fits Phase A's three MTP widths and two DFlash2 ones. The n-max ladder
    has seven and three, all inside the dispatch limit, plus a point past it. That point is the
    figure: MMVQ_MAX_BATCH_SIZE is 8, so a wider verification batch takes a different kernel
    family, and it sits well below the line the widths below it define. Fitting one line across
    both regimes drags the MTP coefficient from 0.2904 to 0.2210 and the fit from 0.9958 to
    0.8316, which is what this exists to make visible rather than argued. The pair read 0.2215
    and 0.8304 until 2026-09-01, from the vintage where the on-path fit was 0.2915 and 0.9959.
    """
    rows = CM.collect(result_nmax)
    mmvq_max, _ = CM.recorded_mmvq_max(result_nmax)
    grp = defaultdict(list)
    for r in rows:
        grp[(r["spec_type"], r["width"])].append(r["k"])
    widths = sorted({w for _, w in grp})

    fig, ax = plt.subplots(figsize=(FIG_W, 5.1))
    notes, drops = [], []
    for spec, (col, mk) in STYLE.items():
        pts = sorted((w, fmean(v)) for (s, w), v in grp.items() if s == spec)
        if not pts:
            continue
        on = [(w, k) for w, k in pts if w <= mmvq_max]
        off = [(w, k) for w, k in pts if w > mmvq_max]
        ax.plot([w for w, _ in on], [k for _, k in on], marker=mk, color=col, lw=1.8, ms=9,
                markeredgecolor=C("bg"), markeredgewidth=1.2, label=spec, zorder=3)
        if len(on) < 2:
            continue
        k0, c, r2 = CM._linfit([w - 1 for w, _ in on], [k for _, k in on])
        gx = np.linspace(on[0][0], on[-1][0], 40)
        ax.plot(gx, k0 + c * (gx - 1), color=col, lw=1.1, ls="--", alpha=0.75, zorder=1)
        notes.append((f"c = {c:.4f}, r^2 = {r2:.4f}   ({spec}, widths {on[0][0]}-{on[-1][0]})", col))
        for w, k in off:
            pred = k0 + c * (w - 1)
            ax.plot([w], [pred], marker="_", color=col, ms=13, mew=1.6, alpha=0.8, zorder=2)
            ax.plot([w], [k], marker=mk, color=col, ms=9, mfc=C("bg"), mew=1.8, zorder=3)
            ax.annotate("", xy=(w, k), xytext=(w, pred),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.1, ls=":", alpha=0.85))
            drops.append((w, k, pred, 100 * (k - pred) / pred, col))

    # Room on the right for the deviation labels, so nothing is clipped at the frame.
    ax.set_xlim(widths[0] - 0.55, widths[-1] + 1.5)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.10)

    # Anchored on the MARKER, not on the midpoint of its arrow. Anchoring on the midpoint put
    # each label at a height neither marker sits at, and with both drops at the same width the
    # blue label landed beside the orange marker: the figure said draft-dflash was the -26 % one
    # when the footer and the data say it is draft-mtp. A label belongs next to its referent.
    ordered = sorted(drops, key=lambda d: -d[1])
    for i, (w, k, pred, pct, col) in enumerate(ordered):
        ax.annotate(f"{pct:+.0f} % against the line",
                    xy=(w, k), xytext=(14, 9 if i == 0 else -9),
                    textcoords="offset points", fontsize=9, color=col, va="center", ha="left")

    ax.axvline(mmvq_max + 0.5, color=C("mut"), lw=1.0, alpha=0.55, zorder=0)
    ax.text(mmvq_max + 0.44, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.5,
            f"MMVQ_MAX_BATCH_SIZE = {mmvq_max}", rotation=90, ha="right", va="center",
            fontsize=8.6, color=C("mut"))

    ax.set_xlabel("verification width  (n-max + 1)")
    ax.set_ylabel("k, cost of one verification step\nin plain decode steps")
    ax.set_title("The cost line stops where the kernel does", pad=12)
    ax.set_xticks(widths)
    for i, (txt, col) in enumerate(notes):
        ax.text(0.03, 0.94 - i * 0.075, txt, transform=ax.transAxes, fontsize=9.2, color=col)
    ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.02), frameon=False, fontsize=9.5)
    _despine(ax)
    _save(fig, "plot_dispatch_boundary", bottom=0.13, captured=captured_on(result_nmax),
          note="The two series run on different llama.cpp trees: every draft-mtp arm on master, "
               "every draft-dflash arm on PR #27342. Drafter and source tree vary together and no "
               "arm separates them, because draft-dflash cannot be run on master at all, so the "
               "gap between the two fits is a configuration difference rather than a drafter one. "
               "A width past the dispatch limit takes a different kernel family. Its marker is "
               "open, it is excluded from the fit, and the arrow is the distance from the line "
               "the widths below define. The two open markers are not the same measurement: at "
               "n-max 8 draft-mtp fills 8.93 of its 9 columns and does leave the kernel, while "
               "draft-dflash fills 7.94 and largely does not, which is most of why one sits -26 % "
               "off the line and the other -7 %. At widths 3, 5 and 7 the two fill identically "
               "(2.99, 4.98, 6.95) and agree on 25 of 25 prompts.")


def fig_width_partition(result):
    fork = _forks(result)
    prompts = sorted(fork)
    n = len(SPEC_ARMS)
    # Two arms that both reach the 400-token cap without diverging hold the same cell value, so
    # counting equality alone scores "neither forked" as agreement. That is 18 % of all pairs here
    # and about a fifth of every 100 % block, so both numbers are reported.
    agree = np.array([[100.0 * sum(fork[p][a] == fork[p][b] for p in prompts) / len(prompts)
                       for b in SPEC_ARMS] for a in SPEC_ARMS])
    cens = np.array([[100.0 * sum(fork[p][a] == fork[p][b] == "SAME" for p in prompts) / len(prompts)
                      for b in SPEC_ARMS] for a in SPEC_ARMS])
    lo = [a for a in SPEC_ARMS if WIDTH[a] <= 4]
    hi = [a for a in SPEC_ARMS if WIDTH[a] >= 5]
    cross = min(agree[SPEC_ARMS.index(a), SPEC_ARMS.index(b)] for a in lo for b in hi)
    differ = sum(1 for p in prompts if fork[p][lo[0]] != fork[p][hi[0]])

    fig, ax = plt.subplots(figsize=(FIG_W, 5.6))
    # The diagonal is an arm against itself, so it is 100 % by construction and carries no
    # measurement. Painted with the data colormap it took the top of the scale, which made each
    # block look denser than its evidence: the top-left block showed four dark cells where two
    # are measured. Masking it puts those cells on the neutral fill, and the mark on them becomes
    # legible for the first time -- it was #8a8a8a at 14 pt on the darkest blue, three points of
    # contrast, on a cell a reader could not tell from missing data.
    diag = np.eye(n, dtype=bool)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(C("flat"))
    im = ax.imshow(np.ma.masked_where(diag, agree), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "same arm", ha="center", va="center", color=C("mut"),
                        fontsize=9.5, style="italic")
            else:
                col = "white" if agree[i, j] > 55 else "#111111"
                ax.text(j, i - 0.08, f"{agree[i, j]:.0f}%", ha="center", va="center",
                        fontsize=13, color=col)
                if cens[i, j] > 0:
                    ax.text(j, i + 0.28, f"of which\n{cens[i, j]:.0f} pt censored", ha="center",
                            va="center", fontsize=7.4, color=col, alpha=0.85,
                            linespacing=1.15)
    labels = [f"w={WIDTH[a]}\n{DRAFTER[a]}" for a in SPEC_ARMS]
    ax.set_xticks(range(n), labels, fontsize=10)
    ax.set_yticks(range(n), [f"{a}\nw={WIDTH[a]} | {DRAFTER[a]}" for a in SPEC_ARMS], fontsize=9.4)
    ax.tick_params(length=0)
    _despine(ax, keep=())
    ax.axhline(1.5, color=C("fg"), lw=3.0)
    ax.axvline(1.5, color=C("fg"), lw=3.0)
    ax.set_title("Two signature groups, spanning both drafters, split at the width boundary\n"
                 f"{cross:.0f} % agreement across the groups, and part of the within-group "
                 f"agreement is censored",
                 fontsize=12, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.030, pad=0.025).set_label(
        f"share of the {len(prompts)} prompts on which both arms show\n"
        f"the same first-divergence or censoring signature (%)", fontsize=9)
    fig.subplots_adjust(left=0.175, right=0.885, top=0.845)
    _save(fig, "plot_width_partition", captured=captured_on(result),
          note=PHASE_A_N + f" The diagonal is an arm against itself: 100 % by construction, "
               f"not a measurement, and left unpainted. A cell counts a prompt when both arms "
               f"show the same signature, "
               f"which includes both reaching the 400-token cap without diverging; the second "
               f"number is how much of the cell that is. Those pairs carry no fork position, so a "
               f"block is weaker than 100 % agreement on where output forks. The groups differ on "
               f"{differ} of {len(prompts)} prompts. The blocks span both drafters, so drafter "
               f"identity does not predict the grouping while width does; the four-build "
               f"intervention separately rules out warp count as the cause. Pass 1; identical in "
               f"all five. w is the kernel's ncols_dst.")


# ------------------------------------------------------------------ 5. what speculation is bound by
# Phase R2 pins the SM clock instead of capping power, so both levers move independently and the
# denominator of each elasticity is a setting rather than an outcome. That makes the two axes
# comparable, which is what this figure needs.
R2_HI = ("sm1700-bwlo", "sm1700", "sm1700-bwhi")     # bandwidth sweep at a pinned 1710 MHz core
R2_COMPUTE = (("sm600", "sm1200"), ("sm1200", "sm1700"))
# Names, not colours. A module-level list holding WONG["blue"] would resolve at import, which is
# before theme() runs, so every dark figure would draw the light palette -- the exact failure the
# _ThemedColours docstring describes, and it was already here.
R2_METHODS = [("baseline", None, "o"), ("mtp-n3", "blue", "s"), ("mtp-n7", "green", "^")]


def _r2_colour(key):
    """Resolve one `R2_METHODS` palette key. Both panels of `fig_bound_by` call this.

    They used to resolve it differently. The top panel looked the key up in `WONG`; the bottom
    unpacked it straight into `color=`. "blue" and "green" are also matplotlib colour names, so
    nothing raised: the bars came out #0000FF and #008000 under markers drawn at #0072B2 and
    #009E73, the same two series in two colours in one figure. On the dark background pure blue
    reads 2.20:1 against #0d1117 -- below the 3.0 floor for a graphical object, and worse than
    the 3.65:1 that `_WONG_DARK` was added to fix. A palette lookup that silently succeeds with
    the wrong value is why this is a function and not a second copy of the expression.
    """
    return WONG[key] if key else C("neutral")


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
    OFFS = {"baseline": (14, 0, "left"), "mtp-n3": (-12, -20, "right"), "mtp-n7": (22, 24, "left")}
    for m, key, mk in R2_METHODS:
        col = _r2_colour(key)
        bw = fmean([_elast(res, m, a, b, EL.MEM_KEY)[0]
                    for a, b in zip(R2_HI, R2_HI[1:])])
        cp = _elast(res, m, "sm1200", "sm1700", EL.SM_KEY)[0]
        ax1.plot([bw], [cp], marker=mk, ms=14, color=col, markeredgecolor=C("bg"),
                 markeredgewidth=1.5, zorder=3, label=m)
        dx, dy, ha = OFFS[m]
        ax1.annotate(f"{m}\n({bw:.2f}, {cp:.2f})", (bw, cp), textcoords="offset points",
                     xytext=(dx, dy), fontsize=9.6, color=col, va="center", ha=ha)
    ax1.plot([0, 1], [1, 0], color=C("mut"), ls=":", lw=1.0, zorder=1)

    def _label_the_diagonal(_fig):
        """Rotate the label to the angle the line has on screen, measured rather than guessed.

        This was `rotation=-31`, a number typed once. The angle a data-space slope of -1 renders
        at depends on the axes box, and this figure's box is 7.11 x 2.48 inches, so the line
        actually lies at -19.2 degrees. Nearly twelve degrees out put the far end of a
        468-pixel label about 48 pixels off the line it names. The box is not settled until
        `_save` has adjusted the footer, which is why this runs from there.
        """
        import math
        p0 = ax1.transData.transform((0.0, 1.0))
        p1 = ax1.transData.transform((1.0, 0.0))
        deg = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        # The dotted line runs through the glyphs of its own label, in the same colour, so the
        # halo is what keeps both readable.
        # Anchored at 0.45 rather than 0.62. The label is about 468 px long, so at this angle it
        # spans roughly 0.21 in x either side of its anchor: centred at 0.62 its tail reached
        # x = 0.83 and ran through the baseline marker at (0.80, 0.27) and that marker's own
        # label. At 0.45 it spans 0.24 to 0.66, which is the stretch of the line no point sits on.
        ax1.annotate("x + y = 1 (visual reference; no conservation law is implied)",
                     (0.45, 0.55), fontsize=7.8, color=C("mut"), path_effects=_halo(2.6),
                     rotation=deg, rotation_mode="anchor", ha="center", va="bottom")
    ax1.set_xlim(-0.03, 1.0); ax1.set_ylim(-0.03, 1.0)
    ax1.set_xlabel("memory-clock elasticity   d(ln tok/s) / d(ln memory clock)\n"
                   "further right = responds more to the memory clock")
    ax1.set_ylabel("SM-clock elasticity\nd(ln tok/s) / d(ln SM clock)\n"
                   "higher = responds more to the SM clock")
    # Not "opposite clocks": both respond to both. What reverses is the ORDERING of the two
    # elasticities, which is the claim the panel can carry.
    ax1.set_title("Baseline and speculative arms reverse the ordering of the two elasticities.\n"
                  "A response measurement, not a roofline", pad=10)
    ax1.grid(alpha=0.5); _despine(ax1)

    # --- the same thing as a regime change: the baseline stops responding to clock, speculation
    #     does not
    labels, width = [], 0.26
    xs = np.arange(len(R2_COMPUTE))
    for i, (m, key, mk) in enumerate(R2_METHODS):
        col = _r2_colour(key)
        vals, errs = [], [[], []]
        for lo, hi in R2_COMPUTE:
            e, elo, ehi = _elast(res, m, lo, hi, EL.SM_KEY)
            vals.append(e); errs[0].append(e - elo); errs[1].append(ehi - e)
        ax2.bar(xs + (i - 1) * width, vals, width * 0.92, color=col, label=m, zorder=2)
        ax2.errorbar(xs + (i - 1) * width, vals, yerr=errs, fmt="none", ecolor=C("fg"),
                     capsize=3, lw=1.2, zorder=3)
        for x, v in zip(xs + (i - 1) * width, vals):
            ax2.text(x, v + 0.025, f"{v:.2f}", ha="center", fontsize=9.2, color=C("fg"))
    # "the baseline stops tracking it" was wrong on the figure's own evidence: the bar beside it
    # reads 0.27, and README.md states that both workloads respond to both clocks and only the
    # ORDERING reverses. The tick now says what the bar shows.
    ax2.set_xticks(xs, ["600 -> 1200 MHz\nall three track the SM clock",
                        "1200 -> 1710 MHz\nthe baseline falls to 0.27, the others hold"],
                   fontsize=9.6)
    ax2.set_ylabel("SM-clock elasticity")
    ax2.set_ylim(0, 1.14)
    ax2.set_title("Below 1200 MHz all three track the SM clock. Above it the\n"
                  "baseline's response falls by two thirds and the speculative arms hold", pad=10)
    ax2.legend(frameon=False, ncol=3, loc="upper right", fontsize=9.4)
    ax2.grid(axis="y", alpha=0.5); _despine(ax2)

    fig.subplots_adjust(left=0.185, right=0.975, top=0.935, hspace=0.62)
    _save(fig, "plot_bound_by", bottom=0.10, captured=captured_on(res),
          after_layout=_label_the_diagonal,
          note="Phase R2, 1575 requests, 0 incidents. Elasticity is how throughput responds to a "
               "clock that was set, so it places neither workload against a hardware limit: nothing "
               "here counts bytes moved or arithmetic issued, and no claim is made about which "
               "resource either one is bound by. The SM clock is pinned with nvidia-smi -lgc "
               "rather than produced by a power cap, so each elasticity has a setting in its "
               "denominator instead of an outcome. Intervals are a cluster bootstrap over prompts "
               "and over the measured clock, per interval, never pooled across a regime change.")


def main():
    OUT.mkdir(exist_ok=True)
    result = A.load(RESULT)
    series, prompt_class, _, _ = A.build_series(result, "decode_tok_s")
    result_r2 = A.load(RESULT_R2) if RESULT_R2.exists() else None
    result_nmax = A.load(RESULT_NMAX) if RESULT_NMAX.exists() else None
    for mode in ("light", "dark"):
        print(f"  --- {mode}")
        with theme(mode):
            phase_a_date = captured_on(result)
            fig_headline(series, prompt_class, captured=phase_a_date)
            fig_per_class(series, prompt_class, captured=phase_a_date)
            fig_cost_model(result)
            if result_nmax is not None:
                fig_dispatch_boundary(result_nmax)
            fig_width_partition(result)
            if result_r2:
                fig_bound_by(result_r2)


if __name__ == "__main__":
    main()
