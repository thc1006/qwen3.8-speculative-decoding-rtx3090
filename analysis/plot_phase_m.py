"""The Phase M figure: two targets, two speculative paths, one session.

Phase M is the only matrix in this study that runs a dense target and an MoE target against their
own baselines in the same session, on the same prompts, in interleaved order. That is what lets a
difference between them be read as a difference between them, rather than as a difference between
two days.

The figure carries the whole result at once:

  * `draft-mtp` is a net WIN on both targets and `draft-simple` with a 0.8B drafter is a large net
    loss on both, so the sign is a property of the speculative path, not of the architecture;
  * both MTP ladders peak at n-max 2, an interior maximum, not at the smallest depth;
  * acceptance is printed under every point, and it is near-identical across the two targets for
    the same drafter. Equal acceptance with unequal net effect is the observation that separates
    "the MoE saturates its experts" from "the overhead is amortised against a faster baseline".

Layout follows plot.py's rule: the two targets stack, they do not sit side by side, because
GitHub scales anything wider than its column and a side-by-side pair arrives unreadable.

    .venv/bin/python analysis/plot_phase_m.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "harness"))

import matplotlib.pyplot as plt  # noqa: E402
import analyze as A  # noqa: E402
import plot as P  # noqa: E402
import speclen  # noqa: E402
import stats as ST  # noqa: E402

RESULT = ROOT / "results/phase_m.json"

# Two targets, named as the reader knows them rather than by file name.
TARGETS = [
    ("moe", "MoE  Qwen3.6-35B-A3B  UD-Q4_K_XL  (3B active of 35B)", "baseline-moe"),
    ("dense", "Dense  Qwen3.8-27B  UD-Q4_K_XL", "baseline-dense"),
]
PROVENANCE = ("Phase M | Qwen3.6-35B-A3B and Qwen3.8-27B, both UD-Q4_K_XL | RTX 3090 24 GB | "
              "llama.cpp c060ca9 | greedy, --parallel 1, fresh server per arm-pass, matched "
              "arms interleaved within a pass | PREREGISTRATION.md")


def _ladder(series, prompt_class, result, prefix, baseline):
    """-> [(n_max, Interval, acceptance, is_mtp, width_eff)] for one target, ordered by n_max."""
    arms = result.get("arms", {})
    acc = _acceptance(result)
    weff = _effective_width(result)
    out = []
    for arm, meta in arms.items():
        if not arm.startswith(prefix + "-") or not meta.get("expects_drafter"):
            continue
        if arm not in series or baseline not in series:
            continue
        ea = meta.get("extra_args", [])
        if "--spec-draft-n-max" not in ea:
            continue
        n = int(ea[ea.index("--spec-draft-n-max") + 1])
        is_mtp = "draft-mtp" in ea
        arm_s, base_s = A._balanced(series[arm], series[baseline])
        if not arm_s:
            continue
        # (baseline, arm). Reversed, every effect in this figure changes sign.
        iv = ST.paired_cluster_bootstrap(base_s, arm_s, prompt_class, relative=True)
        out.append((n, iv, acc.get(arm), is_mtp, weff.get(arm)))
    # Sort on the plain fields only. `sorted(out)` compared whole tuples, and the moment two arms
    # of different methods shared an n-max -- moe-mtp-n2 and moe-draft08b-n2, once the
    # draft-simple ladder reached depth 2 -- the tie fell through to comparing two Intervals,
    # which raises. It worked until the data changed shape.
    return sorted(out, key=lambda t: (t[0], not t[3]))


def _acceptance(result):
    """-> {arm: accepted/drafted pooled over its records}, or {} when nothing drafted."""
    tot = defaultdict(lambda: [0, 0])
    for rec in result["records"]:
        tm = rec.get("timings") or {}
        tot[rec["arm"]][0] += tm.get("t_draft_n") or 0
        tot[rec["arm"]][1] += tm.get("t_draft_n_accepted") or 0
    return {a: acc / d for a, (d, acc) in tot.items() if d}


def _effective_width(result):
    """-> {arm: mean verified columns per target forward pass}.

    `n-max` is what the flag asked for. The drafter can stop earlier, and the server reuses a
    surviving draft tail instead of re-drafting, so what the target actually verifies is one plus
    the drafts it had. On Phase M the MTP arms sit on `n_max` to two decimals and the 0.8B
    `draft-simple` arms sit far below it -- n-max 8 verifies about 5.2 columns, n-max 4 about 3.3
    -- which is why an axis labelled `n-max` alone would put those points to the right of the
    depth they actually ran.
    """
    tot = defaultdict(lambda: [0, 0])
    for rec in result["records"]:
        drafted = (rec.get("timings") or {}).get("t_draft_n") or 0
        # speclen.forwards, never a local copy of `predicted_n - accepted - 1`. That module exists
        # because this study already shipped the derivation three times and they parted company;
        # it also returns the EXACT `draft_n_verif_steps` counter when a record carries one, so a
        # private copy would keep guessing after the llama.cpp patch that exposes it lands.
        f = speclen.forwards(rec)
        if drafted and f:
            tot[rec["arm"]][0] += drafted
            tot[rec["arm"]][1] += f
    return {a: d / f + 1 for a, (d, f) in tot.items() if f}


def _baseline_rate(series, prompt_class, arm):
    return P._strat_mean(series, prompt_class, arm) if arm in series else float("nan")


def fig_phase_m(result, series, prompt_class):
    rows = [(key, title, base, _ladder(series, prompt_class, result, key, base))
            for key, title, base in TARGETS]
    rows = [r for r in rows if r[3]]
    if not rows:
        print("  phase M: no paired arm has data yet")
        return

    fig, axes = plt.subplots(len(rows), 1, figsize=(P.FIG_W, 3.25 * len(rows) + 0.6),
                             sharex=True)
    if len(rows) == 1:
        axes = [axes]

    xs_all = sorted({n for _, _, _, lad in rows for n, *_ in lad})
    # Categorical positions, not the value of n-max. The ladder runs 1, 2, 3, 4, 5, 7, 8, 16, and
    # on a linear axis the single n-max 16 point stretches everything else into the left third.
    # Nothing here is read off the x distance; the ticks carry the depth.
    pos = {n: i for i, n in enumerate(xs_all)}
    lo = min(iv.lo for _, _, _, lad in rows for _, iv, *_ in lad)
    hi = max(iv.hi for _, _, _, lad in rows for _, iv, *_ in lad)
    pad = (hi - lo) * 0.16
    mtp_c, mtp_m = P.STYLE["draft-mtp"]
    simple_c = P.WONG["orange"]

    for ax, (key, title, base, lad) in zip(axes, rows):
        # Above the series, not under it. Zero is the reference every point in this figure is
        # read against, and a marker that covers it takes the reader's only anchor away.
        ax.axhline(0, color=P.C("neutral"), lw=1.0, zorder=4)
        mtp = [(n, iv, a, w) for n, iv, a, is_m, w in lad if is_m]
        sim = [(n, iv, a, w) for n, iv, a, is_m, w in lad if not is_m]
        if mtp:
            ax.plot([pos[n] for n, *_ in mtp], [iv.point for _, iv, *_ in mtp],
                    color=mtp_c, lw=1.4, zorder=2)
        for group, colour, marker, label in ((mtp, mtp_c, mtp_m, "draft-mtp (built-in head)"),
                                             (sim, simple_c, "D",
                                              "draft-simple, 0.8B drafter")):
            if not group:
                continue
            ax.errorbar([pos[n] for n, *_ in group], [iv.point for _, iv, *_ in group],
                        yerr=[[iv.point - iv.lo for _, iv, *_ in group],
                              [iv.hi - iv.point for _, iv, *_ in group]],
                        fmt=marker, color=colour, ecolor=colour, elinewidth=1.3,
                        # 6.4 rendered as a 17 px disc, wider than the interval it sits on: at
                        # n-max 7 the whole confidence interval fitted inside the marker and the
                        # zero line went behind it, so the one point that decides "positive on
                        # both targets" was the one point whose sign could not be read off the
                        # figure. The marker is data about the point estimate; the interval is
                        # data about what the study can say, and the smaller of the two has to win.
                        capsize=3.2, ms=4.6, lw=0, zorder=3, label=label)
        # The peak, named on the figure rather than left to be read off the line.
        if len(mtp) > 1:
            bn, biv = max(mtp, key=lambda t: t[1].point)[:2]
            if bn not in (mtp[0][0], mtp[-1][0]):
                ax.annotate(f"peak  n-max {bn}", (pos[bn], biv.hi),
                            textcoords="offset points", xytext=(0, 9), ha="center",
                            fontsize=8.8, color=P.C("mut"))
        for n, iv, a, w in mtp + sim:
            bits = []
            if a is not None:
                bits.append(f"{100 * a:.0f} %")
            # Only where the drafter did not fill its budget. Printing "w 6.0" beside n-max 5 on
            # every MTP point would be noise; printing nothing beside a draft-simple point that
            # asked for 8 and verified 5.2 would put it at the wrong depth without saying so.
            if w is not None and abs(w - (n + 1)) > 0.25:
                bits.append(f"w {w:.1f}")
            if bits:
                # The offset places these under the interval, and at this y-scale that lands
                # some of them on the zero reference line -- "65 %" in the MoE panel and "37 %"
                # in the dense one were struck through by it.
                #
                # The first fix put an opaque box behind the number, on the reading that the
                # line is a decoration and the number is data. Both halves of that were wrong.
                # The box is a rectangle around glyphs, so it erased whole segments of the
                # line: four of them, three in the MoE panel and one in the dense one. And in
                # a plot of net effect against the baseline, zero is where the sign changes --
                # it carries the claim in the title, so it is not a decoration. `P._halo`
                # strokes the glyphs instead, which masks only the glyphs and leaves the line
                # continuous between them.
                ax.annotate("\n".join(bits), (pos[n], iv.lo), textcoords="offset points",
                            xytext=(0, -13), ha="center", va="top", fontsize=8.2,
                            color=P.C("mut"), linespacing=1.15, zorder=4,
                            path_effects=P._halo(2.4))
        rate = _baseline_rate(series, prompt_class, base)
        ax.set_title(f"{title}     baseline {rate:.0f} tok/s", loc="left", pad=8)
        ax.set_ylabel("net effect vs own baseline (%)")
        # Extra room below: the annotations hang under the lowest point and the deepest
        # draft-simple arm sits near the floor.
        ax.set_ylim(lo - pad * 2.6, hi + pad)
        ax.grid(axis="y", lw=0.6, alpha=0.55)
        ax.set_axisbelow(True)
        P._despine(ax)
    # Upper right of the top panel: the only quadrant no series enters. Lower left put the key
    # at the same height as the MoE draft-simple point, where it reads as another datum.
    axes[0].legend(loc="upper right", frameon=False, ncols=1, handletextpad=0.5,
                   borderaxespad=0.2)
    axes[-1].set_xlabel("n-max, the depth requested; annotations are draft acceptance, and the "
                        "columns actually verified where those differ")
    axes[-1].set_xticks(list(pos.values()))
    axes[-1].set_xticklabels([str(n) for n in xs_all])
    # The positions are ordinal -- enumerate(), one slot per tested depth -- while the labels are
    # the depths themselves. Where the depths are not consecutive the axis draws eight units in
    # the width of one, and nothing said so: the 8 -> 16 step read as a single step, which makes
    # the slope across it look eight times shallower than it is. A break mark on the spine is the
    # standard way to say "this axis is not to scale here", and it is drawn on every gap, not
    # just the one this data happens to have.
    for _a, _b in zip(xs_all, xs_all[1:]):
        if _b - _a <= 1:
            continue
        _mid = (pos[_a] + pos[_b]) / 2.0
        for _ax in axes:
            for _dx in (-0.045, 0.045):
                _ax.plot([_mid + _dx - 0.05, _mid + _dx + 0.05], [0, 0.028],
                         transform=_ax.get_xaxis_transform(), clip_on=False,
                         color=P.C("fg"), lw=1.1, zorder=6)
    axes[-1].set_xlim(-0.6, len(xs_all) - 0.4)
    # The title states what the panels show and nothing beyond it. It used to read "the sign
    # belongs to the drafter, not the architecture", which is a causal claim this phase does not
    # support: its preregistered replication anchor failed, 33 early-stop records are excluded
    # from the per-protocol series, and the mean_len derivation these panels' companions rest on
    # fails its own integrity check here. Readers fixate on the title longer than on anything
    # else in a figure, so an assertive title has to be one the data carries.
    # 78 characters. The one it replaced was 103 and overflowed the figure at this font size,
    # losing the first and last letter -- checked by rendering and reading the PNG, which is the
    # only way a title's width is ever actually known.
    fig.suptitle("Same session, same prompts: MTP positive on both targets, draft-simple "
                 "negative", y=0.995, fontsize=12.5, ha="center")
    fig.subplots_adjust(top=0.90, hspace=0.30)
    n_rec = len(result["records"])
    P._save(fig, "plot_phase_m", bottom=0.16, provenance=PROVENANCE,
            note=f"Phase M, {n_rec} requests. n-max is what the flag asked for; the 0.8B "
                 f"draft-simple arms stop short of it, so they verify fewer columns than their "
                 f"position on the axis suggests and the difference is annotated. That runs "
                 f"against the gap shown, not with it: the arm verifying more columns is the "
                 f"faster one. " + P.CI_NOTE)


def main():
    if not RESULT.exists():
        print(f"  {RESULT} does not exist yet")
        return
    P.OUT.mkdir(exist_ok=True)
    result = A.load(RESULT)
    series, prompt_class, _, _ = A.build_series(result, "decode_tok_s")
    for mode in ("light", "dark"):
        print(f"  --- {mode}")
        with P.theme(mode):
            fig_phase_m(result, series, prompt_class)


if __name__ == "__main__":
    main()
