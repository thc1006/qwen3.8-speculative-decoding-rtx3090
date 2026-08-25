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
    """-> [(n_max, Interval, acceptance, is_mtp)] for one target, ordered by n_max."""
    arms = result.get("arms", {})
    acc = _acceptance(result)
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
        out.append((n, iv, acc.get(arm), is_mtp))
    return sorted(out)


def _acceptance(result):
    """-> {arm: accepted/drafted pooled over its records}, or {} when nothing drafted."""
    tot = defaultdict(lambda: [0, 0])
    for rec in result["records"]:
        tm = rec.get("timings") or {}
        tot[rec["arm"]][0] += tm.get("t_draft_n") or 0
        tot[rec["arm"]][1] += tm.get("t_draft_n_accepted") or 0
    return {a: acc / d for a, (d, acc) in tot.items() if d}


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
    lo = min(iv.lo for _, _, _, lad in rows for _, iv, *_ in lad)
    hi = max(iv.hi for _, _, _, lad in rows for _, iv, *_ in lad)
    pad = (hi - lo) * 0.16
    mtp_c, mtp_m = P.STYLE["draft-mtp"]
    simple_c = P.WONG["orange"]

    for ax, (key, title, base, lad) in zip(axes, rows):
        ax.axhline(0, color=P.C("neutral"), lw=1.0, zorder=1)
        mtp = [(n, iv, a) for n, iv, a, is_m in lad if is_m]
        sim = [(n, iv, a) for n, iv, a, is_m in lad if not is_m]
        if mtp:
            ax.plot([n for n, *_ in mtp], [iv.point for _, iv, _ in mtp],
                    color=mtp_c, lw=1.4, zorder=2)
        for group, colour, marker, label in ((mtp, mtp_c, mtp_m, "draft-mtp (built-in head)"),
                                             (sim, simple_c, "D",
                                              "draft-simple, 0.8B drafter")):
            if not group:
                continue
            ax.errorbar([n for n, *_ in group], [iv.point for _, iv, _ in group],
                        yerr=[[iv.point - iv.lo for _, iv, _ in group],
                              [iv.hi - iv.point for _, iv, _ in group]],
                        fmt=marker, color=colour, ecolor=colour, elinewidth=1.3,
                        capsize=3.2, ms=6.4, lw=0, zorder=3, label=label)
        # The peak, named on the figure rather than left to be read off the line.
        if len(mtp) > 1:
            bn, biv, _ = max(mtp, key=lambda t: t[1].point)
            if bn not in (mtp[0][0], mtp[-1][0]):
                ax.annotate(f"peak  n-max {bn}", (bn, biv.hi),
                            textcoords="offset points", xytext=(0, 9), ha="center",
                            fontsize=8.8, color=P.C("mut"))
        for n, iv, a in mtp + sim:
            if a is not None:
                ax.annotate(f"{100 * a:.0f} %", (n, iv.lo), textcoords="offset points",
                            xytext=(0, -13), ha="center", fontsize=8.2, color=P.C("mut"))
        rate = _baseline_rate(series, prompt_class, base)
        ax.set_title(f"{title}     baseline {rate:.0f} tok/s", loc="left", pad=8)
        ax.set_ylabel("net effect vs own baseline (%)")
        ax.set_ylim(lo - pad * 1.5, hi + pad)
        ax.grid(axis="y", lw=0.6, alpha=0.55)
        ax.set_axisbelow(True)
        P._despine(ax)
    # Upper right of the top panel: the only quadrant no series enters. Lower left put the key
    # at the same height as the MoE draft-simple point, where it reads as another datum.
    axes[0].legend(loc="upper right", frameon=False, ncols=1, handletextpad=0.5,
                   borderaxespad=0.2)
    axes[-1].set_xlabel("n-max (draft depth); the annotation under each point is draft acceptance")
    axes[-1].set_xticks(xs_all)
    axes[-1].set_xlim(min(xs_all) - 0.6, max(xs_all) + 0.6)
    fig.suptitle("Same session, same prompts: the sign belongs to the drafter, not the "
                 "architecture", y=0.995, fontsize=12.5, ha="center")
    fig.subplots_adjust(top=0.90, hspace=0.30)
    n_rec = len(result["records"])
    P._save(fig, "plot_phase_m", bottom=0.16, provenance=PROVENANCE,
            note=f"Phase M, {n_rec} requests. " + P.CI_NOTE)


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
