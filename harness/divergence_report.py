"""Byte-level divergence analysis of speculative arms against their own baseline.

Written for llama.cpp issue #27407, which reports that greedy speculative output diverges
deterministically from the non-speculative baseline on CUDA and classifies it as numerical
divergence under batched verification rather than a logic bug. That report covers `draft-simple`
and `draft-dflash` on two workloads with an IQ2_M target and asks for confirmation.

This produces the three things it does not have:

  1. prevalence  -- how often it happens, across a balanced 25-prompt set and every arm
  2. `draft-mtp` -- the built-in head, which the report does not test
  3. the batch-shape signature -- whether the fork POSITION is shared by drafters that have
     nothing in common except how many positions they verify at once. If two unrelated drafters
     fork at the same character, and that character moves when n-max changes, the reduction
     order under batched verification is doing the work, which is precisely the report's claim.

Determinism across passes is reported alongside, because a divergence that is not reproducible
is a different bug from one that is.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import quality  # noqa: E402
import truncation_audit as TA  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _width(arms: dict, arm: str) -> int:
    """Verification width, n_max + 1, from the arm's own recorded flags."""
    ea = (arms.get(arm) or {}).get("extra_args") or []
    for i, t in enumerate(ea):
        if t == "--spec-draft-n-max" and i + 1 < len(ea):
            try:
                return int(ea[i + 1]) + 1
            except ValueError:
                return 1
    return 1


def _family(arms: dict, arm: str) -> str:
    """Drafter family, so arms that share only a width can be told from arms that share more."""
    ea = (arms.get(arm) or {}).get("extra_args") or []
    for i, t in enumerate(ea):
        if t == "--spec-type" and i + 1 < len(ea):
            return ea[i + 1]
    return "none"


def report(result: dict) -> None:
    arms = result.get("arms", {})
    recs = result["records"]

    # arm -> prompt -> pass -> divergence record
    div: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for r in recs:
        d = r.get("divergence")
        if d:
            div[r["arm"]][r["prompt"]][r["pass"]] = d

    if not div:
        # The old message guessed at two causes and named neither of the common ones, which sent
        # a reader hunting through bench.py for a defect that was not there. The comparison is
        # attached POST-PASS, because arm order rotates within a pass and the baseline arm can
        # run after the arms measured against it, so a file whose first pass has not closed
        # carries no divergence at all and that is not a fault. Say which case this is.
        by_pass = defaultdict(set)
        for r in recs:
            by_pass[r["pass"]].add(r["arm"])
        want = {a for a, m in arms.items()
                if m.get("extra_args") or m.get("expects_drafter")} or set(arms)
        complete = [p for p, seen in sorted(by_pass.items()) if want <= seen]
        temps = {r.get("temperature") for r in recs}
        print("no divergence records. Diagnosis:")
        if not recs:
            print("  the file holds no records at all.")
        elif not complete:
            print(f"  no pass has closed yet: {len(recs)} record(s) across pass(es) "
                  f"{sorted(by_pass)}, and the fullest holds "
                  f"{max(len(v) for v in by_pass.values())} of the {len(want)} arms this matrix "
                  f"declares. Divergence is attached after a pass ends, because arm order rotates"
                  f" and the baseline can run after the arms measured against it. Re-run this "
                  f"once a pass completes.")
        elif temps and temps != {0.0}:
            shown = sorted(t for t in temps if t is not None)
            print(f"  the arms are not all greedy: temperatures present are {shown}. "
                  f"Divergence is only attached at temperature 0.")
        elif any(r.get("baseline_comparison_unavailable") for r in recs):
            miss = sorted({r.get("baseline_comparison_wanted") for r in recs
                           if r.get("baseline_comparison_unavailable")} - {None})
            print(f"  the baseline text was missing for some arms; they wanted {miss}. "
                  f"Check divergence_baseline_map in the result file.")
        else:
            print("  passes closed and the arms are greedy, so this is unexpected: check "
                  "_attach_baseline_comparisons in bench.py.")
        return

    print("=" * 100)
    print("BYTE-LEVEL DIVERGENCE FROM THE NON-SPECULATIVE BASELINE (greedy, same prompt & pass)")
    print("=" * 100)

    # `~tokens` carries a tilde because it is not measured. It divides the exact character offset
    # by the output's MEAN characters per token, and a tokenizer is variable-length, so the figure
    # locates the fork to within a stretch of output rather than to a token. The character column
    # beside it is exact and is what the width partition is built on. See
    # truncation_audit.chars_per_token.
    #
    # The earliest fork is the one number here taken across prompts rather than within one, and
    # a character index does not mean the same thing in each class: measured on Phase A the
    # output runs 1.56 characters per token in Chinese against 4.65 in prose, a spread of 3.0.
    # Reported in tokens beside it, from each record's own ratio, so the column can be compared.
    tok_earliest: dict[str, float] = {}
    for r in recs:
        d = r.get("divergence")
        if not d or d.get("identical"):
            continue
        i = quality.fork_position(d)
        cpt = TA.chars_per_token(r)
        if i is None or not cpt:
            continue
        t = i / cpt
        if r["arm"] not in tok_earliest or t < tok_earliest[r["arm"]]:
            tok_earliest[r["arm"]] = t

    print("\n--- prevalence ---")
    print(f"{'arm':22s} {'n':>5s} {'identical':>10s} {'rate':>7s} "
          f"{'median shared prefix':>21s}  {'earliest fork':>13s} {'~tokens':>8s}")
    for arm in sorted(div):
        flat = [d for p in div[arm].values() for d in p.values()]
        ident = sum(1 for d in flat if d["identical"])
        forks = [d for d in flat if not d["identical"]]
        med = statistics.median(d["common_prefix_frac"] for d in forks) if forks else None
        earliest = min((i for i in map(quality.fork_position, forks) if i is not None),
                       default=None)
        te = tok_earliest.get(arm)
        print(f"{arm:22s} {len(flat):5d} {ident:10d} {ident/len(flat)*100:6.1f}% "
              f"{(f'{med:.3f}' if med is not None else '-'):>21s}  "
              f"{(str(earliest) if earliest is not None else '-'):>13s} "
              f"{(f'{te:.0f}' if te is not None else '-'):>8s}")

    print("\n--- batch-shape signature: fork position per prompt, by arm ---")
    print("    (identical fork positions across drafters that share only their verification")
    print("     width is the signature of reduction-order-dependent argmax flips)")
    prompts = sorted({p for a in div for p in div[a]})
    arm_names = sorted(div)
    w = max(len(p) for p in prompts) + 1
    print(f"\n{'prompt':{w}s} " + " ".join(f"{a[:12]:>13s}" for a in arm_names))
    shared_signature = 0
    cross_family = 0
    comparable = 0
    for pr in prompts:
        cells = []
        pos = []
        for a in arm_names:
            passes = div[a].get(pr, {})
            d = passes.get(min(passes)) if passes else None
            if d is None:
                cells.append(f"{'-':>13s}")
            elif d["identical"]:
                cells.append(f"{'SAME':>13s}")
            else:
                cells.append(f"{('@' + str(d['first_diff_char'])):>13s}")
                pos.append(d["first_diff_char"])
        if len(pos) >= 2 and len(set(pos)) < len(pos):
            shared_signature += 1
        # The caption asks for drafters that share ONLY their width. Arms of one family at one
        # width share far more than that, so counting them here is counting the thing the claim
        # assumes. On phase_c the three DFlash2 arms differ only in drafter quantization and
        # agree on 25 of 25, which alone produces a pooled 20 of 25 and carries no information
        # about width. Reported apart: one representative per family, then the count.
        fam_pos = {}
        for a in arm_names:
            passes = div[a].get(pr, {})
            dd = passes.get(min(passes)) if passes else None
            if dd is None or dd["identical"]:
                continue
            f, wd = _family(arms, a), _width(arms, a)
            fam_pos.setdefault((f, wd), dd["first_diff_char"])
        by_width = defaultdict(list)
        for (f, wd), v in fam_pos.items():
            by_width[wd].append(v)
        if any(len(v) >= 2 and len(set(v)) < len(v) for v in by_width.values()):
            cross_family += 1
        if len({(f, wd) for (f, wd) in fam_pos}) and any(
                len({f for (f, wd2) in fam_pos if wd2 == wd}) >= 2 for wd in by_width):
            comparable += 1
        print(f"{pr:{w}s} " + " ".join(cells))
    print(f"\n  prompts where at least two arms fork at the SAME character: "
          f"{shared_signature}/{len(prompts)}")
    if comparable:
        print(f"  of the {comparable} prompts where two families are comparable at one width,")
        print(f"  they land on the same character on                               : "
              f"{cross_family}/{comparable}")
        print("  The first line counts same-family arms too, and a family at one width agrees")
        print("  with itself by construction, so where the two numbers are far apart the first")
        print("  one is measuring the family and not the width.")
    else:
        print("  NOT MEASURABLE HERE: no width in this file carries two drafter families, so the")
        print("  caption's criterion cannot be evaluated. The line above counts arms that differ")
        print("  in width, in family, or in neither, and on its own says which of those only in")
        print("  a matrix built to separate them. phase_nmax is the file that does.")

    print("\n--- does the fork position move with n-max? ---")
    groups: dict[str, list[int]] = defaultdict(list)
    for pr in prompts:
        for a in arm_names:
            passes = div[a].get(pr, {})
            d = passes.get(min(passes)) if passes else None
            if d and not d["identical"]:
                groups[f"{pr}|{a}"] = [d["first_diff_char"]]
    for pr in prompts:
        row = []
        for a in arm_names:
            v = groups.get(f"{pr}|{a}")
            row.append(f"{a.split('@')[0]}={v[0]}" if v else None)
        row = [x for x in row if x]
        if len(row) >= 2:
            print(f"  {pr:24s} " + "  ".join(row))

    print("\n--- determinism across passes (same arm, same prompt) ---")
    print(f"{'arm':22s} {'checked':>8s} {'reproducible':>13s}")
    for arm in sorted(div):
        tot = rep = 0
        for pr, passes in div[arm].items():
            if len(passes) < 2:
                continue
            shas = {d["sha_arm"] for d in passes.values()}
            tot += 1
            rep += (len(shas) == 1)
        if tot:
            print(f"{arm:22s} {tot:8d} {rep:13d}")
        else:
            print(f"{arm:22s} {'-':>8s} {'(need >1 pass)':>13s}")

    print("\n--- arm metadata (for a bug report) ---")
    for a in arm_names:
        meta = arms.get(a, {})
        print(f"  {a:22s} tree={meta.get('tree','?'):9s} args={' '.join(meta.get('extra_args', []))}")
    env = result.get("env", {})
    print(f"\n  gpu           : {env.get('gpu')}")
    print(f"  llama.cpp     : {env.get('llama_cpp_revisions')}")
    print(f"  model sha256  : {env.get('model_sha256')}")
    print(f"  overclock     : {env.get('overclock_state')}")
    print(f"  design        : greedy, sampler chain pinned, cache_prompt="
          f"{result.get('design', {}).get('cache_prompt')}, "
          f"max_tokens={result.get('design', {}).get('max_tokens')}")


def group_stability(result: dict) -> None:
    """Is the fork-position grouping a stable property, or a single-pass coincidence?

    Pass 1 showed the arms partitioning into exactly two fork-position groups, with the
    partition tracking n-max ({2,3} vs {4,5,7}) rather than which drafter was used -- an
    independent 1.1 GB block-diffusion drafter landing on the same character as the target's own
    built-in head. That is a strong claim and it is only worth making if it reproduces.

    Three checks, reported separately because they can fail independently:
      1. determinism   -- same arm, same prompt, different pass: same fork character?
      2. partition     -- within one pass, which arms share a fork character?
      3. consistency   -- is the partition the same in every pass?
    """
    recs = result["records"]
    fork: dict[tuple[str, str, int], object] = {}
    for r in recs:
        d = r.get("divergence")
        if d:
            fork[(r["arm"], r["prompt"], r["pass"])] = (
                quality.fork_cell(d))

    arms = sorted({a for a, _, _ in fork})
    prompts = sorted({p for _, p, _ in fork})
    passes = sorted({q for _, _, q in fork})

    print("\n" + "=" * 100)
    print(f"GROUP STABILITY  ({len(passes)} passes: {passes})")
    print("=" * 100)

    if len(passes) < 2:
        print("\nonly one pass present -- determinism and consistency cannot be checked yet.")
        return

    # ---- 1. determinism -------------------------------------------------
    print("\n--- 1. determinism: same arm + prompt across passes ---")
    print(f"{'arm':22s} {'prompts':>8s} {'stable':>7s} {'unstable':>9s}  examples")
    any_unstable = False
    for a in arms:
        stable = unstable = 0
        ex = []
        for p in prompts:
            vals = [fork.get((a, p, q)) for q in passes]
            vals = [v for v in vals if v is not None]
            if len(vals) < 2:
                continue
            if len(set(vals)) == 1:
                stable += 1
            else:
                unstable += 1
                if len(ex) < 3:
                    ex.append(f"{p}={vals}")
        if unstable:
            any_unstable = True
        print(f"{a:22s} {stable+unstable:8d} {stable:7d} {unstable:9d}  {'; '.join(ex)}")
    if not any_unstable:
        print("\n  every fork position reproduces exactly across passes.")

    # ---- 2 & 3. partition per pass, and whether it is the same each time --
    print("\n--- 2/3. fork-position partition of arms, per pass ---")
    sigs: dict[int, str] = {}
    for q in passes:
        groups_per_prompt = []
        for p in prompts:
            by_pos: dict[object, list[str]] = {}
            for a in arms:
                v = fork.get((a, p, q))
                # `fork_cell` returns a character index for a fork and a string for every state
                # that has none: SAME when the texts never disagreed inside the window, PREFIX
                # when the shorter one ran out, - when there is no record. Letting those into
                # `by_pos` groups arms by a position none of them has: every censored arm lands
                # in one bucket and reads as a shared fork. It is how `baseline@pr27342`, which
                # never diverges at all, came to be a group in a fork-position partition.
                if not isinstance(v, int):
                    continue
                by_pos.setdefault(v, []).append(a)
            # only informative when at least two arms diverge and agree
            parts = sorted(tuple(sorted(v)) for v in by_pos.values())
            groups_per_prompt.append(tuple(parts))
        # the modal partition of this pass
        from collections import Counter
        c = Counter(groups_per_prompt)
        modal, n = c.most_common(1)[0]
        sigs[q] = repr(modal)
        # An empty modal partition is a real answer and used to print as a dangling arrow. It
        # means that on the commonest prompt shape no arm had a determined fork position at all,
        # which is what BF16 looks like: 156 of its 300 cells never diverge. Saying so beats
        # printing nothing after the arrow.
        pretty = (" | ".join("{" + ",".join(g) + "}" for g in modal) if modal
                  else "(no arm has a determined fork position on the modal prompt)")
        print(f"  pass {q}: modal partition on {n}/{len(prompts)} prompts -> {pretty}")

    same = len(set(sigs.values())) == 1
    print(f"\n  partition identical in every pass: {same}")
    if same and all(sigs[q] == repr(()) for q in passes):
        # Stability of nothing is not evidence of anything. Without this the report said an empty
        # partition was "a stable property of the configuration".
        print("  -> and it is empty in every pass, which says the arms could not be told apart "
              "here rather than that they behave alike.")
    elif same:
        print("  -> the grouping is a stable property of the configuration, not a single-pass "
              "coincidence.")
    else:
        print("  -> the grouping VARIES between passes. Do not claim it is batch-shape "
              "determined without explaining the variation.")

    # ---- does the partition line up with n-max rather than with the drafter? ----
    print("\n--- 4. does the partition track n-max or the drafter? ---")
    meta = result.get("arms", {})
    def nmax(a: str):
        args = meta.get(a, {}).get("extra_args", [])
        if "--spec-draft-n-max" in args:
            return int(args[args.index("--spec-draft-n-max") + 1])
        return None
    def drafter(a: str):
        args = meta.get(a, {}).get("extra_args", [])
        if "--spec-type" in args:
            return args[args.index("--spec-type") + 1]
        return "none"
    for a in arms:
        print(f"  {a:22s} spec-type={drafter(a):14s} n-max={nmax(a)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    a = ap.parse_args()
    res = load(Path(a.result))
    report(res)
    group_stability(res)


if __name__ == "__main__":
    main()
