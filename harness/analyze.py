"""Turn a bench.py result file into the pre-registered report.

Reporting rules, fixed in PREREGISTRATION.md and enforced here rather than left to prose:

* The headline for each arm is the class-stratified paired effect against its OWN-TREE
  baseline. Per-class effects are printed too, and are labelled exploratory.
* An interval that spans zero prints as "no detected effect". The point estimate is shown but
  is never given a direction in the verdict column.
* Requests that were flagged -- early termination, degeneracy, collapse relative to baseline --
  are excluded from the throughput aggregate and counted in a separate integrity block. A
  benchmark that silently averages in a collapsed generation reports a broken arm as the winner.
"""
from __future__ import annotations

import argparse
import json
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import completeness as _CO  # noqa: E402
import speclen  # noqa: E402
import statistics
from collections import defaultdict
from pathlib import Path

import stats as ST


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


# An exclusion rule that can empty the dataset is more dangerous than the artifact it guards
# against. An early version of this function excluded on the ABSOLUTE degeneracy flag, and a
# pipeline test showed it removing 100% of records -- the baseline included, since the baseline
# has no reference of its own to be judged against. Exclusion is therefore driven by the
# comparison against the baseline's own output for the same prompt, plus an extreme absolute
# floor that only unambiguous collapse can cross. Everything else is recorded and reported, not
# thrown away.
EXTREME_DISTINCT_RATIO = 0.05      # unambiguous collapse; real prose/code never lands here


def _usable(rec: dict) -> tuple[bool, str]:
    """Whether a record enters the per-protocol series.

    Dropping a record that stopped before the cap is post-treatment selection whenever the
    treatment can move the stopping point, and speculative decoding can: 76 to 80 % of these
    requests diverge from their baseline, so an arm can reach an end-of-text token the baseline
    did not. On the fixed-400-token matrices nothing is dropped, because every record hits the
    cap. It starts to matter on any run with a budget large enough to finish, which is what
    TODO.md D2 does, so `report()` prints the intention-to-treat series beside this one rather
    than letting the exclusion happen quietly.
    """
    if rec.get("decode_tok_s") in (None, 0):
        return False, "no decode rate"
    if not rec.get("hit_cap", True):
        return False, f"early stop (n={rec.get('predicted_n')})"
    if (rec.get("rel_degeneracy") or {}).get("collapsed"):
        return False, "collapsed vs baseline"
    dr = (rec.get("degeneracy") or {}).get("distinct_ratio")
    if dr is not None and dr < EXTREME_DISTINCT_RATIO:
        return False, f"extreme collapse (distinct ratio {dr:.3f})"
    return True, ""


def _flagged(rec: dict) -> str:
    """Recorded-but-not-excluded quality concerns, so they stay visible."""
    if (rec.get("degeneracy") or {}).get("is_degenerate"):
        return (rec["degeneracy"].get("reason") or "degenerate")[:70]
    return ""


def build_series_itt(result: dict, metric: str = "decode_tok_s"):
    """Every record that carries the metric, with no exclusion applied.

    The intention-to-treat counterpart of build_series. If the two disagree, the exclusion rule is
    doing work and has to be justified rather than assumed harmless.
    """
    series: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    prompt_class: dict[str, str] = {}
    for rec in result["records"]:
        prompt_class[rec["prompt"]] = rec["class"]
        v = rec.get(metric)
        if v is not None:
            series[rec["arm"]][rec["prompt"]].append(float(v))
    return series, prompt_class


def build_series(result: dict, metric: str = "decode_tok_s"):
    """-> {arm: {prompt: [values...]}}, prompt_class, integrity counters."""
    series: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    prompt_class: dict[str, str] = {}
    excluded: dict[str, list[str]] = defaultdict(list)
    flagged: dict[str, list[str]] = defaultdict(list)
    for rec in result["records"]:
        prompt_class[rec["prompt"]] = rec["class"]
        fl = _flagged(rec)
        if fl:
            flagged[rec["arm"]].append(f"{rec['prompt']}/p{rec['pass']}: {fl}")
        ok, why = _usable(rec)
        if not ok:
            excluded[rec["arm"]].append(f"{rec['prompt']}/p{rec['pass']}: {why}")
            continue
        v = rec.get(metric)
        if v is None:
            continue
        series[rec["arm"]][rec["prompt"]].append(float(v))
    return series, prompt_class, excluded, flagged


def _balanced(arm_series: dict[str, list[float]], baseline: dict[str, list[float]]):
    """Restrict both arms to the prompts they share, so the pairing is exact."""
    common = sorted(set(arm_series) & set(baseline))
    return ({k: arm_series[k] for k in common}, {k: baseline[k] for k in common})


def report(result: dict, baseline_map: dict[str, str] | None = None,
           metric: str = "decode_tok_s") -> None:
    series, prompt_class, excluded, flagged = build_series(result, metric)
    arms = list(result["arms"].keys())
    present = [a for a in arms if a in series]

    # default: every arm compared against the first arm (single-tree studies)
    # Prefer the map recorded by the run itself. Relying on the operator to remember
    # --baseline-map is how a dual-tree study silently compares a PR-branch arm against a
    # master-branch baseline.
    baseline_map = baseline_map or result.get("baseline_map") or {}
    _CO.require_complete(result)
    itt_series, _ = build_series_itt(result)
    n_pp = sum(len(v) for arm in series.values() for v in arm.values())
    n_itt = sum(len(v) for arm in itt_series.values() for v in arm.values())
    if n_pp != n_itt:
        print(f"\n[selection] the per-protocol series holds {n_pp} records and the "
              f"intention-to-treat series {n_itt}.")
        print(f"            {n_itt - n_pp} were excluded. Speculation moves where a request stops, "
              f"so excluding")
        print(f"            requests that stopped early selects on a post-treatment variable. Both "
              f"series are")
        print(f"            reported below where they differ.")
    else:
        print(f"\n[selection] per-protocol and intention-to-treat hold the same {n_pp} records; "
              f"the exclusion rule does no work here.")


    default_baseline = present[0] if present else None

    if not baseline_map:
        # Derive the map instead of falling back to "compare everything to the first arm".
        # A run spanning two llama.cpp trees would otherwise compare a PR-branch arm against a
        # master-branch baseline and fold the branch difference into the method effect. A
        # baseline arm is one that passes no extra server flags; each arm is matched to the
        # baseline built from its own tree.
        spec = result.get("arms", {})
        baselines_by_tree: dict[str, str] = {}
        for name, meta in spec.items():
            if not meta.get("extra_args") and not meta.get("expects_drafter"):
                baselines_by_tree.setdefault(meta.get("tree", "?"), name)
        if baselines_by_tree:
            derived = {}
            for name, meta in spec.items():
                b = baselines_by_tree.get(meta.get("tree", "?"))
                if b:
                    derived[name] = b
            if derived:
                baseline_map = derived
                print(f"\n(no baseline map in the result file; derived one from arm trees: "
                      f"{baselines_by_tree})")
        if not baseline_map and len({v.get("tree") for v in spec.values()}) > 1:
            print("\n!! WARNING: this run spans more than one llama.cpp tree and no baseline "
                  "map could be derived. Every arm is being compared to the first arm, which "
                  "mixes trees. Pass --baseline-map.\n")

    print(f"\n{'='*100}")
    print(f"metric: {metric}   passes: {result['design']['passes']}   "
          f"prompts: {result['design']['n_prompts']} "
          f"({result['design']['prompt_classes']})")
    print(f"design: interleaved={result['design']['interleaved']}  "
          f"cache_prompt={result['design'].get('cache_prompt')}  "
          f"fresh server per arm-pass={result['design']['fresh_server_per_arm_per_pass']}")
    print("="*100)

    print("\n--- per-arm absolute (class-stratified mean over prompts) ---")
    print(f"{'arm':28s} {'strat mean':>11s} {'raw mean':>10s} {'n obs':>7s}  worst-prompt CV")
    for a in present:
        per_class: dict[str, list[float]] = defaultdict(list)
        allv: list[float] = []
        for tag, vals in series[a].items():
            m = statistics.fmean(vals)
            per_class[prompt_class[tag]].append(m)
            allv.extend(vals)
        strat = ST.stratified_mean(per_class)
        cv = ST.within_prompt_cv(series[a])
        worst = max(cv.items(), key=lambda kv: kv[1]) if cv else ("-", 0.0)
        print(f"{a:28s} {strat:11.2f} {statistics.fmean(allv):10.2f} {len(allv):7d}  "
              f"{worst[0]}={worst[1]:.1f}%")

    # An arm that passes no server flags and expects no drafter IS a baseline. Phase M declares
    # two of them, one per target, and BASELINE_MAP covers only the speculative arms, so
    # `baseline-dense` fell through to `default_baseline` and appeared in the primary table as
    # "-71.59 % SLOWER" against `baseline-moe`. That number is true and is not a speculative
    # effect; printed in this table, in the same shape as every real row, it reads as one.
    spec_meta = result.get("arms", {})
    baseline_arms = {n for n, m in spec_meta.items()
                     if not m.get("extra_args") and not m.get("expects_drafter")}
    unmapped = [a for a in present
                if a not in baseline_arms and baseline_map and a not in baseline_map]
    if unmapped:
        print(f"\n!! WARNING: {', '.join(unmapped)} is not in the baseline map, so it is being "
              f"compared to {default_baseline}, which the matrix did not choose. Fix the "
              f"matrix's BASELINE_MAP.")
    if len(baseline_arms & set(present)) > 1:
        ordered = [a for a in present if a in baseline_arms]
        print("\n--- unspeculated contrast (baselines against each other; not an effect) ---")
        ref = ordered[0]
        for a in ordered[1:]:
            arm_s, base_s = _balanced(series[a], series[ref])
            if not arm_s:
                continue
            iv = ST.paired_cluster_bootstrap(base_s, arm_s, prompt_class, relative=True)
            # Name what actually differs. Phase A's two baselines are the same model on two
            # llama.cpp trees, Phase M's are two models on one tree, and calling either of those
            # "how much faster one target is" is wrong half the time.
            ma, mr = spec_meta.get(a, {}), spec_meta.get(ref, {})
            diff = [k for k in ("model", "tree", "gpu_state")
                    if (ma.get(k) or None) != (mr.get(k) or None)]
            what = " and ".join(diff) if diff else "nothing recorded in the arm metadata"
            print(f"  {a:24s} vs {ref:22s} {str(iv):>22s}   differs in: {what}")
        print("  Both sides are unspeculated, so this is a property of the setup, not an effect. "
              "A\n  contrast that spans zero is a control: whatever differs did not move baseline "
              "throughput.")

    near_zero_seen: list = []
    width_understated_seen: list = []
    print("\n--- PRIMARY: paired effect vs baseline (class-stratified, cluster bootstrap 95% CI) ---")
    print(f"{'arm':28s} {'vs':22s} {'delta %':>22s}  verdict")
    for a in present:
        b = baseline_map.get(a, default_baseline)
        if a == b or b not in series or a in baseline_arms:
            continue
        arm_s, base_s = _balanced(series[a], series[b])
        if not arm_s:
            continue
        # NOTE argument order: (baseline, arm). Passing (arm, baseline) silently inverts
        # the sign of every reported effect. harness/test_harness.py holds that case.
        iv = ST.paired_cluster_bootstrap(base_s, arm_s, prompt_class, relative=True)
        # The percentile bootstrap undercovers at 25 prompts, by 4 to 7 points depending on the
        # tail, and the error makes intervals too narrow. A verdict whose interval nearly touches
        # zero is the one that moves when the missing coverage is put back, so it is marked
        # rather than printed like the rest. stats.Interval.margin_half_widths carries the
        # measurement.
        verdict = ("no detected effect" if iv.spans_zero
                   else ("FASTER" if iv.point > 0 else "SLOWER"))
        # WIDTH FIRST, and it is a different complaint from being near zero.
        # `stats.Interval` says of singleton classes that "callers must say so
        # rather than print it as if it were estimated", and this caller -- the
        # primary one -- read `near_zero` and not `width_understated`. Under
        # `--prompts-per-class 1` every class holds one prompt, resampling one
        # item with replacement returns that item, and the interval collapses to
        # zero width: +55.00 [+55.00, +55.00]. `margin_half_widths` divides by a
        # half-width of zero and returns 0.0, so `near_zero` fired and the line
        # read "FASTER (margin 0.00 half-widths)" -- telling the reader the
        # effect might not clear zero when the truth is that the design could not
        # estimate precision at all. A wrong diagnosis, not a silence.
        if iv.width_understated:
            verdict += ("  WIDTH NOT ESTIMATED: single-prompt class(es) "
                        + ",".join(iv.singleton_classes)
                        + " contributed no variance, so the bounds are the point")
            width_understated_seen.append((a, iv))
        elif iv.near_zero:
            verdict += f"  (margin {iv.margin_half_widths:.2f} half-widths, see below)"
            near_zero_seen.append((a, iv))
        print(f"{a:28s} {b:22s} {str(iv):>22s}  {verdict}")

    if width_understated_seen:
        print("\n  WIDTH NOT ESTIMATED. A class holding one prompt contributes no")
        print("  variance to a cluster bootstrap: drawing one item with replacement")
        print("  always returns that item. The bounds below are the point estimate,")
        print("  not an interval, and no verdict about precision can be read off them.")
        print("  This is what --prompts-per-class does, and a dry run analysed like a")
        print("  result is why it is named here rather than left to the margin.")
        for a, iv in width_understated_seen:
            print(f"    {a:24s} {str(iv):>22s}   classes "
                  f"{','.join(iv.singleton_classes)}")

    if near_zero_seen:
        # This printed the 800-replication figures, and a t-interval coverage of 94.1 % that
        # NOTHING in this repository computes: coverage_sim.py has no t machinery at all. Both
        # came from a simulation whose code was never committed, and both were reaching seven
        # generated reports as if they were current results. The reproducible set replaced them;
        # the old ones stay, named as history, because the 1.3 half-width threshold was chosen
        # against them and a reader deserves to see what it was chosen against.
        print("\n  COVERAGE NOTE. The interval above is a percentile bootstrap, which is not")
        print("  second-order accurate and undercovers at this sample size. Measured in this")
        print("  repository at 2000 replications x 2000 resamples on 25 prompts, and printed by")
        print("  harness/coverage_sim.py into analysis/bootstrap_coverage.txt: 91.1 % normal,")
        print("  92.0 % uniform, 87.5 % heavy-tailed, 90.2 % binary, against a nominal 95 %,")
        print("  each carrying 0.6 to 0.7 points of Monte Carlo error. The 1.3 half-width")
        print("  threshold below was set against an earlier 800-replication run that reported")
        print("  90.9 / 90.6 / 88.0 and a t interval at 94.1 %; that run's code was never")
        print("  committed, so its t comparison is quoted history and not something reproducible")
        print("  here. The intervals are too narrow either way. These verdicts sit inside the")
        print("  margin where that matters and should not be leaned on:")
        for a, iv in near_zero_seen:
            print(f"    {a:24s} {str(iv):>22s}   margin {iv.margin_half_widths:.2f} half-widths")
        print("  Everything not listed clears zero by more than the correction is worth.")

    # The intention-to-treat table the [selection] note above promises. It was computing
    # itt_series and never printing it, so the note said "both series are reported below where
    # they differ" while only one was: a statement the tool made about itself that was false.
    if n_itt != n_pp:
        print("\n--- INTENTION TO TREAT: the same effect over every record, none excluded ---")
        print("  The per-protocol table above drops requests that stopped before the token cap.")
        print("  Speculation moves where a request stops, so that is an exclusion decided by an")
        print("  outcome. This table keeps them. Where the two agree, the exclusion rule did no")
        print("  work; where they differ, the per-protocol number is the one that needs the")
        print("  argument.")
        print(f"{'arm':28s} {'vs':22s} {'delta % (ITT)':>22s}  {'delta % (PP)':>22s}")
        for a in present:
            b = baseline_map.get(a, default_baseline)
            if a == b or b not in itt_series or a in baseline_arms:
                continue
            arm_i, base_i = _balanced(itt_series[a], itt_series[b])
            if not arm_i:
                continue
            iv_i = ST.paired_cluster_bootstrap(base_i, arm_i, prompt_class, relative=True)
            pp = ""
            if b in series and a in series:
                arm_p, base_p = _balanced(series[a], series[b])
                if arm_p:
                    pp = str(ST.paired_cluster_bootstrap(base_p, arm_p, prompt_class,
                                                         relative=True))
            print(f"{a:28s} {b:22s} {str(iv_i):>22s}  {pp:>22s}")

    print("\n--- SECONDARY (exploratory): per-class effect ---")
    for a in present:
        b = baseline_map.get(a, default_baseline)
        if a == b or b not in series:
            continue
        arm_s, base_s = _balanced(series[a], series[b])
        if not arm_s:
            continue
        cls_iv = ST.per_class_intervals(base_s, arm_s, prompt_class, relative=True)
        cells = "  ".join(f"{c}:{iv.point:+6.1f}%" for c, iv in sorted(cls_iv.items()))
        print(f"  {a:26s} {cells}")

    print("\n--- integrity ---")
    inc = result.get("incidents", [])
    print(f"  incidents recorded: {len(inc)}")
    for kind in sorted({i['kind'] for i in inc}):
        print(f"    {kind}: {sum(1 for i in inc if i['kind']==kind)}")
    total_recs = len(result["records"])
    tot_excl = sum(len(v) for v in excluded.values())
    print(f"  requests excluded from aggregates: {tot_excl} / {total_recs}")
    for a, reasons in sorted(excluded.items()):
        print(f"    {a}: {len(reasons)}")
        for r in reasons[:4]:
            print(f"        {r}")
    tot_flag = sum(len(v) for v in flagged.values())
    print(f"  requests flagged but KEPT (absolute-threshold concerns): {tot_flag}")
    for a, reasons in sorted(flagged.items()):
        print(f"    {a}: {len(reasons)}   e.g. {reasons[0] if reasons else ''}")
    if total_recs and tot_excl / total_recs > 0.25:
        print(f"  !! WARNING: {tot_excl/total_recs*100:.0f}% of records excluded. An exclusion "
              f"rule this aggressive is more likely to be wrong than the data is. Inspect "
              f"before trusting any effect below.")

    # ---------------------------------------------------------------- energy & acceptance
    print("\n--- energy and acceptance ---")
    print("  (tok/J is decode-only: prefill energy is measured separately and subtracted.")
    print("   No study in the prior-art sweep publishes an energy figure for this model.)")
    print(f"\n{'arm':22s} {'tok/J dec':>10s} {'vs base':>9s} {'J/request':>10s} {'vs base':>9s} "
          f"{'accept':>7s} {'draft len':>10s}")
    e_by: dict[str, list[float]] = defaultdict(list)
    j_by: dict[str, list[float]] = defaultdict(list)
    a_by: dict[str, list[float]] = defaultdict(list)
    l_by: dict[str, list[float]] = defaultdict(list)
    g_by: dict[str, list[float]] = defaultdict(list)
    c_by: dict[str, list[float]] = defaultdict(list)
    for rec in result["records"]:
        ok, _ = _usable(rec)
        if not ok:
            continue
        if rec.get("tok_per_joule_decode"):
            e_by[rec["arm"]].append(rec["tok_per_joule_decode"])
        if rec.get("decode_energy_j"):
            j_by[rec["arm"]].append(rec["decode_energy_j"])
        pw = rec.get("power") or {}
        if pw.get("energy_instant_vs_average_pct") is not None:
            g_by[rec["arm"]].append(pw["energy_instant_vs_average_pct"])
        if pw.get("sample_span_s") and rec.get("wall_ms"):
            c_by[rec["arm"]].append(pw["sample_span_s"] / (rec["wall_ms"] / 1000.0))
        tm = rec.get("timings") or {}
        dn, da = tm.get("t_draft_n") or 0, tm.get("t_draft_n_accepted") or 0
        if dn:
            a_by[rec["arm"]].append(da / dn)
            # Tokens emitted per target forward pass. The derivation, and why the first
            # generated token belongs to neither side of the ratio, is in speclen.py. It lives
            # there because this comment used to record that the correction had reached
            # cost_model.py and not this file.
            ml = speclen.mean_len(rec)
            if ml is not None:
                l_by[rec["arm"]].append(ml)
    # power.draw on Ampere is a rolling average of about a second. A steady load integrates the
    # same either way; a speculative one does not, because a drafter pass at width 1 and a wide
    # verification are different power states and the average smooths the peak. Measured on the
    # depth ladder the gap is 0.00 to 0.34 % for the baselines and 0.58 to 1.97 % for the
    # speculative arms, always positive, so the averaged field understates exactly the arms whose
    # energy is being compared against a baseline. Reported per arm, because a bias that is one
    # sided is not covered by any interval.
    # The integral runs from the first sample to the last, and the sampler queries nvidia-smi
    # before it waits, so its period is the query plus interval_s rather than interval_s. What
    # falls outside is energy the figure does not contain. Records written before sample_span_s
    # existed carry only a count and cannot report this.
    if c_by:
        cov = [v for vs in c_by.values() for v in vs]
        print(f"\n--- energy window coverage: {statistics.median(cov)*100:.1f} % of wall time "
              f"(min {min(cov)*100:.1f} %) ---")
        print("  Energy outside the first and last sample is not in the figure above.")

    if g_by:
        print("\n--- averaged against instantaneous power, per arm ---")
        for a in sorted(g_by):
            v = g_by[a]
            print(f"  {a:22s} {statistics.median(v):+6.2f} %   n={len(v)}")
        print("  Positive means power.draw integrated low. It leaves the baselines nearly alone")
        print("  and understates the speculative arms, so an energy saving read off the averaged")
        print("  field is larger than the instantaneous field supports, by about this much.")

    # Per arm, against ITS OWN baseline. A single reference taken from default_baseline was
    # what this did, and the throughput table three blocks up has always used baseline_map: on
    # Phase M, which runs a dense target and an MoE target in one matrix, that divided every
    # dense arm's energy by the MoE baseline's. 143 tok/s against 41, so the error is not
    # subtle in the throughput domain and it was not visible here because energy has no
    # familiar scale to check it against.
    def _ref(arm, table):
        b = baseline_map.get(arm, default_baseline)
        return (statistics.fmean(table[b]) if table.get(b) else None), b

    for a in present:
        e = statistics.fmean(e_by[a]) if e_by.get(a) else None
        j = statistics.fmean(j_by[a]) if j_by.get(a) else None
        ref_e, _rb = _ref(a, e_by)
        ref_j, _rb = _ref(a, j_by)
        de = f"{(e/ref_e-1)*100:+8.1f}%" if (e and ref_e) else "        -"
        dj = f"{(j/ref_j-1)*100:+8.1f}%" if (j and ref_j) else "        -"
        acc = f"{statistics.fmean(a_by[a]):7.3f}" if a_by.get(a) else "      -"
        ln = f"{statistics.fmean(l_by[a]):10.2f}" if l_by.get(a) else "         -"
        print(f"{a:22s} {(f'{e:.4f}' if e else '-'):>10s} {de} "
              f"{(f'{j:.0f}' if j else '-'):>10s} {dj} {acc} {ln}")

    if len({baseline_map.get(a, default_baseline) for a in present}) > 1:
        print("  Each arm is measured against its own baseline from baseline_map, not against a "
              "single\n  reference: this matrix holds more than one.")

    print("\n--- losslessness (greedy arms vs baseline text, same prompt+pass) ---")
    div: dict[str, list[dict]] = defaultdict(list)
    for rec in result["records"]:
        d = rec.get("divergence")
        if d:
            div[rec["arm"]].append(d)
    if not div:
        print("  (no divergence records)")
    for a, ds in sorted(div.items()):
        ident = sum(1 for d in ds if d["identical"])
        fr = [d["common_prefix_frac"] for d in ds if not d["identical"]]
        med = f"{statistics.median(fr):.3f}" if fr else "-"
        print(f"  {a:26s} no divergence through cap {ident}/{len(ds)}   "
              f"median shared prefix when forked: {med}")

    print("\n--- determinism (greedy, same arm/prompt across passes) ---")
    det: dict[str, list[bool]] = defaultdict(list)
    for rec in result["records"]:
        if "deterministic_vs_pass1" in rec:
            det[rec["arm"]].append(rec["deterministic_vs_pass1"])
    for a, flags in sorted(det.items()):
        print(f"  {a:26s} reproducible {sum(flags)}/{len(flags)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--metric", default="decode_tok_s")
    ap.add_argument("--baseline-map", default=None,
                    help='JSON: {"arm":"its baseline arm"} for dual-tree studies')
    a = ap.parse_args()
    bm = json.loads(a.baseline_map) if a.baseline_map else None
    report(load(Path(a.result)), bm, a.metric)


if __name__ == "__main__":
    main()
