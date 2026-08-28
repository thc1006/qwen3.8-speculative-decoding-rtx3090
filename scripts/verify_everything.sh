#!/usr/bin/env bash
# One command that re-checks every claim this repository makes about itself.
#
# Written because "it is done" is not evidence. Each section below is a check that has caught a
# real defect in this study, and each prints what it looked at rather than only whether it liked
# it -- a check that reports "OK" without saying what it examined is the same shape as a check
# that examined nothing.
#
# CPU-heavy. Do not run while .gpu-in-use.lock exists: telemetry.host_load samples once per
# arm-pass and a running phase will record this as contention.
set -uo pipefail
cd "$(dirname "$0")/.."
FAIL=0
hdr() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
bad() { FAIL=1; printf '   FAIL: %s\n' "$*"; }

if [ -f .gpu-in-use.lock ]; then
  echo "REFUSING: .gpu-in-use.lock exists, so a measurement is running."
  cat .gpu-in-use.lock
  exit 2
fi

hdr "1. the harness's own tests"
# The verdict line, not the last three lines of output. unittest writes "Ran N tests" and
# "OK"/"FAILED" to stderr while tests print to stdout, so `| tail -3` showed whatever a fixture
# happened to print last and never showed whether anything passed. A check that does not report
# its own verdict is the shape of defect this script exists to find.
if python3 harness/test_harness.py > /tmp/verify_tests.log 2>&1; then
  grep -E '^(Ran |OK|FAILED)' /tmp/verify_tests.log | sed 's/^/   /'
else
  grep -E '^(Ran |OK|FAILED|FAIL:|ERROR:)' /tmp/verify_tests.log | sed 's/^/   /'
  bad "test suite; full output in /tmp/verify_tests.log"
fi

hdr "2. every committed result, audited"
python3 harness/audit_results.py || bad "audit_results reported a problem"

hdr "3. documents link only at paths a clone would have"
python3 - <<'PY' || bad "a document links at a path a clone would not have"
import re, pathlib, subprocess, sys
tracked = set(subprocess.check_output(["git", "ls-files"], text=True).split("\n")) - {""}
dirs = {"/".join(t.split("/")[:i]) for t in tracked for i in range(1, len(t.split("/")))}
bad = []
for name in ("README.md", "TODO.md", "PREREGISTRATION.md"):
    p = pathlib.Path(name)
    if not p.exists():
        continue
    text = p.read_text(encoding="utf-8")
    targets = [t for _, t in re.findall(r"\[([^\]]+)\]\(([^)#][^)]*)\)", text)]
    targets += re.findall(r'src="([^"]+)"', text) + re.findall(r'srcset="([^"]+)"', text)
    for t in targets:
        if t.startswith(("http://", "https://", "mailto:")):
            continue
        path = t.split("#")[0].rstrip("/")
        if path and path not in tracked and path not in dirs:
            bad.append(f"{name} -> {path}")
print(f"   {len(tracked)} tracked paths, {len(bad)} broken links")
for b in bad:
    print("   FAIL:", b)
sys.exit(1 if bad else 0)
PY
[ $? -ne 0 ] && bad "a document links at an untracked path"

hdr "4. the README's numbers against the result files"
python3 - <<'PY' || bad "a README number does not match the result files"
import json, collections, statistics, sys
sys.path.insert(0, "harness")
import stats as ST
d = json.load(open("results/phase_a.json"))
cls = {r["prompt"]: r["class"] for r in d["records"]}
per = collections.defaultdict(lambda: collections.defaultdict(list))
for r in d["records"]:
    per[r["arm"]][r["prompt"]].append(r["decode_tok_s"])
def strat(a):
    byc = collections.defaultdict(list)
    for p, v in per[a].items():
        byc[cls[p]].append(statistics.fmean(v))
    return ST.stratified_mean(byc)
# Each arm against the baseline built from its own tree. Every arm was compared against
# `baseline@master`, including the two DFlash2 arms that run on PR #27342, which charges the
# branch's own no-speculation difference to the method. The two baselines agree to 0.008 % here,
# so it moved the DFlash2 figures by 0.01 points and the mistake sat behind a coincidence of this
# run rather than anything the design guarantees.
meta = d.get("arms", {})
base_of = {(m or {}).get("tree"): a for a, m in meta.items() if not (m or {}).get("extra_args")}
readme = open("README.md").read()
problems = []
base = strat(base_of.get("master", "baseline@master"))
quoted = "41.55" in readme
if not quoted:
    problems.append("the baseline tok/s figure is no longer quoted in the README")
print(f"   baseline {base:.2f} tok/s, quoted in README: {quoted}")
for arm, pct in (("mtp-n2", 59.8), ("mtp-n3", 52.3), ("dflash2-n4", 51.9),
                 ("mtp-n5", 32.1), ("dflash2-n7", 22.6)):
    ref = base_of.get((meta.get(arm) or {}).get("tree"))
    if ref is None:
        problems.append(f"{arm}: no baseline in its own tree")
        continue
    got = (strat(arm) / strat(ref) - 1) * 100
    ok = abs(got - pct) < 0.1 and f"+{pct} %" in readme
    if not ok:
        problems.append(f"{arm}: recomputed {got:+.2f} % against {ref}, README says +{pct} %")
    print(f"   {arm:12s} vs {ref:16s} recomputed {got:+.2f} %, README says +{pct} %  "
          f"{'ok' if ok else 'MISMATCH'}")
raise SystemExit(1 if problems else 0)
PY

hdr "5. no committed document carries a withdrawn claim"
python3 - <<'PY' || bad "a committed document carries a withdrawn claim"
import subprocess, pathlib
claims = ("not the architecture", "sign belongs to the drafting method",
          "rules out a large architecture effect", "flag was accepted and did nothing",
          "no quantization anywhere", "the cost is linear", "costs a fixed c",
          # Withdrawn by TODO B7: the window is fixed in tokens, so there is no censored subset
          # to check the rest against and no partition surviving on the rest.
          "The partition survives on the 10", "15 of 25 prompts",
          # Withdrawn by Correction 38. Agreement inside the window is not identity while anything
          # stopped on the cap: 39 of the 75 cross-tree comparisons are right-censored, and at 400
          # tokens every one of the 125 was. The bare phrase `byte-identical` is NOT listed --
          # control_determinism.txt and the warp_intervention reports use it correctly, for two
          # runs of the same input -- so the withdrawn readings are listed by their own wording.
          "share the whole trajectory", "byte-identical output on",
          "reproduces master's bytes")
marks = ("withdraw", "Withdraw", "used to read", "an earlier version", "An earlier version",
         "no longer", "retract", "superseded", "Superseded")
files = [f for f in subprocess.check_output(["git", "ls-files", "*.md"], text=True).split("\n")
         if f and "PREREGISTRATION" not in f and not f.startswith(("upstream/", "llamacpp"))]
hits = []
for f in files:
    lines = pathlib.Path(f).read_text(errors="replace").splitlines()
    for i, line in enumerate(lines, 1):
        # A withdrawal runs to a sentence, not to a line: "An earlier version of this paragraph"
        # sat one line above the claim it was withdrawing, and a line-local exemption missed it
        # and would have reported the withdrawal itself as a residual claim.
        ctx = " ".join(lines[max(0, i - 3):i + 2])
        if any(m in ctx for m in marks):
            continue
        hits += [f"{f}:{i} {c!r}" for c in claims if c in line]
# What this actually checked, not what it would be nice to have checked. "0 residual claims"
# reads as "no withdrawn claim survives anywhere"; it means "none of these listed strings appear".
# The list is hand-maintained and lags every withdrawal until someone adds to it -- three of
# tonight's were missing from it until this commit.
print(f"   {len(files)} documents scanned for {len(claims)} withdrawn wordings, {len(hits)} found")
for h in hits:
    print("   FAIL:", h)
raise SystemExit(1 if hits else 0)
PY

hdr "6. every result file is claimed by the evidence registry"
python3 - <<'PY' || bad "a result file or a registry entry is unaccounted for"
import glob, json, pathlib, re, sys

# Every result file has to be claimed by an entry in evidence/registry.json, and every entry has
# to point at files that exist. The first version of this check matched result filenames against
# the bold labels of the README's later-phases table, which put a hole exactly where the primary
# result is: there was no `phase_a` prefix in the map, so Phase A, the extended-cap run and the
# host-B replication were all unchecked, and `phase_a_cap1600.rerun.json` would have been invisible
# to the generated evidence block without anything noticing.
reg = json.loads(pathlib.Path("evidence/registry.json").read_text())
skip = reg.get("skip_patterns") or []
readme = pathlib.Path("README.md").read_text()
problems = []

claimed = {}
for phase in reg["phases"]:
    hits = []
    for pat in phase["results"]:
        hits += [p for p in glob.glob(pat) if not any(s in pathlib.Path(p).name for s in skip)]
    if not hits:
        problems.append(f"registry entry {phase['id']} matches no result file: {phase['results']}")
    for h in hits:
        claimed.setdefault(h, []).append(phase["id"])

on_disk = [p for p in sorted(glob.glob("results/phase_*.json"))
           if not any(s in pathlib.Path(p).name for s in skip)]
for p in on_disk:
    if p not in claimed:
        problems.append(f"{p} is not claimed by any registry entry")
for p, ids in claimed.items():
    if len(ids) > 1:
        problems.append(f"{p} is claimed by {ids}; its records would be counted twice")

# Nothing described as running may have a finished result file. Phase B sat under a "**Running:**"
# line for a day after its 525 records were committed.
for m in re.finditer(r"\*\*Running:\*\*\s*Phase\s+([A-Za-z0-9-]+)", readme):
    letter = m.group(1)
    for phase in reg["phases"]:
        if phase["id"].lower() != letter.lower():
            continue
        for pat in phase["results"]:
            for p in glob.glob(pat):
                n = len(json.loads(pathlib.Path(p).read_text()).get("records") or [])
                if n:
                    problems.append(f"README calls Phase {letter} running; {p} holds {n} records")

print(f"   {len(on_disk)} result files, {len(reg['phases'])} registry entries, "
      f"{len(problems)} mismatches")
for x in problems:
    print("   FAIL:", x)
raise SystemExit(1 if problems else 0)
PY

hdr "7. the README's evidence block is what the result files say"
# The block it checks is generated by harness/render_evidence.py from evidence/registry.json plus
# the result files themselves. Until the block exists in README.md this fails, and that is the
# correct state to be in: it means the status paragraph is still hand-maintained prose, which is
# how Phase B stayed "Running" for a day and Phase R never got a row at all.
python3 harness/render_evidence.py --check || bad "the README's evidence block has drifted from the result files"

hdr "8. generated reports still match the analyser that writes them"
# `analysis/phase_m_anchor.txt` reported -72.3 % for days, the pooled median, while the README
# and the analyser both reported -65.6 % -- and the analyser states in the same breath that only
# the class-stratified primary may be compared against the registered band. Nothing could see
# that the committed file was older than the code. The reports carry the sha256 of their inputs
# and no timestamp, so regenerating from the same result has to produce the same bytes.
# A non-zero exit is this analyser's verdict, not an error: it gates on the anchor holding, and
# the anchor does not hold. Only an empty output means it failed to run.
out=$(python3 harness/anchor_verdict.py results/phase_m.json 2>&1)
if [ -z "$out" ]; then
  bad "anchor_verdict.py produced nothing"
elif ! diff -q <(printf '%s\n' "$out") analysis/phase_m_anchor.txt >/dev/null; then
  bad "analysis/phase_m_anchor.txt is not what anchor_verdict.py writes now"
  diff <(printf '%s\n' "$out") analysis/phase_m_anchor.txt | head -6 | sed 's/^/     /'
else
  printf '   phase_m_anchor.txt regenerates byte-identical\n'
fi

hdr "9. every generated report and figure is what its generator writes now"
# Section 8 does this for one report. It had to be done for all of them: twenty-three committed
# artifacts were older than the analysers that produce them, including `phase_a_report.txt`, which
# still said "byte-identical" -- wording this study withdrew -- and `phase_c_cost.txt`, which
# published the k, c and k0 that cost_model.py now refuses to compute for that matrix. A generated
# file is a claim, and a claim nobody rechecks is the thing this script exists for.
python3 - <<'PY' || bad "a generated report is not what its analyser writes now"
import pathlib, subprocess, sys
TOOLS = {"_report.txt": "analyze", "_cost.txt": "cost_model", "_divergence.txt": "divergence_report",
         "_stability.txt": "pass_stability", "_width_groups.txt": "width_groups"}
SKIP = {"bootstrap_coverage.txt", "phase_m_anchor.txt"}   # section 8 covers the anchor


def resolve(name):
    for suf, tool in TOOLS.items():
        if name.endswith(suf):
            return tool, name[:-len(suf)]
    return "analyze", name[:-4]


stale, checked, no_source = [], 0, 0
for art in sorted(pathlib.Path("analysis").glob("*.txt")):
    if art.name in SKIP:
        continue
    tool, stem = resolve(art.name)
    if stem == "analysis_hostB":
        stem = "phase_a_hostB"
    res = pathlib.Path(f"results/{stem}.json")
    if not res.exists():
        no_source += 1
        continue
    r = subprocess.run([sys.executable, f"harness/{tool}.py", str(res)],
                       capture_output=True, text=True)
    checked += 1
    if r.stdout + r.stderr != art.read_text():
        stale.append(f"{art.name} (from {tool}.py)")
print(f"   {checked} reports regenerated and compared, {no_source} have no single result file "
      f"to regenerate from, {len(stale)} differ")
for x in stale:
    print("   FAIL:", x)
raise SystemExit(1 if stale else 0)
PY

# The figures too. Regenerating them is byte-reproducible here -- matplotlib 3.11.1, same data,
# same bytes -- which is what makes this checkable at all. It needs the venv, because matplotlib is
# not in the system interpreter, and it says so rather than skipping quietly when that is missing.
if [ -x .venv/bin/python ] && .venv/bin/python -c 'import matplotlib' 2>/dev/null; then
  before=$(git status --porcelain analysis/*.png)
  .venv/bin/python analysis/plot.py >/dev/null 2>&1
  .venv/bin/python analysis/plot_phase_m.py >/dev/null 2>&1
  .venv/bin/python analysis/plot_qsmall_ladder.py >/dev/null 2>&1
  after=$(git status --porcelain analysis/*.png)
  if [ "$before" = "$after" ]; then
    printf '   16 figures regenerated, none changed\n'
  else
    bad "a figure is not what its plot script draws now"
    printf '%s\n' "$after" | sed 's/^/     /'
  fi
else
  bad "no .venv with matplotlib: the figures were NOT checked"
fi

hdr "10. the GPU is where the runs left it"
# Through the module's own API. `python3 harness/gpustate.py --check` was here, and gpustate.py
# has no __main__ and no argparse: it imported, did nothing, exited 0, and the fallback after the
# || never ran. A section that reported nothing and passed.
python3 - <<'GPUCHK'
import sys
sys.path.insert(0, "harness")
import gpustate as G
now, stock = G.read_state(0), G.stock_for(0)
print(f"   power limit {now['power_limit_w']:.0f} W, stock {stock.power_limit_w} W")
print(f"   offsets mem {now['mem_transfer_offset']} core {now['core_offset']}, "
      f"stock {stock.mem_transfer_offset} / {stock.core_offset}")
drift = []
if abs(now["power_limit_w"] - stock.power_limit_w) > 0.5:
    drift.append("power_limit_w")
for k in ("mem_transfer_offset", "core_offset"):
    if now[k] != getattr(stock, k):
        drift.append(k)
if drift:
    print("   FAIL: the card is not at the state the runs assume:", drift)
    sys.exit(1)
print("   at stock")
GPUCHK
[ $? -ne 0 ] && bad "the card is not at stock"

printf '\n'
if [ $FAIL -eq 0 ]; then echo "All sections passed."; else echo "At least one section failed."; fi
exit $FAIL
