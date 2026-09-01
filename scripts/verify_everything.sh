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

# --no-gpu runs sections 1 to 9, which need only Python and the committed files. It exists so
# that CI can run THIS script rather than a copy of some of it. The workflow used to reimplement
# section 9 inline, and the two had already drifted: the copy lacked the check that a manifest
# entry naming a deleted file is a failure, and it lacked the retry that tells a transient
# apart from a stale artifact. It also never ran sections 3, 4 or 5 at all -- broken links, the
# README's numbers against the result files, and the guard against a withdrawn claim coming
# back, which is the one that matters most in a repository that withdraws things.
NO_GPU=0
case "${1:-}" in
  --no-gpu) NO_GPU=1 ;;
  "") ;;
  *) echo "usage: $0 [--no-gpu]" >&2; exit 2 ;;
esac
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
# The audit excludes a partial write and a pre-repair copy, and one of those is committed, so a
# section headed "every committed result" audits all but one of them. The exclusion is right --
# a superseded copy kept for provenance is not a result this study stands on -- but naming it
# is what stops the header from being the only account of the scope.
git ls-files 'results/*.json' | grep -E '\.partial\.|\.pre_repair\.' \
  | sed 's/^/   not audited, superseded and kept for provenance: /' || true

hdr "3. documents link only at paths a clone would have"
# This was a second implementation of the harness's own link guard, and the weaker one. It read
# three documents -- README, TODO and PREREGISTRATION -- while twenty-four are tracked, and it
# resolved every link against the repository root, which is correct only for a document that
# sits there. The line it printed, "<N> tracked paths, 0 broken links", counted the universe
# links are checked AGAINST rather than what was checked, so it read as full coverage of a
# check that had opened three files; by this script's own opening rule that is a check
# reporting OK without saying what it examined. The harness guard, unified on
# tracked_markdown() and resolving each link relative to the document carrying it, found twelve
# broken links in the twenty-one documents this copy never opened, two of them <picture>
# sources. One
# implementation now, invoked by name so the section still says what it covers.
#
# The line that followed, `[ $? -ne 0 ] && bad ...`, could never fire: `|| bad` above it had
# already consumed the failure and returned 0.
python3 - <<'PY' || bad "a document links at a path a clone would not have"
import subprocess, sys
sys.path.insert(0, "harness")
from test_harness import tracked_markdown          # the one definition of "a document here"
r = subprocess.run([sys.executable, "harness/test_harness.py",
                    "TestEveryDocumentLinkPointsAtSomethingAClonWouldHave"],
                   capture_output=True, text=True, timeout=300)
print(f"   {len(tracked_markdown())} tracked documents, "
      f"{'every link resolves' if r.returncode == 0 else 'FAILURES below'}")
if r.returncode:
    for line in (r.stdout + r.stderr).splitlines()[-30:]:
        print("   " + line)
sys.exit(r.returncode)
PY

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
        # Search a WINDOW with its whitespace collapsed, not the single line. A withdrawn
        # wording is a phrase, and a phrase survives being hard-wrapped -- but `c in line`
        # does not: reflow "share the whole trajectory" across a line break and this scan
        # stops finding it and reports zero, which is the shape of a guard that examined
        # nothing. Nothing about the reflow would look like a change to a check. The window
        # is the same one the withdrawal marks above already use, so a claim split over two
        # lines is found and a withdrawal one line above it still exempts it.
        window = " ".join(" ".join(lines[i - 1:i + 1]).split())
        hits += [f"{f}:{i} {c!r}" for c in claims if c in window]
# What this actually checked, not what it would be nice to have checked. "0 residual claims"
# reads as "no withdrawn claim survives anywhere"; it means "none of these listed strings appear".
# The list is hand-maintained and lags every withdrawal until someone adds to it -- three of
# tonight's were missing from it until this commit.
print(f"   {len(files)} documents scanned for {len(claims)} withdrawn wordings, {len(hits)} found")
for h in hits:
    print("   FAIL:", h)
raise SystemExit(1 if hits else 0)
PY

hdr "6. every committed result is claimed by the registry, or named here as one it does not"
python3 - <<'PY' || bad "a result file or a registry entry is unaccounted for"
import glob, json, pathlib, re, subprocess, sys

# Every result file has to be claimed by an entry in evidence/registry.json, and every entry has
# to point at files that exist. The first version of this check matched result filenames against
# the bold labels of the README's later-phases table, which put a hole exactly where the primary
# result is: there was no `phase_a` prefix in the map, so Phase A, the extended-cap run and the
# host-B replication were all unchecked, and a re-run added later would have been invisible to the
# generated evidence block without anything noticing.
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
# The glob is `results/phase_*.json` less the skip patterns, so a dry run and a superseded copy
# fall outside it. Both are committed. A section headed "every" that quietly holds two files
# back is the shape this script exists to catch, so they are named rather than left to a count.
committed = subprocess.check_output(["git", "ls-files", "results/*.json"],
                                    text=True, timeout=60).split()
for x in sorted(set(committed) - set(on_disk)):
    print(f"   not a claim the registry makes, so not checked here: {x}")
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

hdr "9. every generated report and figure that can be rebuilt here, against its generator"
# Section 8 does this for one report. It had to be done for all of them: twenty-three committed
# artifacts were older than the analysers that produce them, including `phase_a_report.txt`, which
# still said "byte-identical" -- wording this study withdrew -- and `phase_c_cost.txt`, which
# published the k, c and k0 that cost_model.py now refuses to compute for that matrix. A generated
# file is a claim, and a claim nobody rechecks is the thing this script exists for.
python3 - <<'PY' || bad "a generated report is not what its analyser writes now"
import json, pathlib, subprocess, sys
TOOLS = {"_report.txt": "analyze", "_cost.txt": "cost_model", "_divergence.txt": "divergence_report",
         "_stability.txt": "pass_stability", "_width_groups.txt": "width_groups"}
SKIP = {"bootstrap_coverage.txt", "phase_m_anchor.txt"}   # section 8 covers the anchor


def resolve(name):
    for suf, tool in TOOLS.items():
        if name.endswith(suf):
            return tool, name[:-len(suf)]
    return "analyze", name[:-4]


# `no_source` used to be a counter and nothing else: an artifact whose generator the suffix rules
# could not guess was tallied and skipped, and the section still passed. Fifteen files sat in that
# bucket. phase_b_mechanism.txt was one, and it kept supplying the coefficients under a bolded
# causal claim in the README for a day after results/phase_b.json had been replaced by a clean
# re-measurement. analysis/MANIFEST.json now names the generator for each of them, established by
# running the command and requiring a byte-identical result, and anything in neither the manifest
# nor the suffix rules is a FAILURE.
manifest = json.loads(pathlib.Path("analysis/MANIFEST.json").read_text())
regen, external = manifest["regenerate"], manifest["external"]

# The loop below walks `analysis/*.txt` -- the artifacts that EXIST -- so its
# scope is whatever is on disk. Deleting a report removes it from the check
# silently, and a manifest entry naming a file that is gone is never noticed:
# the guard cannot fail in that direction at all. Both are checked here, before
# the loop, because a registry that describes something absent is a claim about
# a document that is not there.
missing = sorted(k for k in list(regen) + list(external)
                 if not pathlib.Path(k).exists())
if missing:
    print(f"   FAIL: {len(missing)} manifest entr(ies) name a file that does not "
          f"exist. Either the artifact was deleted and its entry is stale, or it "
          f"was renamed and the entry was not:")
    for m in missing:
        print(f"           {m}")
    sys.exit(1)

stale, checked, unmapped, flaky = [], 0, [], []
for art in sorted(pathlib.Path("analysis").glob("*.txt")):
    key = f"analysis/{art.name}"
    if art.name in SKIP or key in external:
        continue
    if key in regen:
        argv = [sys.executable] + regen[key]
        label = " ".join(regen[key][:1])
    else:
        tool, stem = resolve(art.name)
        if stem == "analysis_hostB":
            stem = "phase_a_hostB"
        res = pathlib.Path(f"results/{stem}.json")
        if not res.exists():
            unmapped.append(art.name)
            continue
        argv = [sys.executable, f"harness/{tool}.py", str(res)]
        label = f"{tool}.py"
    r = subprocess.run(argv, capture_output=True, text=True)
    checked += 1
    got, want = r.stdout + r.stderr, art.read_text()
    if got != want:
        # RUN IT AGAIN BEFORE CALLING IT A DIFFERENCE. On 2026-08-31 this failed on
        # phase_nmax_cost.txt and passed on the identical tree a minute later, while a
        # model was loading on the same machine. cost_model.py is deterministic -- every
        # seed in it is a literal, five reruns are byte-identical, and its output does not
        # move with PYTHONHASHSEED -- so the first run had failed rather than the artifact
        # drifted, and nothing in the output told those apart. A retry does: "differed,
        # then matched" is a transient, "differed twice" is a stale artifact, and the line
        # below says which instead of leaving it to whoever re-runs the gate.
        r2 = subprocess.run(argv, capture_output=True, text=True)
        got2 = r2.stdout + r2.stderr
        if got2 == want:
            flaky.append(f"{art.name} (from {label}) differed on the first run "
                         f"(exit {r.returncode}, {len(got)} bytes) and matched on the "
                         f"second: a transient, not a stale artifact")
            continue
        if got2 != got:
            stale.append(f"{art.name} (from {label}) is NOT REPRODUCIBLE: two runs of the "
                         f"same command gave {len(got)} and {len(got2)} bytes, neither "
                         f"matching the committed {len(want)}")
            continue
        # Say WHY. This reported only a filename, and on 2026-08-30 it flagged
        # phase_nmax_cost.txt once and passed on the identical tree a minute later.
        # cost_model.py is deterministic -- every seed in it is a literal and five
        # reruns are byte-identical -- so the run had failed rather than drifted,
        # and there was nothing in the output to tell those apart. A flake that
        # cannot be distinguished from a real difference gets waved away, and then
        # a real one does too.
        first = next((f"line {i}: committed {a!r} / computed {b!r}"
                      for i, (a, b) in enumerate(zip(want.splitlines(),
                                                     got.splitlines()), 1) if a != b),
                     f"identical for {min(len(want.splitlines()), len(got.splitlines()))} "
                     f"lines, then one ends: committed {len(want)} bytes, "
                     f"computed {len(got)}")
        stale.append(f"{art.name} (from {label}, exit {r.returncode}) {first}")
if flaky:
    print(f"   {len(flaky)} artifact(s) differed once and matched on retry, which is a")
    print(f"   transient and not a stale artifact. Named because a silent retry hides a")
    print(f"   machine that cannot be trusted to regenerate anything:")
    for x in flaky:
        print(f"           {x}")
if unmapped:
    print(f"   FAIL: {len(unmapped)} generated artifact(s) have no generator, in neither the suffix")
    print(f"         rules nor analysis/MANIFEST.json. Add them, with the argv that reproduces them:")
    for u in unmapped:
        print(f"           {u}")
    sys.exit(1)
print(f"   {checked} reports regenerated and compared, {len(external)} declared not rebuildable "
      f"here, {len(stale)} differ")
for x in stale:
    print("   FAIL:", x)
raise SystemExit(1 if stale else 0)
PY

# The figures too. Regenerating them is byte-reproducible here -- matplotlib 3.11.1, same data,
# same bytes -- which is what makes this checkable at all. It needs the venv, because matplotlib is
# not in the system interpreter, and it says so rather than skipping quietly when that is missing.
# The venv if there is one, otherwise whatever python has matplotlib. Hard-coding
# .venv/bin/python meant this section could only ever pass on the machine that has one, and CI
# installs matplotlib into the system interpreter -- so pointing the workflow at this script
# would have turned "the figures were NOT checked" into a red build on every push.
PYFIG=""
for cand in .venv/bin/python python3; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import matplotlib' 2>/dev/null; then
    PYFIG="$cand"; break
  fi
done
# Three things this section got wrong, all at once, and all invisible from its output.
#
# `plot_qsmall_ladder.py` takes its result files on the command line; the other two do not. It
# was invoked with none, printed its usage, exited 2, and wrote nothing. The exit code went
# nowhere -- `>/dev/null 2>&1` with no `||` -- and `git status` was then compared before and
# after, which is clean precisely BECAUSE nothing was drawn. So the section announced "16
# figures regenerated, none changed" while regenerating 14 and checking 14, and the two it
# skipped were stale: `plot_qsmall_ladder_dark.png` still carried the pre-`_WONG_DARK` blue at
# 3.65:1 on the dark background, six commits after that was fixed everywhere it was checked.
# The count 16 was a literal in the message, so it could not disagree with what happened.
#
# Hence: arguments, a checked exit status, and a count of the files actually written this run
# against the number on disk. A generator that draws nothing now fails instead of passing.
if [ -n "$PYFIG" ]; then
  before=$(git status --porcelain analysis/*.png)
  marker=$(mktemp)
  run_fig() {
    if ! "$PYFIG" "$@" >/dev/null 2>&1; then
      bad "figure generator exited non-zero: $*"
    fi
  }
  run_fig analysis/plot.py
  run_fig analysis/plot_phase_m.py
  run_fig analysis/plot_qsmall_ladder.py results/phase_qsmall_*.json
  on_disk=$(find analysis -maxdepth 1 -name '*.png' | wc -l)
  written=$(find analysis -maxdepth 1 -name '*.png' -newer "$marker" | wc -l)
  # Collected while the marker still exists: `find ! -newer` on a deleted file errors and
  # prints nothing, which would have made this branch report a failure with no list under it.
  skipped=$(find analysis -maxdepth 1 -name '*.png' ! -newer "$marker")
  rm -f "$marker"
  after=$(git status --porcelain analysis/*.png)
  if [ "$written" -ne "$on_disk" ]; then
    bad "$written of $on_disk figures were written this run; the rest were not checked"
    printf '%s\n' "$skipped" | sed 's/^/     not drawn: /'
  elif [ "$before" = "$after" ]; then
    printf '   %s figures regenerated, none changed\n' "$on_disk"
  else
    bad "a figure is not what its plot script draws now"
    printf '%s\n' "$after" | sed 's/^/     /'
  fi
else
  bad "no interpreter with matplotlib on PATH or in .venv: the figures were NOT checked"
fi

if [ "$NO_GPU" = 1 ]; then
  hdr "10. the GPU is where the runs left it"
  echo "   SKIPPED: --no-gpu. This section reads the card and cannot run without one."
  echo
  if [ "$FAIL" = 0 ]; then echo "Sections 1 to 9 passed."; else echo "SOMETHING FAILED above."; fi
  exit "$FAIL"
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
print(f"   fan {now.get('fan_control')}, {now.get('fan_count')} fans, "
      f"target {now.get('fan_targets_pct')} current {now.get('fan_current_pct')}")
drift = []
if abs(now["power_limit_w"] - stock.power_limit_w) > 0.5:
    drift.append("power_limit_w")
for k in ("mem_transfer_offset", "core_offset"):
    if now[k] != getattr(stock, k):
        drift.append(k)
# Cooling belongs here for the same reason the power limit does: a card left under manual fan
# control runs the next study at a different sustained clock, and until 2026-08-29 nothing in
# this repository would have said so. "unknown" counts as drift -- a host that cannot answer the
# question has not answered it no.
if now.get("fan_control") != "auto":
    drift.append(f"fan_control={now.get('fan_control')!r}")
if drift:
    print("   FAIL: the card is not at the state the runs assume:", drift)
    sys.exit(1)
print("   at stock")
GPUCHK
[ $? -ne 0 ] && bad "the card is not at stock"

printf '\n'
if [ $FAIL -eq 0 ]; then echo "All sections passed."; else echo "At least one section failed."; fi
exit $FAIL
