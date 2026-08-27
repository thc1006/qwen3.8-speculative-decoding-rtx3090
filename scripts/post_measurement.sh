#!/usr/bin/env bash
# Everything that had to wait for the GPU, in the order it has to happen.
#
# Written during the Phase A re-run at a 1600-token cap, when the CPU guard was correctly refusing
# every analyser in the repository. Running the work by hand out of a checklist is how a step gets
# skipped, so it is a script.
#
# Two parts, and the split matters. Part A decides whether the re-run is usable and STOPS if it is
# not: there is no point rewriting the README's evidence block while the measurement underneath it
# is broken. Part B is deferred maintenance that has nothing to do with the re-run, and it can be
# forced past a Part A failure with --maintenance-anyway when that is genuinely what you want.
#
# An earlier version of this header said "it stops at the first thing that fails". It did not:
# there was no `set -e`, every step ended in `|| bad`, and the script ran to the end regardless.
# The sentence was a claim the script did not satisfy, which is the defect this repository exists
# to hunt, sitting in the script that was supposed to close the audit.
set -uo pipefail
cd "$(dirname "$0")/.."
FAIL=0
hdr() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
bad() { FAIL=1; printf '   FAIL: %s\n' "$*"; }
note() { printf '   %s\n' "$*"; }

# Temporaries live BESIDE their target, not in /tmp. `mv` across filesystems is a copy followed by
# an unlink, which is exactly the non-atomic write this pattern exists to avoid; a rename within
# one directory is atomic. The trap covers every exit path, including the early ones.
TMPFILES=()
cleanup() { [ "${#TMPFILES[@]}" -gt 0 ] && rm -f "${TMPFILES[@]}"; return 0; }
trap cleanup EXIT
# Assigns into the variable named by $1 rather than echoing, because `V=$(mktmp path)` runs the
# function in a command substitution -- a subshell -- so `TMPFILES+=(...)` appended to a copy that
# died with it and the parent array stayed empty. The trap was correct and swept nothing; the run
# that proved it left two .err files behind in analysis/. Static review did not find this. Running
# it did.
mktmp() { local __v=$1 __t; __t=$(mktemp "$2.XXXXXX"); TMPFILES+=("$__t"); printf -v "$__v" '%s' "$__t"; }

RERUN=results/phase_a_cap1600.rerun.json
COMMITTED=results/phase_a_cap1600.json
MAINTENANCE_ANYWAY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --maintenance-anyway) MAINTENANCE_ANYWAY=1 ;;
    # An unrecognised flag used to be ignored in silence, so a typo in --maintenance-anyway would
    # have stopped at Part A while looking like it had been honoured.
    *) echo "unknown argument: $1"; echo "usage: $0 [--maintenance-anyway]"; exit 2 ;;
  esac
  shift
done

# The audit fails on these two and that is known and accepted, so a bare "verify_everything
# failed" carries no information. Naming them here turns the check into a real one: a file that
# leaves this set, or joins it, is news.
EXPECTED_AUDIT_FAILURES="phase_a_cap1600 phase_b"

if [ -f .gpu-in-use.lock ]; then
  echo "REFUSING: .gpu-in-use.lock still exists. This script is the thing that runs after."
  cat .gpu-in-use.lock
  exit 2
fi
# Reuse the guard's own detector rather than `pgrep -f 'harness/bench\.py'`, which matches any
# command line that merely contains the path -- `vim harness/bench.py` would have stopped this
# script. The hook compares basenames at argument positions in the real NUL-separated argv.
if python3 - <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("g", ".claude/hooks/no_cpu_during_measurement.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
hit = g.bench_process()
if hit:
    print(f"pid {hit[0]}: {hit[1]}")
sys.exit(0 if hit else 1)
PY
then
  echo "REFUSING: a benchmark is running with no lock file. Check it before running analysers."
  exit 2
fi

# ------------------------------------------------------------------ PART A: is the re-run usable
hdr "A1. did the re-run finish"
[ -f "$RERUN" ] || { echo "   $RERUN is not there; nothing to do"; exit 2; }
python3 - "$RERUN" <<'PY' || bad "the re-run did not finish cleanly"
import json, sys
d = json.load(open(sys.argv[1]))
n = len(d.get("records") or [])
inc = d.get("incidents") or []
closed = "gpu_state_at_end" in d
print(f"   {n} records, {len(inc)} incidents, closing fields present: {closed}")
for i in inc:
    print(f"     {i.get('kind')} at {i.get('arm')} pass {i.get('pass')}: {str(i.get('detail'))[:70]}")
# A file that exists is not a file that finished. bench.py writes after every arm, so a killed run
# leaves a well-formed JSON with fewer records and no closing fields; the previous attempt left
# exactly that at 150 of 525.
ok = True
# Derived, not hardcoded. `525` was written in by hand; if the matrix ever gains an arm or a pass,
# a hardcoded total silently checks the wrong number, which is the failure mode this whole script
# exists to catch elsewhere.
des = d.get("design") or {}
want = (des.get("n_prompts") or 0) * (des.get("passes") or 0) * len(d.get("arms") or {})
if not want:
    print("   cannot derive the expected record count from the design", file=sys.stderr); ok = False
elif n != want:
    print(f"   expected {want} records "
          f"({des['n_prompts']} prompts x {des['passes']} passes x {len(d['arms'])} arms)",
          file=sys.stderr); ok = False
if not closed:
    print("   no gpu_state_at_end: the run did not reach its own shutdown", file=sys.stderr); ok = False
if inc:
    print(f"   {len(inc)} incident(s): the point of this re-run was to have none", file=sys.stderr)
    ok = False
raise SystemExit(0 if ok else 1)
PY
python3 harness/audit_results.py "$RERUN" || bad "the re-run does not audit clean"

# Stop here rather than at the end of Part A. A2 compares an unfinished run against a finished one
# and A3 reports that the cross-tree control is "absent" -- which, on a run that stopped inside
# pass 1, means the pass never ended, not that the fix failed. Diagnosing a truncated file produces
# confident statements about nothing.
if [ "$FAIL" != 0 ]; then
  echo
  echo "The re-run did not finish. Nothing downstream is interpretable, so nothing downstream runs."
  echo "Archive it as results/*.partial.<epoch>.json and start again, or investigate the log."
  exit 1
fi

hdr "A2. the re-run against the file it supersedes"
# Same matrix, same cap, same host: the one comparison in this repository where the two runs really
# are the same experiment, so a disagreement is about the machine and not the design.
# --allow-incidents is for the COMMITTED file's two, which are why the re-run exists. It also
# suppresses the re-run's, which is why A1 above checks those separately and hard, rather than
# leaving the coverage to depend on the order of two steps.
python3 harness/compare_reproduction.py "$COMMITTED" "$RERUN" --allow-incidents \
  || bad "the re-run and the committed file do not agree"

hdr "A3. did the cross-tree control come back"
python3 - "$RERUN" <<'PY' || bad "tree_divergence is still absent"
import collections, json, sys
d = json.load(open(sys.argv[1]))
tv = [r for r in d["records"] if r.get("tree_divergence")]
ident = sum(1 for r in tv if r["tree_divergence"].get("identical"))
by = collections.Counter(r["arm"] for r in tv)
print(f"   {len(tv)} records carry tree_divergence, {ident} identical, arms {dict(by)}")
if not tv:
    print("   the bench.py fix from Correction 34 did not take effect", file=sys.stderr)
    raise SystemExit(1)
if ident != len(tv):
    # This exits non-zero on purpose, and the message says why. It is not a malfunction: it is the
    # most interesting outcome this control can have. The first version printed it to stderr and
    # returned 0, so a branch that does NOT reproduce master's bytes with speculation off would
    # have been reported under "All post-measurement steps passed."
    print(f"   FINDING, not a failure of this script: {len(tv) - ident} of {len(tv)} cross-tree "
          f"comparisons are NOT identical. The PR branch does not reproduce master's bytes with "
          f"speculation off. Read them before anything else in this run is quoted.",
          file=sys.stderr)
    raise SystemExit(1)
PY

if [ "$FAIL" != 0 ] && [ "$MAINTENANCE_ANYWAY" = 0 ]; then
  echo
  echo "Part A failed. Stopping before Part B, which rewrites committed artifacts and the README:"
  echo "there is no point regenerating the paperwork while the measurement under it is in doubt."
  echo "Re-run with --maintenance-anyway if the deferred maintenance is what you actually want."
  exit 1
fi

# ------------------------------------------------- PART B: maintenance that waited for a free GPU
hdr "B1. coverage, at a replication count that can tell the figures apart (TODO D6)"
# Measured, not guessed: one paired_cluster_bootstrap at n_boot=2000 over 25 prompts takes 9.2 ms
# here, and the sweep is 8 rows, so 2000 replications is about 2.5 minutes plus whatever the n=50
# rows add. The comment this replaces said "roughly 6.7x the work of the committed run", which was
# a ratio nobody had turned into a duration.
mktmp COV analysis/.bootstrap_coverage; mktmp COV_ERR analysis/.bootstrap_coverage_err
if python3 harness/coverage_sim.py --replications 2000 > "$COV" 2>"$COV_ERR"; then
  # Write only after it succeeds. `cmd > committed_file` truncates the file before the command
  # runs, so a crash left the repository with an empty artifact and no way to tell.
  mv "$COV" analysis/bootstrap_coverage.txt
  head -16 analysis/bootstrap_coverage.txt | sed 's/^/   /'
else
  bad "coverage_sim did not finish; analysis/bootstrap_coverage.txt left untouched"
  head -5 "$COV_ERR" | sed 's/^/     /'
fi

hdr "B2. the README's evidence block, generated (TODO D7)"
python3 harness/render_evidence.py || bad "render_evidence did not write the block"

hdr "B3. the anchor report, regenerated from the current analyser"
# A non-zero exit is this analyser's verdict, not an error: it gates on the anchor holding, and the
# anchor does not hold. stderr is kept OUT of the artifact -- merging it meant a traceback would
# have been written into the committed report and only the grep below would have noticed.
mktmp ANC analysis/.phase_m_anchor; mktmp ANC_ERR analysis/.phase_m_anchor_err
python3 harness/anchor_verdict.py results/phase_m.json > "$ANC" 2>"$ANC_ERR"
# `grep -q PRIMARY` alone accepts a truncated file that happens to reach that line. The report has
# a fixed shape: a provenance header naming its two inputs, the registered band, the primary
# estimate and a verdict. Check for all of them, or a half-written report replaces a whole one.
if grep -q "generated from" "$ANC" && grep -q "analyser " "$ANC" \
   && grep -q "registered band" "$ANC" && grep -q "PRIMARY" "$ANC" \
   && grep -qE "ANCHOR (DOES NOT HOLD|HOLDS)" "$ANC"; then
  mv "$ANC" analysis/phase_m_anchor.txt
  note "anchor report regenerated, $(wc -l < analysis/phase_m_anchor.txt) lines"
else
  bad "the anchor report is not the expected shape; analysis/phase_m_anchor.txt left untouched"
  note "got $(wc -l < "$ANC") lines; first stderr:"
  head -5 "$ANC_ERR" | sed 's/^/     /'
fi

hdr "B4. the audit's failures are the ones already accounted for"
python3 - <<PY || bad "the set of audit failures is not the expected one"
import re, subprocess, sys
expected = set("$EXPECTED_AUDIT_FAILURES".split())
out = subprocess.run([sys.executable, "harness/audit_results.py"], capture_output=True, text=True).stdout
# Table rows only. `^(\S+)\s+.*FAIL` also matched the summary line "34 of 36 clean; 2 with at
# least one FAIL" and put "34" in the set, which would have made this check fail forever for a
# reason that has nothing to do with any file.
got = {m.group(1) for m in re.finditer(r"^((?:dryrun_)?phase_\S*)\s+.*\bFAIL\s*$", out, re.M)}
print(f"   expected {sorted(expected)}")
print(f"   actual   {sorted(got)}")
new, gone = got - expected, expected - got
for f in sorted(new):
    print(f"   FAIL: {f} newly fails the audit", file=sys.stderr)
for f in sorted(gone):
    print(f"   {f} no longer fails; update EXPECTED_AUDIT_FAILURES in this script")
raise SystemExit(1 if new else 0)
PY

hdr "B5. everything, checked"
# verify_everything's section 2 fails on the two files named above, by design and on purpose, so
# its overall verdict is expected to be non-zero until they are dealt with. B4 is the check that
# carries information; this one is here for sections 1 and 3 through 9.
bash scripts/verify_everything.sh
note "section 2 is expected to fail on $EXPECTED_AUDIT_FAILURES; B4 is what checks that set"

echo
if [ "$FAIL" = 0 ]; then
  echo "All post-measurement steps passed. Nothing is committed; read the diff first."
else
  echo "At least one step failed. Nothing is committed."
fi
echo "Then decide what becomes of $COMMITTED: the re-run supersedes it, and whether the older file"
echo "stays as the record of that contention is a call for the person who owns the study. Whichever"
echo "way it goes, evidence/registry.json and EXPECTED_AUDIT_FAILURES here both need updating."
exit "$FAIL"
