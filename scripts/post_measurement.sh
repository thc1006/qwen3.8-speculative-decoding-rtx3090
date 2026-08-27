#!/usr/bin/env bash
# Everything that had to wait for the GPU, in the order it has to happen.
#
# Written during the Phase A re-run at a 1600-token cap, when the CPU guard was correctly refusing
# every analyser in the repository. The work below is all prepared and tested on subsets; what it
# needed was a free machine. Running it by hand out of a checklist is how a step gets skipped, so
# it is a script, and it stops at the first thing that fails.
set -uo pipefail
cd "$(dirname "$0")/.."
FAIL=0
hdr() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
bad() { FAIL=1; printf '   FAIL: %s\n' "$*"; }

RERUN=results/phase_a_cap1600.rerun.json
COMMITTED=results/phase_a_cap1600.json

if [ -f .gpu-in-use.lock ]; then
  echo "REFUSING: .gpu-in-use.lock still exists. This script is the thing that runs after."
  cat .gpu-in-use.lock
  exit 2
fi

hdr "1. did the re-run finish, and is it clean"
if [ ! -f "$RERUN" ]; then
  echo "   $RERUN is not there; nothing to do"
  exit 2
fi
python3 harness/audit_results.py "$RERUN" || bad "the re-run does not audit clean"

hdr "2. the re-run against the file it supersedes"
# Same matrix, same cap, same host: this is the one comparison in the repository where the two
# runs really are the same experiment, so a disagreement here is about the machine and not the
# design. The committed file carries two host_contended incidents, which is why it exists.
python3 harness/compare_reproduction.py "$COMMITTED" "$RERUN" --allow-incidents \
  || bad "the re-run and the committed file do not agree"

hdr "3. did the cross-tree control come back"
python3 - <<'PY' || bad "tree_divergence is still absent"
import json, sys
d = json.load(open("results/phase_a_cap1600.rerun.json"))
n = sum(1 for r in d["records"] if r.get("tree_divergence"))
ident = sum(1 for r in d["records"]
            if (r.get("tree_divergence") or {}).get("identical"))
print(f"   {n} records carry tree_divergence, {ident} of them identical")
if not n:
    print("   the bench.py fix from Correction 34 did not take effect", file=sys.stderr)
    raise SystemExit(1)
PY

hdr "4. coverage, at a replication count that can tell the figures apart (TODO D6)"
# 300 replications give a Monte Carlo standard error of 1.4 to 2.0 points, which is why 93.7 %
# here and 90.9 % in stats.py look like a contradiction and are 2.0 standard errors apart. About
# 1900 pins it to half a point. This is the slow step: roughly 6.7x the work of the committed run.
python3 harness/coverage_sim.py --replications 2000 > analysis/bootstrap_coverage.txt \
  || bad "coverage_sim did not finish"
head -14 analysis/bootstrap_coverage.txt | sed 's/^/   /'

hdr "5. the README's evidence block, generated (TODO D7)"
python3 harness/render_evidence.py || bad "render_evidence did not write the block"

hdr "6. the anchor report, regenerated from the current analyser"
python3 harness/anchor_verdict.py results/phase_m.json > analysis/phase_m_anchor.txt 2>&1
grep -q "PRIMARY" analysis/phase_m_anchor.txt || bad "the anchor report looks wrong"

hdr "7. everything, checked"
bash scripts/verify_everything.sh || bad "verify_everything reported a problem"

echo
if [ "$FAIL" = 0 ]; then
  echo "All post-measurement steps passed. Nothing is committed; read the diff first."
else
  echo "At least one step failed. Nothing is committed."
fi
echo "Then decide what becomes of $COMMITTED: the re-run supersedes it, and whether the older"
echo "file stays as a record of the contention or goes is a call for the person who owns the study."
exit "$FAIL"
