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
python3 - <<'PY' || exit 1
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
python3 - <<'PY'
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
base = strat("baseline@master")
readme = open("README.md").read()
print(f"   baseline {base:.2f} tok/s, quoted in README: {'41.55' in readme}")
for arm, pct in (("mtp-n2", 59.8), ("mtp-n3", 52.3), ("dflash2-n4", 51.9),
                 ("mtp-n5", 32.1), ("dflash2-n7", 22.6)):
    got = (strat(arm) / base - 1) * 100
    ok = abs(got - pct) < 0.1 and f"+{pct} %" in readme
    print(f"   {arm:12s} recomputed {got:+.2f} %, README says +{pct} %  {'ok' if ok else 'MISMATCH'}")
PY

hdr "5. no committed document carries a withdrawn claim"
python3 - <<'PY'
import subprocess, pathlib
claims = ("not the architecture", "sign belongs to the drafting method",
          "rules out a large architecture effect", "flag was accepted and did nothing",
          "no quantization anywhere", "the cost is linear", "costs a fixed c",
          # Withdrawn by TODO B7: the window is fixed in tokens, so there is no censored subset
          # to check the rest against and no partition surviving on the rest.
          "The partition survives on the 10", "15 of 25 prompts")
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
print(f"   {len(files)} documents scanned, {len(hits)} residual claims")
for h in hits:
    print("   FAIL:", h)
PY

hdr "6. the GPU is where the runs left it"
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
