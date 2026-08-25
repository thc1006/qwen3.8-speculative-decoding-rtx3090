#!/usr/bin/env python3
"""Removes the cache incidents a detector raised against a condition the file never claimed.

Until 8143dd3 the cache check in bench.py fired on any t_cache_n > 0 and wrote "despite
cache_prompt=False" into the message as a constant. A matrix that sets CACHE_PROMPT = True on
purpose - phase_l does, because every request in an arm shares a filler of up to 96 K tokens -
therefore reported one incident per request against a condition it had not set.

Such an incident is identifiable rather than a judgement call: it says cache_prompt=False and the
file's own design block says cache_prompt is True. Both cannot be true of the same run, and the
design block is what the run was configured with.

Nothing measured changes. Exclusion from the aggregates is decided by analyze._usable() per
record and never reads this list, so no record's membership in any aggregate moves. What changes
is the integrity headline, which counted 180 of these per rung and would have buried a real one.

The removed entries are kept in `incidents_repaired` rather than deleted, so the count is still
recoverable from the file and the repair is visible to anyone reading it later.

    repair_cache_incidents.py results/phase_l_8192.json [--apply]

Without --apply it reports and writes nothing.
"""

import json
import sys
from datetime import datetime, timezone

FALSE_CLAIM = "despite cache_prompt=False"


def find(result):
    """-> (false positives, the rest). Empty first list means there is nothing to repair."""
    if not result.get("design", {}).get("cache_prompt"):
        return [], result.get("incidents", [])
    bad, keep = [], []
    for inc in result.get("incidents", []):
        if inc.get("kind") == "prompt_cache_hit" and FALSE_CLAIM in inc.get("detail", ""):
            bad.append(inc)
        else:
            keep.append(inc)
    return bad, keep


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    apply_it = "--apply" in sys.argv[2:]
    with open(path) as fh:
        result = json.load(fh)

    n_rec = len(result.get("records", []))
    bad, keep = find(result)
    print(f"  {path}")
    print(f"    design.cache_prompt : {result.get('design', {}).get('cache_prompt')}")
    print(f"    records             : {n_rec}")
    print(f"    incidents           : {len(result.get('incidents', []))}")
    print(f"    false positives     : {len(bad)}")
    print(f"    genuine, kept       : {len(keep)}")
    if keep:
        kinds = {}
        for inc in keep:
            kinds[inc.get("kind")] = kinds.get(inc.get("kind"), 0) + 1
        print(f"      by kind: {kinds}")
    if not bad:
        print("    nothing to repair")
        return 0
    if not apply_it:
        print("    dry run; pass --apply to write")
        return 0

    result["incidents"] = keep
    result.setdefault("incidents_repaired", []).append({
        "when": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "removed": len(bad),
        "kind": "prompt_cache_hit",
        "why": ("raised against cache_prompt=False while this file's design block sets "
                "cache_prompt=True; the detector was not conditional on the declared mode "
                "before commit 8143dd3"),
        "measured_values_changed": "none: exclusion is decided per record by analyze._usable()",
        "sample": bad[0] if bad else None,
    })
    tmp = path + ".repair.tmp"
    with open(tmp, "w") as fh:
        json.dump(result, fh)
    import os
    os.replace(tmp, path)
    print(f"    removed {len(bad)}; recorded under incidents_repaired")
    return 0


if __name__ == "__main__":
    sys.exit(main())
