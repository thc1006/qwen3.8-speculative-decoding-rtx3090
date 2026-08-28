#!/usr/bin/env python3
"""Render the README's evidence-status block from the result files, or check it has not drifted.

The block this writes used to be a hand-maintained blockquote. It carried a date, and the date did
not stop it: on 2026-08-27 it said Phase B was running while 525 committed records sat in
`results/phase_b.json`, and it never mentioned Phase R at all, which has 1125. Every number here is
computed from the files each time this runs, so the only things a human maintains are the ones a
file cannot state -- the question, how strongly the phase may be read, and what it must not be used
to claim. Those live in `evidence/registry.json`.

  render_evidence.py            rewrite the block in README.md
  render_evidence.py --check    exit non-zero if the block is not what would be written now
  render_evidence.py --only A,B restrict to some phases, for a cheap check

Reading every result file costs a few seconds of CPU on 61 MB of JSON, which is enough to be
recorded as contention by a running measurement. `--only` exists so this can be exercised without
that; the CPU guard denies the full run while the GPU lock is held.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import completeness as CP     # noqa: E402

ROOT = Path(__file__).parent.parent
REGISTRY = ROOT / "evidence" / "registry.json"
README = ROOT / "README.md"
BEGIN = "<!-- BEGIN GENERATED: EVIDENCE_STATUS -->"
END = "<!-- END GENERATED: EVIDENCE_STATUS -->"
FBEGIN = "<!-- BEGIN GENERATED: FORBIDDEN_CLAIMS -->"
FEND = "<!-- END GENERATED: FORBIDDEN_CLAIMS -->"

INFERENCE = {
    "primary": "primary result",
    "reported": "within-run contrasts reported",
    "exploratory": "exploratory",
    "association": "association, not a controlled contrast",
    "control": "control",
    "not-evaluable": "**not evaluable**",
}


def files_for(phase, skip):
    out = []
    for pat in phase["results"]:
        for p in sorted(glob.glob(str(ROOT / pat))):
            name = Path(p).name
            if any(s in name for s in skip):
                continue
            out.append(Path(p))
    return out


def facts(phase, skip):
    """(files, records, expected, incidents, short note) computed from the files themselves."""
    n_files = n_rec = n_exp = n_inc = 0
    short = []
    for p in files_for(phase, skip):
        d = json.loads(p.read_text())
        got, expected, _ = CP.completeness(d)
        n_files += 1
        n_rec += got
        n_exp += expected or 0
        n_inc += len(d.get("incidents") or [])
        if expected and got < expected:
            short.append(f"{p.name} {got}/{expected}")
    return n_files, n_rec, n_exp, n_inc, short


def validate(reg):
    """Raise on a registry the renderer would otherwise pass through unchanged.

    The registry's own comment says `inference` is a controlled vocabulary and not free text. It
    was not enforced: `INFERENCE.get(value, value)` fell back to printing whatever was there, so
    `reportd` would have reached the README as `reportd` with nothing raised. A file that declares
    its own constraint and a reader that does not apply it is the shape of defect this repository
    keeps finding; it should not be in the thing built to stop it.
    """
    seen, problems = set(), []
    for phase in reg["phases"]:
        pid = phase.get("id")
        if not pid:
            problems.append("a phase has no id")
            continue
        if pid in seen:
            problems.append(f"duplicate phase id {pid!r}: its rows would appear twice")
        seen.add(pid)
        inf = phase.get("inference")
        if inf not in INFERENCE:
            problems.append(f"{pid}: inference {inf!r} is not one of {sorted(INFERENCE)}")
        if not phase.get("results"):
            problems.append(f"{pid}: no result patterns")
    if problems:
        raise SystemExit("evidence/registry.json is not valid:\n  " + "\n  ".join(problems))


def render_forbidden(reg, only=None):
    """The claims each phase may not be used to make, from the registry, as a table.

    Twenty-six of these were declared and nothing read them: they sat in a JSON file as a note to
    whoever wrote it. Publishing them is most of the point -- a limit a reader cannot see is a
    limit only the author is bound by. They are NOT mechanically enforced and this says so: they
    are sentences about what an argument may not do, not strings a scanner can grep for. Section 5
    of verify_everything.sh catches specific withdrawn wordings; these are the wider constraints
    those wordings came from.
    """
    rows = ["| phase | must not be used to claim |", "|---|---|"]
    for phase in reg["phases"]:
        if only and phase["id"] not in only:
            continue
        f = phase.get("forbidden") or []
        if not f:
            continue
        rows.append(f"| {phase['id']} | " + "<br>".join(f) + " |")
    return "\n".join(rows)


def render(only=None, reg=None):
    reg = reg or json.loads(REGISTRY.read_text())
    validate(reg)
    skip = reg.get("skip_patterns") or []
    rows = ["| phase | data, computed from the files | inference |", "|---|---|---|"]
    for phase in reg["phases"]:
        if only and phase["id"] not in only:
            continue
        n_files, n_rec, n_exp, n_inc, short = facts(phase, skip)
        if not n_files:
            data = "**no result file**"
        else:
            data = f"{n_rec} records"
            if n_files > 1:
                data += f" over {n_files} files"
            if n_exp and n_rec < n_exp:
                data += f", **short of {n_exp}**"
            elif n_exp:
                data += ", complete"
            data += f", {n_inc} incident" + ("" if n_inc == 1 else "s")
            if short:
                data += " (" + "; ".join(short[:2]) + ")"
        inf = INFERENCE.get(phase["inference"], phase["inference"])
        note = phase.get("note")
        if note:
            inf += f" -- {note}"
        # Plain, not bold. `| **M** |` is exactly the syntax the later-phases findings table
        # uses for its rows, so this block put a second `| **M** |` and a second `| **Q** |` into
        # the document, and every check that greps for a phase row started matching whichever came
        # first -- which is this one. Two tests broke that way: the Phase Q rung count and the
        # Phase M "cost interpretation withheld" guard were both reading a status row that has
        # neither. A generated block must not collide with the hand-written table it sits above.
        rows.append(f"| {phase['id']} | {data} | {inf} |")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--stdout", action="store_true", help="print the block and write nothing")
    args = ap.parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()} or None

    reg = json.loads(REGISTRY.read_text())
    block = render(only, reg)
    fblock = render_forbidden(reg, only)
    if args.stdout:
        print(block)
        print()
        print(fblock)
        return 0

    text = README.read_text()
    new = text
    for begin, end, b in ((BEGIN, END, block), (FBEGIN, FEND, fblock)):
        if begin not in new or end not in new:
            print(f"README.md carries no {begin} ... {end} block", file=sys.stderr)
            return 2
        head, rest = new.split(begin, 1)
        _, tail = rest.split(end, 1)
        new = f"{head}{begin}\n{b}\n{end}{tail}"

    if args.check:
        if only:
            print("--check with --only would compare a partial block against the whole one",
                  file=sys.stderr)
            return 2
        if new == text:
            print("   evidence block matches the result files")
            return 0
        print("   the evidence block is not what the result files say now:", file=sys.stderr)
        import difflib
        for line in list(difflib.unified_diff(text.splitlines(), new.splitlines(),
                                              "committed", "computed", lineterm=""))[:24]:
            print("   " + line, file=sys.stderr)
        return 1

    # Write beside the target and rename, rather than truncating the README and writing into it.
    # `os.replace` is atomic within a filesystem, so a crash leaves the old README rather than half
    # of a new one. The same fix went into post_measurement.sh's two `cmd > committed_artifact`
    # redirects, which truncated the file before the command that fills it had run at all.
    tmp = README.with_suffix(".md.tmp")
    tmp.write_text(new)
    os.replace(tmp, README)
    print("   evidence block rewritten from the result files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
