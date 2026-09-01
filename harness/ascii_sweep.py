#!/usr/bin/env python3
"""Replaces typographic unicode in the files this study writes with ASCII.

Em dashes, minus signs, middle dots and arrows are the visible signature of text pasted out of a
model, and llama.cpp's AGENTS.md asks contributors for ASCII on the same grounds. They also break
in terminals, in diffs and in anything that reads the file as latin-1.

Each character gets the replacement its context calls for rather than a single fallback: a middle
dot between navigation links is a separator, a middle dot in `c(w - 1)` is multiplication, and
collapsing both to a hyphen would turn one of them into subtraction.

Skipped, deliberately:
  assets/           Project Gutenberg source texts. The em dashes there are Tolstoy's and Melville's
                    and rewriting them would corrupt the prompt corpus.
  results/          measurements, including generated text
  analysis/*.txt    generated; fix the generator and regenerate instead of editing the output
"""

import io
import os
import re
import subprocess
import sys

# ordered: the multi-character contexts must be tried before the single-character fallback
RULES = [
    ("→", "->"),      # rightwards arrow
    ("≤", "<="),      # less-than or equal
    ("≥", ">="),      # greater-than or equal
    ("≈", "~"),       # almost equal
    ("²", "^2"),      # superscript two, as in r^2
    ("×", "x"),       # multiplication sign
    ("−", "-"),       # minus sign
    ("–", "-"),       # en dash, used here only in numeric ranges
    ("‘", "'"), ("’", "'"),
    ("“", '"'), ("”", '"'),
    ("…", "..."),
    ("\u00b0C", "C"),   # degree sign, only ever used here as "60 degC"
    ("\u00b0", " deg"),
    ("\u00b1", "+/-"),  # plus-minus
    ("\u2260", "!="),   # not equal
    ("\u00a7", "section "),
    ("\u0394", "delta"),
    ("\u2193", "down"), ("\u2191", "up"),
]

SKIP_DIRS = ("assets/", "results/", ".git/")
SKIP_EXT = (".json", ".jsonl", ".gguf", ".png", ".svg", ".patch", ".pyc")


def convert(text):
    # A middle dot separating navigation links is punctuation; one inside an expression is a
    # multiplication operator. The space on both sides is what distinguishes them.
    # A trailing separator has no space after it, so match end-of-line before the spaced form;
    # falling through to the multiplication rule would leave a bare "*" that Markdown reads as
    # the start of emphasis.
    # A middle dot separating navigation links is punctuation, one inside an expression is
    # multiplication, and one inside a markdown table cell is neither: a pipe there is a column
    # separator and splits the row. That is not hypothetical - it broke five rows in README.md
    # and TODO.md on the first pass, and the tables rendered with the wrong number of columns.
    out = []
    for line in text.split("\n"):
        if line.lstrip().startswith("|"):
            out.append(line.replace(" · ", ", ").replace("·", "*"))
        else:
            out.append(re.sub(r" ·$", " |", line).replace(" · ", " | ").replace("·", "*"))
    text = "\n".join(out)
    # An em dash with spaces reads as a parenthetical and keeps its spacing; one without is
    # joining two words and becomes a plain hyphen.
    text = text.replace(" — ", " - ").replace("—", "-")
    for a, b in RULES:
        text = text.replace(a, b)
    return text


def main():
    paths = sys.argv[1:]
    if not paths:
        out = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                             timeout=15).stdout
        paths = [p for p in out.splitlines()
                 if not p.startswith(SKIP_DIRS) and not p.endswith(SKIP_EXT)
                 and not (p.startswith("analysis/") and p.endswith(".txt"))]
    changed = 0
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            src = io.open(p, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        dst = convert(src)
        if dst == src:
            continue
        io.open(p, "w", encoding="utf-8").write(dst)
        before = sum(1 for c in src if ord(c) > 127)
        after = sum(1 for c in dst if ord(c) > 127)
        print("  %-46s %d -> %d non-ASCII" % (p, before, after))
        changed += 1
    print("  %d files rewritten" % changed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
