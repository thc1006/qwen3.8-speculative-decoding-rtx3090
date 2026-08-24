"""Build context filler of an exact token length, from real non-repeating prose.

Why the text has to be real. Filling a context by repeating a paragraph hands the drafter the
n-gram predictability of that repetition and inflates acceptance for a reason that has nothing
to do with context depth. Two independent RTX 3090 reports quantify that artifact directly: with
`ngram-mod`, repeated passes of the same prompt read 111.1 tok/s cold and then 124.4 and 122.5
warm. A long-context measurement built on repeated filler would be measuring its own filler.

Sources are three public-domain novels, which between them give roughly 1.2 M tokens, far more
than the 128 K this study needs, so each depth can be cut without recycling text.

Token counts come from the running server's /tokenize endpoint, so they are the model's own
tokenizer and not an estimate. Truncation is by binary search on the character length, which
converges in about 20 requests and lands exactly.
"""
from __future__ import annotations

import json
import re
import urllib.request
from functools import lru_cache
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# Project Gutenberg wraps each text in a licence header and footer. They repeat across files and
# are not prose, so they come out.
_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S | re.I)
_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S | re.I)


@lru_cache(maxsize=1)
def corpus() -> str:
    """All available prose, stripped of Gutenberg boilerplate and normalised whitespace."""
    parts = []
    for p in sorted(ASSETS.glob("*.txt")):
        t = p.read_text(encoding="utf-8", errors="ignore")
        m = _START.search(t)
        if m:
            t = t[m.end():]
        m = _END.search(t)
        if m:
            t = t[:m.start()]
        t = re.sub(r"\r\n", "\n", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        parts.append(t.strip())
    if not parts:
        raise RuntimeError(
            f"no .txt found in {ASSETS}. Fetch the public-domain sources first; see "
            f"docs/PHASE_L_DESIGN.md.")
    return "\n\n".join(parts)


def count_tokens(port: int, text: str, timeout_s: float = 300.0) -> int:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/tokenize",
        data=json.dumps({"content": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return len(json.loads(r.read())["tokens"])


def filler_of(port: int, n_tokens: int, *, offset_chars: int = 0,
              tolerance: int = 8, max_probes: int = 30) -> tuple[str, int]:
    """Prose cut to `n_tokens` as measured by the model's own tokenizer.

    Returns (text, realised token count). The realised count is returned rather than assumed,
    and callers should record it: a depth that did not actually materialise should be visible
    in the data, not inferred from what was requested.
    """
    src = corpus()
    if offset_chars:
        offset_chars %= len(src)
        src = src[offset_chars:] + src[:offset_chars]

    # ~3.6 chars per token for English prose with this tokenizer; the search corrects it anyway
    lo, hi = 1, min(len(src), int(n_tokens * 12) + 4096)
    best_text, best_n = src[:hi], count_tokens(port, src[:hi])
    if best_n < n_tokens:
        return best_text, best_n          # corpus exhausted; caller sees the shortfall

    for _ in range(max_probes):
        mid = (lo + hi) // 2
        cand = src[:mid]
        got = count_tokens(port, cand)
        if abs(got - n_tokens) <= tolerance:
            return cand, got
        if got < n_tokens:
            lo = mid + 1
        else:
            hi = mid - 1
            best_text, best_n = cand, got
        if lo > hi:
            break
    return best_text, best_n
