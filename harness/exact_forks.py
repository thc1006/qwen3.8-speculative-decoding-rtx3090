#!/usr/bin/env python3
"""Where two greedy outputs first differ, in tokens, without a re-measurement.

The fork positions this repository reported were characters divided by the output's MEAN
characters per token. That is an average over a variable-length tokenizer, and measured against the
exact answer on 366 divergent records it is off by more than five tokens on 56.6 % of them and by
more than twenty on 15.6 %, with a range of -26 to +61. The median error is zero, which is what let
it look serviceable.

The exact answer does not need the run repeated. Both outputs share a byte-identical prefix, and a
BPE tokenizer segments identical text identically, so tokenizing the two stored strings and finding
the first index where they differ gives the fork position exactly. What re-tokenization does NOT
reproduce is the total count -- the round trip through text loses a token or two at the end, on 5 of
8 sampled records -- but that is a property of the tail, and the fork is not in the tail.

The alternative was to record emitted token ids during the run. `return_tokens` exists in the pinned
server and touches nothing but a push_back, so it would have been safe; it is unreachable here
because the harness posts to `/v1/chat/completions`, and `tokens` is in the native `to_json()` and
not in `to_json_oaicompat_chat()`. Switching endpoints would change the request path and the chat
template handling for every future run, which is a worse trade than tokenizing what is already
stored.

    llama-server -m <target.gguf> -ngl 0 -c 512 --port 18899     # CPU only; no GPU needed
    python3 harness/exact_forks.py results/phase_a_cap1600.json
"""
from __future__ import annotations

import json
import statistics
import sys
import urllib.request

PORT = 18899


def tokenize(text: str, port: int = PORT) -> list[int]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/tokenize",
        data=json.dumps({"content": text, "add_special": False}).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())["tokens"]


def first_differing_token(a_ids: list[int], b_ids: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a_ids, b_ids)):
        if x != y:
            return i
    return min(len(a_ids), len(b_ids))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    data = json.loads(open(sys.argv[1]).read())
    baseline = {(r["prompt"], r["pass"]): r["text"]
                for r in data["records"] if r["arm"] == "baseline@master"}
    if not baseline:
        print("no baseline@master records; nothing to compare against")
        return 2

    cache: dict[str, list[int]] = {}

    def ids(text):
        if text not in cache:
            cache[text] = tokenize(text)
        return cache[text]

    rows, errs = [], []
    for r in data["records"]:
        dv = r.get("divergence") or {}
        if dv.get("first_diff_char") is None:
            continue
        base = baseline.get((r["prompt"], r["pass"]))
        if base is None:
            continue
        fork = first_differing_token(ids(r["text"]), ids(base))
        cpt = (len(r["text"]) / r["predicted_n"]) if r.get("predicted_n") else None
        est = round(dv["first_diff_char"] / cpt) if cpt else None
        rows.append((r["arm"], r["prompt"], r["pass"], dv["first_diff_char"], fork, est))
        if est is not None:
            errs.append(fork - est)

    print("=" * 92)
    print("FIRST DIVERGING TOKEN, BY TOKENIZING THE STORED OUTPUTS")
    print("=" * 92)
    print(f"  {len(rows)} divergent records, tokenized against their own baseline pass.")
    print()
    forks = [r[4] for r in rows]
    print(f"  fork position   earliest {min(forks)}   median {statistics.median(forks):.0f}   "
          f"latest {max(forks)}")
    print()
    print("  against the character-per-token estimate this repository used to report:")
    print(f"    median error {statistics.median(errs):+.0f}, mean {statistics.fmean(errs):+.1f}, "
          f"sd {statistics.stdev(errs):.1f}, range {min(errs):+d} to {max(errs):+d}")
    print(f"    off by more than  5 tokens on {100 * sum(1 for e in errs if abs(e) > 5) / len(errs):.1f} % "
          f"of records")
    print(f"    off by more than 20 tokens on {100 * sum(1 for e in errs if abs(e) > 20) / len(errs):.1f} % "
          f"of records")
    print()
    # This printed "The median error is zero" whatever the median was, and on the 400-token cap it
    # is +1. A sentence that does not follow the number printed two lines above it is the defect
    # this repository keeps finding; it should not be in the tool written to remove one.
    med = statistics.median(errs)
    big = 100 * sum(1 for e in errs if abs(e) > 5) / len(errs)
    huge = 100 * sum(1 for e in errs if abs(e) > 20) / len(errs)
    print(f"  The median error is {med:+.0f}, which is why the estimate looked serviceable, and it is")
    print(f"  not a defence of it: {big:.0f} % of records are wrong by more than five tokens in one")
    print(f"  direction or the other, and {huge:.0f} % by more than twenty.")
    print()
    print("  earliest fork per arm")
    by_arm: dict[str, int] = {}
    for arm, _p, _pa, _c, fork, _e in rows:
        by_arm[arm] = min(by_arm.get(arm, 10 ** 9), fork)
    for arm in sorted(by_arm):
        print(f"    {arm:22s} token {by_arm[arm]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
