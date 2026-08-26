"""Put `env.model_size_bytes` into a result that ran before bench.py recorded it.

The quantization ladders delete their weights once a rung verifies, and the rung's file size is
its x coordinate on a bits-per-weight axis. bench.py records it now; results measured before that
do not have it, and `ladder_trend.py` refuses a rung with no size rather than placing it by
guesswork -- so one missing number makes a whole ladder unplottable.

The size comes from models/SHA256SUMS, which is where this repository records the identity of
weights it has staged out. It is only written into a result whose `env.model_sha256` MATCHES the
hash recorded beside that size. Without that check this would be a tool for writing an arbitrary
number into a measurement file, which is worse than the missing field.

Never overwrites an existing value: a result that already carries a size recorded it from the
file it actually loaded, and this file's provenance is weaker than that.

    python3 harness/backfill_model_size.py results/phase_qsmall_Q4_K_M.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMS = ROOT / "models/SHA256SUMS"


def sizes_by_hash() -> dict[str, tuple[int, str]]:
    """{sha256 -> (bytes, path)} for every entry whose size is recorded in a nearby comment.

    The file is a standard `<hash>  <path>` list; sizes live in comments, because adding a column
    would stop `sha256sum -c` working. A size counts only on the line IMMEDIATELY above its hash.

    An earlier version attached a size to the next hash line ANYWHERE below it, and a prose block
    mentioning two sizes gave the second one to a hash that owned the first: UD-Q5_K_XL's hash
    came back with UD-Q4_K_XL's byte count. That number would then have been written into a
    measurement file as the size of the weights that produced it. Adjacency is checked rather
    than assumed, and a size that is not adjacent to a hash is silently ignored rather than
    attached to whatever comes next.
    """
    out: dict[str, tuple[int, str]] = {}
    pending: int | None = None
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            m = re.search(r"\b(\d{9,})\s*bytes\b", line)
            pending = int(m.group(1)) if m else None
            continue
        parts = line.split()
        if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]) and pending is not None:
            out[parts[0]] = (pending, parts[1])
        pending = None
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    table = sizes_by_hash()
    if not table:
        print(f"no sizes recorded in {SUMS}", file=sys.stderr)
        return 1
    rc = 0
    for path in sys.argv[1:]:
        p = Path(path)
        d = json.loads(p.read_text(encoding="utf-8"))
        env = d.get("env") or {}
        have = env.get("model_size_bytes")
        sha = env.get("model_sha256")
        if have:
            print(f"{p.name}: already has model_size_bytes={have}; left alone")
            continue
        if not sha or sha not in table:
            print(f"{p.name}: env.model_sha256 {str(sha)[:16]}... is not in SHA256SUMS with a "
                  f"recorded size; refusing to invent one")
            rc = 1
            continue
        size, named = table[sha]
        env["model_size_bytes"] = size
        d["env"] = env
        # Say where it came from, in the file, so this is not indistinguishable from a field
        # bench.py wrote at measurement time.
        env["model_size_bytes_source"] = (
            f"backfilled from models/SHA256SUMS, matched on model_sha256; the file recorded there "
            f"is {named}")
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(d), encoding="utf-8")
        tmp.replace(p)
        print(f"{p.name}: model_size_bytes={size} written (hash matched {named})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
