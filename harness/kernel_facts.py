#!/usr/bin/env python3
"""Reads the CUDA dispatch facts an analysis depends on out of the tree it will actually run.

Three separate defects this study shipped had the same shape: a fact about the kernel written into
a Python file, and later untrue.

  `width_groups.py` hard-coded `calc_nwarps`, so it gave width 9 a warp count the table never
  assigns and scored H8 against a prediction that build never made.
  `cost_model.py` fitted one line across `MMVQ_MAX_BATCH_SIZE`, which dragged the MTP coefficient
  by 24 % and the fit from r2 = 0.9958 to 0.8316.
  `warp_intervention.py` had to be told each forced build's table by hand, because the result file
  did not carry it.

Recording them per run turns all three from a silent wrong answer into a mismatch a reader can
see. Nothing here parses C++ properly - it reads two named constants and one switch out of one
file, and says so when it cannot, rather than guessing.
"""

import hashlib
import re
from pathlib import Path


def _read(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def parse_generic_table(src):
    """{ncols_dst -> warp count} from the GENERIC arm of `calc_nwarps`, or None if absent.

    Accepts either a whole `mmvq.cu` or the extracted block the warp runs save as `table.txt`,
    since both begin at the same line. Cases fall through to the next `return`, which is what
    makes `case 1: case 2: case 3: case 4: return 4;` four entries rather than one.

    warp_intervention.py needs this to check the table it assumes for a build against the source
    that build was compiled from. Its docstring claimed such a check existed; it did not, and a
    build whose table the scorer had wrong would have been scored against the wrong prediction
    in silence.
    """
    start = src.find("if (table_id == MMVQ_PARAMETERS_GENERIC)")
    if start < 0:
        return None
    end = src.find("} else if", start)
    block = src[start:end if end > 0 else start + 2000]

    table, pending = {}, []
    for tok in re.finditer(r"case\s+(\d+)\s*:|return\s+(\d+)\s*;|default\s*:", block):
        if tok.group(1):
            pending.append(int(tok.group(1)))
        elif tok.group(2) and pending:
            for c in pending:
                table[c] = int(tok.group(2))
            pending = []
        elif tok.group(0).startswith("default"):
            pending = []
    return table


def mmvq_facts(tree):
    """`MMVQ_MAX_BATCH_SIZE` and the GENERIC arm of `calc_nwarps`, from `tree`'s own source."""
    out = {"source": None, "mmvq_max_batch_size": None, "generic_nwarps": None, "note": None}
    src_path = Path(tree) / "ggml/src/ggml-cuda/mmvq.cu"
    src = _read(src_path)
    if src is None:
        out["note"] = f"{src_path} not readable; nothing recorded rather than assumed"
        return out
    out["source"] = str(src_path)

    # The constant is used in mmvq.cu but defined in a header, so search the backend rather than
    # the one file. Reporting null when it is not found is deliberate: a wrong 8 would be worse
    # than an absent one.
    m = None
    for cand in (src_path, *sorted((Path(tree) / "ggml/src/ggml-cuda").glob("*.cuh")),
                 *sorted((Path(tree) / "ggml/src/ggml-cuda").glob("*.h"))):
        text = _read(cand)
        if text is None:
            continue
        m = (re.search(r"#define\s+MMVQ_MAX_BATCH_SIZE\s+(\d+)", text)
             or re.search(r"MMVQ_MAX_BATCH_SIZE\s*=\s*(\d+)", text))
        if m:
            out["mmvq_max_batch_size"] = int(m.group(1))
            out["mmvq_max_batch_size_source"] = str(cand)
            break
    if not m:
        out["note"] = "MMVQ_MAX_BATCH_SIZE not found in the backend sources"

    table = parse_generic_table(src)
    if table is None:
        out["note"] = "no MMVQ_PARAMETERS_GENERIC branch found"
        return out
    out["generic_nwarps"] = {str(k): v for k, v in sorted(table.items())} or None
    if not table:
        out["note"] = "the GENERIC branch did not parse as a case/return table"
    return out


def backend_binaries(tree):
    """Hash the library that carries the kernels, not the launcher.

    `llama-server` is a 17 KB wrapper that dlopens the backend and is byte-identical across builds
    whose CUDA code differs. A build check that hashed it reported three matching builds of three
    different tables.
    """
    out = {}
    for pat in ("build/bin/libggml-cuda.so*", "build/bin/libggml.so*", "build/bin/llama-server"):
        for p in sorted(Path(tree).glob(pat)):
            if p.is_symlink() or not p.is_file():
                continue
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[p.name] = {"sha256_16": h.hexdigest()[:16], "bytes": p.stat().st_size}
    return out


def snapshot(trees):
    """{tree name -> dispatch facts} for every tree a matrix declares."""
    out = {}
    for name, path in (trees or {}).items():
        out[str(name)] = {"path": str(path), "mmvq": mmvq_facts(path),
                          "binaries": backend_binaries(path)}
    return out


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(snapshot({"tree": sys.argv[1] if len(sys.argv) > 1 else "."}), indent=2))
