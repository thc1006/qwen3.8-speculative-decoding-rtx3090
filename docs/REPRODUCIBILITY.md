# Reproducing Phase A

Two different things are called reproduction here and they make different claims, so the script
names them separately.

## Exact rerun

Pins llama.cpp, CUDA, the models, the card **and this repository**, because the harness is part of
the experiment.

```bash
git fetch --tags
git switch --detach phase-a-v1
git tag -v phase-a-v1
./scripts/reproduce_phase_a.sh
```

The checkout is part of the procedure. `phase-a-v1` is the harness Phase A was measured on; the tip
of `master` is a later one, with a different analysis path, different quality rules and different
statistics. Pinning everything except the harness and calling the result an exact reproduction is
what the script used to allow: it compared tag *names* with `git describe --tags --exact-match`,
which does not say which of several tags on one commit comes back -- and two point at that commit
-- and it only printed a warning. It compares commits now and stops.

## Independent replication

```bash
ALLOW_HARNESS_DRIFT=1 ./scripts/reproduce_phase_a.sh
```

Pins everything except this repository. A weaker claim, and a useful one: it asks whether the
measurement survives a later analysis path.

## The pinned revisions

llama.cpp master `c060ca974c773c7c3d17fd1b66dc9d312bc292c0` and the DFlash2 branch at
`d1a522fc89c96d1a3057e35681f0c4859810623c`, both at their full forty characters. The DFlash2
commits are also archived in `repro/dflash2-d1a522fc.bundle`, because `pull/27342/head` is a live
ref that has already moved past what was measured.

## What the script stops for

Four of these seven bullets used to break off mid-clause -- "because mixing a prebuilt master
with", "checksums are checked against", "It never overwrites", "and any incident is" -- under a
heading that said "refuses to do" over a list of checks. Each one below is now what
`scripts/reproduce_phase_a.sh` actually does, read off the script rather than remembered.

Every check here calls `die` and stops the run. The one exception is named at the end.

- the two llama.cpp trees must be at their **full 40-character** pinned commits, and the script
  stops if either is not; `pull/27342/head` has already moved past what was measured, which is
  why the commits are also archived as a bundle;
- both trees are configured and built with identical flags, because mixing a prebuilt master with
  a freshly built branch would put the build itself inside the comparison it exists to control;
- model checksums are checked against `models/SHA256SUMS.phase_a`, the manifest the lock file
  names, so a different quantization cannot be measured under the name of the one that was;
- the card is checked for compute capability 8.6 and about 20 GB free before anything is built;
- the harness's own tests must pass first;
- the result is written to `results/reproductions/phase_a_<host>_<utc>.json`, and it never
  overwrites: a path that already exists is a failure rather than a target, so the committed
  artifact cannot be replaced by a rerun;
- the record count must equal the 875 the lock file declares.

The exception is incidents. A recorded incident prints a warning and lets the run finish, because
the file is still worth keeping and reading. It is the *comparison* that then refuses: it exits
non-zero until `--allow-incidents` says the incident has been read, and asks for that in writing.

## Comparing the result

Absolute tok/s are host-specific. Compare the paired effects, not the levels; see
[the fleet note](GPU_AS_FOUND.md) for why figures from different hosts are never pooled here.

Overlapping intervals between a rerun and the committed result are a **failure to exclude, not
agreement**. No practical-equivalence margin has been preregistered for a rerun, and run-level
variance across independent sessions has not been estimated, so the comparison in
`harness/compare_reproduction.py` reports arm-by-arm overlap and says in its own output that this
is not proof of a reproduction.

## A reduced dry run

```bash
python3 harness/bench.py --matrix phase_a --prompts-per-class 1 --out /tmp/dry.json
```

Reduced runs label themselves in the output file, so they can never be read back as a full result.

## Figures

```bash
.venv/bin/pip install matplotlib && .venv/bin/python analysis/plot.py
```

`analysis/plot.py` imports numpy directly, which installing matplotlib provides. Neither is needed
to reproduce the numbers. The test suite reads `CITATION.cff` with PyYAML when it is installed and
skips that one check when it is not.
