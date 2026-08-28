# Runners

The root of the repository held seventeen of these. They are grouped here by what they drive and
by where they run, because that is the distinction that matters when picking one: a script in
`warp/` will `cd` into a directory that exists on another machine.

Every script assumes it is invoked from anywhere and resolves the repository itself, so
`bash scripts/run_phase_q.sh` works from the root.

## Phase drivers — this host

| script | drives | notes |
|---|---|---|
| `run_remaining.sh` | the whole remaining chain | calls `run_phase_l.sh` and `run_phase_q.sh` in order, with the anchor gate between them |
| `run_phase_q.sh` | Phase Q, the 27B target-quantisation ladder | stages one rung of weights at a time against limited disk and verifies a rung is complete before deleting it |
| `run_phase_qsmall.sh` | Phase Q-small, the 9B ladder | four rungs to bf16; the instrument that reaches the bit span the 27B ladder cannot |
| `run_phase_l.sh` | Phase L, the context-depth ladder | budgeted in seconds so it stops at a rung boundary rather than mid-rung |
| `run_chain.sh` | an earlier chain | superseded by `run_remaining.sh`; kept because results reference it |

## Not runners — they read rather than measure

| script | does | notes |
|---|---|---|
| `verify_everything.sh` | re-checks every claim the repository makes about itself | ten sections: tests, the result audit, links, the README's numbers, withdrawn claims, registry coverage, the generated evidence block, the anchor report, every other generated report, and the GPU's state. CPU-heavy; the measurement guard refuses it while a run holds the GPU lock |
| `reproduce_phase_a.sh` | rebuilds Phase A from `repro/phase_a.lock.json` | pins the engines, the toolchain, the models, the card and this repository's own tag, writes to `results/reproductions/` and compares the paired effects rather than only the record count |
| `post_measurement.sh` | everything that had to wait for a free GPU | Part A decides whether a re-run is usable and stops if it is not; Part B is deferred maintenance. `--maintenance-anyway` forces past a Part A failure |

## Forced-warp intervention — `warp/`

The intervention is finished (four builds from one configure, three hosts). These are kept
because the result files name them and because the same procedure would be needed to repeat it,
not because anything is expected to run them again.

`run_warp_*.sh` and `run_control_repeat_hostc.sh` execute ON the remote host and `cd` into
`$HOME/qwen38-remote` (host B) or `$HOME/qwen38-a6000` (host C). `collect_*.sh` run HERE and pull
the results back. `chain_down2_hostb.sh` does both: it copies the runner over, starts it, and
collects.

Absolute paths are deliberate. A runner that resolved its own directory would follow the wrong
tree when copied to a host whose checkout lives somewhere else, and these are copied by `scp`.
