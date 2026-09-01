# The documents in this repository

Nine of these were reachable from nothing: `TODO.md`, `docs/A6000_PLAN.md`,
`docs/DISCOVERY.md`, `docs/EXPERIMENT_PLAN.md`, `docs/PHASE_L_DESIGN.md`,
`repro/FINDINGS_27572.md`, `repro/output_reorder_ordering/README.md`,
`results/salvage/README.md` and `scripts/README.md`. A document nothing links to is a document
nobody reads and nobody rechecks, which is how several of them went stale. This index exists so
that cannot happen silently again.

## The result, and what bounds it

| document | what it is for |
|---|---|
| [`../README.md`](../README.md) | the primary result, its limits, and how to reproduce it |
| [`PHASES.md`](PHASES.md) | every follow-up phase: its question, its status, and what it may not be used to claim |
| [`STATISTICAL_SCOPE.md`](STATISTICAL_SCOPE.md) | what the intervals are, what they cover, and what none of them carry |
| [`../PREREGISTRATION.md`](../PREREGISTRATION.md) | the append-only record: hypotheses before their data, and 55 dated corrections |

## The four mechanism reports

| document | what it is for |
|---|---|
| [`COST_MODEL.md`](COST_MODEL.md) | why deeper drafting stops paying, and what `k`, `c` and `k0` do and do not identify |
| [`GREEDY_DIVERGENCE.md`](GREEDY_DIVERGENCE.md) | how far the output moves from serial greedy decoding, and the width boundary it moves at |
| [`ENERGY.md`](ENERGY.md) | the decode-energy figure, the instrument, and the six phases that bound it |
| [`RESOURCE_RESPONSE.md`](RESOURCE_RESPONSE.md) | which clock each workload responds to, measured rather than inferred |

## Method, and the record of what went wrong

| document | what it is for |
|---|---|
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | the exact rerun, the independent replication, and what the script refuses |
| [`METHODOLOGY_AUDIT.md`](METHODOLOGY_AUDIT.md) | the audit of the predecessor study that this harness was designed against; its verdict on the central conclusion is withdrawn |
| [`GPU_AS_FOUND.md`](GPU_AS_FOUND.md) | the card arrived overclocked while the README said stock, and what that changed |
| [`UPSTREAM_CONTRIBUTIONS.md`](UPSTREAM_CONTRIBUTIONS.md) | what is worth sending upstream, to whom, and what must be said when posting it |
| [`DISCOVERY.md`](DISCOVERY.md) | how this repository expects to be found, and the vocabulary that would make it findable |

## Designs, written before their data

Each of these is dated and superseded in part by what actually ran. Read the result in
[`PHASES.md`](PHASES.md) first, then these for why the design is what it is.

| document | what it is for |
|---|---|
| [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) | the 2026-08-24 plan and the pilot behind it. Superseded as status; kept for the pre-registered directional predictions |
| [`PHASE_L_DESIGN.md`](PHASE_L_DESIGN.md) | the long-context ladder: the VRAM derivation, and the two departures from this study's standard protocol |
| [`PHASE_V_DESIGN.md`](PHASE_V_DESIGN.md) | the vLLM comparison, its memory arithmetic, and the three hypotheses it registered |
| [`A6000_PLAN.md`](A6000_PLAN.md) | prepared for a card that turned out to be on another host; the correctness work it forced is the part that survived |

## Working notes

| document | what it is for |
|---|---|
| [`../TODO.md`](../TODO.md) | the working list: what is done, what is open, and what was deleted from scope with reasons |
| [`../scripts/README.md`](../scripts/README.md) | what each driver script does, and which of them hard-code this checkout's path |
| [`../repro/FINDINGS_27572.md`](../repro/FINDINGS_27572.md) | a non-reproduction of llama.cpp #27572 on sm_86, stated as that |
| [`../repro/output_reorder_ordering/README.md`](../repro/output_reorder_ordering/README.md) | the probe harness behind the `output_reorder()` change |
| [`../results/salvage/README.md`](../results/salvage/README.md) | 21 baseline observations from the aborted overclocked run, kept rather than discarded |
| [`../assets/README.md`](../assets/README.md) | the three public-domain texts used as long-context filler, and their checksums |
