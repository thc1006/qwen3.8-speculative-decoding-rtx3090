# Being found by retrieval agents

Most traffic to a repo like this one now arrives through a model, not a person. Someone asks an
assistant why their Qwen3 setup is slow, the assistant runs two or three searches, reads what
comes back, and answers. The repo either turns up in those searches and survives the summary, or
it does not exist as far as that user is concerned.

That is a different optimisation problem from ordinary SEO, and the differences matter more than
the similarities.

## The channels, ranked by what they actually return

**1. An exact symbol in a search box.** Far and away the highest precision. An agent helping
someone debug `--spec-draft-n-max` searches for that literal string. It appears in perhaps a few
hundred documents on the public web, so anything containing it and also containing a measurement
ranks immediately. The same holds for issue and PR numbers, architecture strings from GGUF
metadata, and function names from the source. These tokens are nearly unique and cost nothing to
include honestly -- **for the ones this study actually used**. Two in the inventory below did
not: `--spec-draft-ngl` appears nowhere in this repository's code, and `draft-dspark` only inside
a docstring listing what `--spec-type` accepts. Including a token a study did not use is the
opposite of the point being made here.

**2. Being cited in the upstream thread.** An agent reading llama.cpp issue #27623 follows the
links in it. That is not SEO at all, it is participation, and it outperforms every keyword
decision on this page. `docs/UPSTREAM_CONTRIBUTIONS.md` tracks where this study has something a
thread actually needs. A comment with a reproduction on a second architecture is worth more than
any topic list.

**3. Natural-language search.** Agents issue questions, not keywords: "is speculative decoding
worth it on a 3090", "why did my tokens per second drop at long context". Question-shaped
headings match question-shaped queries, which is why the README's `## Findings` table is written
as questions with the answer beside each. This sentence named a `## What this answers` section
that the README does not have. An agent that has to read three
paragraphs to find out whether the answer is yes usually moves on.

**4. Semantic retrieval over chunks.** Where an index exists, the unit is a passage, not a page.
A passage that states its own subject survives chunking; one that says "as shown above" does not.
Concretely: repeat the model name and the flag inside the paragraph that reports the number,
rather than relying on a heading three levels up.

**5. GitHub topics.** Real but modest. Topics feed GitHub's own repo search and its "related
repositories", not general web search. Twenty is the cap, they are cheap, and the marginal ones
still cost nothing.

## What an agent actually types

Grouped by the situation the person is in, because that is what determines the vocabulary.

*Deciding whether to bother:* is speculative decoding worth it, does MTP actually speed up
llama.cpp, speculative decoding speedup benchmark rtx 3090, qwen3 tokens per second 3090.

*Tuning a flag they already found:* spec-draft-n-max best value, draft-mtp vs draft-dflash,
optimal draft length speculative decoding, spec-draft-p-min.

*Something is wrong:* llama.cpp slow long context, decode throughput collapse 80k, speculative
decoding slower than baseline, MoE speculative decoding net loss, why is my output different
with speculative decoding.

*Doing their own study:* speculative decoding energy per token, tokens per joule llm inference,
speculative decoding acceptance rate measurement, llama.cpp benchmark methodology.

*Following a thread:* llama.cpp 27342, DFlash2 benchmark, llama.cpp 27623, llama.cpp 25618
divergence.

The fourth and fifth groups are where this repo is strongest and where competition is thinnest.
Nothing in the prior-art sweep published an energy figure for this model. The long-context and
MoE-versus-MTP cells were open when this was written and have since been filled -- Phase L is a
documented failure to reproduce the cliff on sm_86, Phase M is complete with its causal reading
withheld -- and a negative or withheld result is still the only published answer in either.

## Exact-match inventory

The tokens worth keeping present in the README, because each is close to unique and each is
something this study really touched. This is a checklist, not a block to paste anywhere.

Flags and spec types: `--spec-type`, `draft-mtp`, `draft-dflash`, `draft-simple`, `draft-dspark`,
`--spec-draft-n-max`, `--spec-draft-p-min`, `--spec-draft-ngl`, `-ngld`, `-ctk q8_0`, `-fa on`.

Upstream threads: llama.cpp PR #27342 (DFlash2), issue #25618 (speculative decoding changes
output), issue #26750 (divergence on Blackwell), issue #27623 (long-context decode collapse),
vllm issue #38182 (prefix cache interacts with MTP).

Model and metadata strings, as they appear in the GGUF: `Qwen3.8-27B`, `Qwen3.6-35B-A3B`,
`qwen35`, `qwen35moe`, `nextn_predict_layers`, `blk.N.nextn.eh_proj.weight`,
`full_attention_interval`, `UD-Q4_K_XL`, `Q4_K_M`, Gated DeltaNet.

Hardware: RTX 3090, GA102, sm_86, 936 GB/s, 24 GB, consumer Ampere; a second RTX 3090 and an
RTX A6000 48 GB, which carry the cross-host replication and the forced-warp intervention rather
than being planned.

Measurement vocabulary: tokens per joule, tok/J, acceptance rate, mean accepted length,
verification step, draft acceptance, decode throughput, memory-clock elasticity, SM-clock
elasticity.

That last pair used to read "memory-bandwidth bound, compute bound", which is the third entry on
this page to name something the study may not claim. `evidence/registry.json` forbids a roofline
or per-kernel bottleneck attribution for both Phase R and Phase R2, and `TODO.md` records
"compute-bound verify" being downgraded from a conclusion to a described observation. The
elasticities are what was measured; the bound is what was refused.

Absent from the README, re-derived by grep on 2026-09-01: `draft-simple`, `draft-dspark`,
`--spec-draft-p-min`, `--spec-draft-ngl`, `sm_86`, `GA102`, `936 GB/s`, `qwen35moe`. Issue #26750
was on this list and is in the README now. **This list goes stale silently, so re-derive it rather
than trusting it**, and note that two of its entries name things this study did not use.

## Topics

Twenty, the GitHub maximum. Chosen so that each one is a term someone would plausibly search on
its own, rather than a synonym of a term already present.

```
speculative-decoding  llama-cpp             qwen                  qwen3
multi-token-prediction mtp                  rtx-3090              cuda
ampere                llm-inference         local-llm             gpu-benchmark
reproducible-research energy-efficiency     inference-optimization gguf
long-context          mixture-of-experts    kv-cache              quantization
```

`mtp` alongside `multi-token-prediction` is deliberate: the abbreviation is what appears in flag
names and issue titles, and GitHub topic matching does not expand it. `ampere` catches people
searching by architecture rather than by card. `energy-efficiency`, `long-context`,
`mixture-of-experts` and `quantization` each correspond to a phase of this study rather than to
an aspiration; they come off the list if the phase does not produce anything.

Set them with:

```bash
gh repo edit thc1006/qwen3.8-speculative-decoding-rtx3090 \
  --add-topic qwen3 --add-topic mtp --add-topic ampere \
  --add-topic energy-efficiency --add-topic inference-optimization \
  --add-topic gguf --add-topic long-context \
  --add-topic mixture-of-experts --add-topic kv-cache --add-topic quantization
```

## The About box, which this page used to leave out

The 350-character description is the one line GitHub prints beside the repository in every
search result, and it is the `og:description` a link preview and a retrieval agent both read
first. It ranked nowhere on the list above until 2026-09-01, which was an omission: it is
stronger than topics and costs a tenth as much thought.

Two properties decide what goes in it. Truncation is from the end, so the first 150 characters
carry the load. And it is one of the few places where the exact-match tokens of channel 1 and
the number of the last section can sit in the same sentence.

What replaced the earlier line, and why:

| change | reason |
|---|---|
| `n-max 2` became `--spec-draft-n-max 2` | channel 1 wants the literal string someone types into a search box, and `n-max 2` matches nothing |
| `MTP` and `DFlash2` became `draft-mtp` and `draft-dflash` | the same, for the `--spec-type` values |
| added `sm_86` | listed as absent from the README by the inventory above, and it is what someone with a 3090 searches when a report names another architecture |
| added `Preregistered, DOI-archived` | the strongest audience segment here is people doing their own study, and that is what tells them whether this is worth reading |
| `~35-37% lower energy` became `Decode energy -37.1%` | a number is retrievable; a range with a tilde is not, and the sensitivity band belongs in `docs/ENERGY.md` where it can be qualified |
| added `not separated from it` | the README scores the two methods as not established against each other, and two numbers side by side read as a ranking without it |

The website field takes the concept DOI rather than the repository URL: it is the identifier
that resolves to whichever version is newest, and a reader who wants the code is already here.

Topics are at the cap and each is justified above; nothing was traded.

**The largest lever is still not on this page.** Channel 2 -- a comment in the upstream thread
that a maintainer needs -- outperforms every field in the About box, and
`docs/UPSTREAM_CONTRIBUTIONS.md` tracks where this study has something a thread actually wants.

## The failure mode worth avoiding

Keyword stuffing backfires specifically for agent retrieval, and the reason is structural. The
agent does not rank the page and hand over a link. It reads the page and writes a summary. A
document padded with terms it cannot substantiate produces a summary that says the repo discusses
many topics, which is the least useful thing an agent can report. A document that says the built-in
MTP head at `--spec-draft-n-max 2` gives 59.8 % more decode throughput on an RTX 3090, measured
over 25 prompts, produces a summary containing that sentence. The count that belongs here is the
prompt count: `README.md` states that the inferential unit is the prompt and that 875 is not a
sample size.

The number is the retrievable object. The keyword is only how the number gets found.

Two consequences worth holding to. Put the claim in the first sentence under its heading, because
truncation happens from the end. And keep the negative results visible, since "llama.cpp
speculative decoding slower" is a real query with real volume, and a repo that answers it honestly
is more useful to the person asking than one that only reports wins.
