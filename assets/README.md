# Filler corpus

Three public-domain books, used only as context filler for the depth ladder in Phase L. Nothing
here is generated, summarised, or scored; the text exists to occupy KV positions.

| file | work | source |
|---|---|---|
| `2701-0.txt` | Melville, *Moby-Dick* | Project Gutenberg ebook 2701 |
| `pg1342.txt` | Austen, *Pride and Prejudice* | Project Gutenberg ebook 1342 |
| `pg2600.txt` | Tolstoy, *War and Peace* | Project Gutenberg ebook 2600 |

```
0670d7bb10b99d05f095a28942801aa74d4921d1b34dbdc76900e2c4c2bd2189  2701-0.txt
74f2665d6e6925fc2c17dec644bec9e87df478a0f1836822125e8acbb3777806  pg1342.txt
2d5bb2ad5f422765e714617e21fa31bbaf8958aa79682c86fca6660fcc5d1b2b  pg2600.txt
```

The bytes are committed rather than fetched at run time, because the drafter's acceptance rate
depends on the exact tokens it sees. A re-download that differs by so much as a line ending
would change the filler and therefore the measurement, and the difference would not be visible
in the result file. `harness/filler.py` strips the Gutenberg header and licence block at load
time and asserts that no marker survives; what reaches the model is the prose alone.

Real prose is used rather than a repeated paragraph on purpose. Repetition is exactly what a
speculative drafter predicts best, so filling context with it would raise acceptance for a
reason that has nothing to do with depth, which is the variable under test.
