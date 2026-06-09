# Data layout

The `data/` directory is expected to contain the same CSV files used by
the v1 codebase. The simplest setup is:

```bash
make data
```

This will symlink the CSVs from `../Multitask\ BERT/source_code/data/`
into `data/`. If that directory is not available, place the files
manually:

| File | Columns (header) |
|---|---|
| `ids-sst-train.csv`, `ids-sst-dev.csv` | `id, sentence, sentiment` |
| `ids-sst-test-student.csv` | `id, sentence` |
| `quora-train.csv`, `quora-dev.csv` | `id, sentence1, sentence2, is_duplicate` |
| `quora-test-student.csv` | `id, sentence1, sentence2` |
| `sts-train.csv`, `sts-dev.csv` | `id, sentence1, sentence2, similarity` |
| `sts-test-student.csv` | `id, sentence1, sentence2` |

Tab-separated `.tsv` is also supported (the data loader auto-detects by
suffix).

## Sizes (rough)

| Task | Train | Dev | Test (student) |
|---|---|---|---|
| SST-5 | ~8,500 | ~1,100 | ~2,200 |
| Quora | ~140,000 | ~24,000 | ~80,000 |
| STS-B | ~5,700 | ~1,500 | ~1,400 |

Quora dominates. With `training.strategy: round_robin` we under-use Quora
because each "round" advances all loaders by one batch. Use
`interleaved.yaml` to consume Quora more aggressively.

## Preprocessing

Light normalisation is applied in `_normalise()` (lowercase + space-pad
punctuation). This matches v1's `preprocess_string`. Tokenization is
delegated to the `transformers` `BertTokenizer`.
