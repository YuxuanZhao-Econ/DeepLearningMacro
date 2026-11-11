# IMDb sentiment project

Source: [Stanford NLP IMDb dataset](https://huggingface.co/datasets/stanfordnlp/imdb),
originally released with Maas et al., *Learning Word Vectors for Sentiment Analysis* (2011).
The original labeled splits each have 25,000 English movie reviews.
Labels are 0 = negative and 1 = positive. The additional unlabeled data is not used.
The upstream card lists the license as `other`; see `UPSTREAM_DATASET_CARD.md`.

## Reproduce preparation

From the repository root, run `python experiments/prepare_imdb.py` using Python with Arrow.
The helper downloads only the public training/test Parquet files and dataset card, and writes
lossless TSV files: source row index, label, and hex-encoded UTF-8 original review text.
It does not tokenize, fit a vocabulary, split validation data, or train a neural network.
The observed source and transport-file SHA256 hashes are in `manifest.json`.
The Julia course consumes the supplied TSV files without requiring Python.

## Julia preparation

`src/SentimentLab.jl` removes HTML tags, lowercases, and extracts ASCII English words,
preserving apostrophe contractions and negation words. It does not use a sentiment dictionary.
It groups exact normalized token sequences in the original training pool, keeps one copy of
same-label groups, and removes conflicting-label groups. A seeded, label-stratified 10% split
forms validation. The vocabulary is fitted only on the remaining training portion.
The earlier standalone helper uses 12,000 entries including `<unk>`; the current notebook
uses 20,000. Both fit only training text and use alphabetical order to resolve frequency ties.
The current notebook includes this preparation logic directly rather than importing the helper.

Test data is normalized the same way, deduplicated, and filtered for any exact normalized
overlap with the entire original training pool. Final counts:

| Split | Reviews | Role |
|---|---:|---|
| Train | 22,411 | Fit vocabulary and model parameters |
| Validation | 2,490 | Choose epoch by cross-entropy |
| Test | 24,675 | Evaluate the selected model |

The procedure excludes 99 original training rows and 325 original test rows.
The exact source row IDs, texts, and labels remain available through the preparation function.
This is a documented filtered split, not an unmodified leaderboard evaluation.
Exact deduplication does not remove all possible near duplicates.

Mean pooling uses all tokenized words. A sparse normalized-frequency column computes the
same mean embedding as explicit lookup and averaging, without padding or truncation.
The model is deliberately insensitive to word order.
