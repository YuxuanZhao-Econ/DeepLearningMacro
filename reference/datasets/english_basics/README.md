# Simple English foundation corpus

Source: [Ronen Eldan and Yuanzhi Li's TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
described in [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759).
The source dataset is synthetic text. This experiment uses the GPT-4-only V2 text files;
it downloads no teacher/student model weights. The corpus and these prepared text/token
derivatives are made available under **CDLA-Sharing-1.0**:
<https://cdla.dev/sharing-1-0/>. The source dataset card is preserved in
`TINYSTORIES_UPSTREAM_README.md`. Local transformations do not constitute source-author endorsement.

`experiments/download_english_basics.py` downloads bounded prefixes (64 MiB training,
4 MiB official validation), stopping at the last complete `<|endoftext|>` separator.
`download_manifest.json` records exact URLs, sizes, and hashes. These prefixes are
not claimed to be a random or representative sample of the entire released corpus.

`experiments/prepare_english_basics.py` normalizes punctuation/whitespace to ASCII,
keeps 200–7,000-character stories, and removes exact normalized duplicates. It also
excludes any official-validation story matching training. The remaining official
validation prefix is shuffled with seed 20261028 and halved into our validation/test
sets. Semantic near-duplicates and paraphrases may remain.

The result is 81,694 training stories (65,800,399 normalized characters), 2,581 validation
stories, and 2,581 test stories. The corrected BPE stream has 17,656,261 training tokens,
including beginning/end markers. No jokes enter the model's training data or tokenizer.

The 2,048-token BPE vocabulary is fit only on the first 16,777,701 normalized training
characters with `tokenizers==0.22.2`. A word/number/punctuation pre-tokenizer preserves
an optional leading space and prevents merging a word with its following space.
Julia implements the saved merge rules directly and checks against reference encodings.
IDs are **one-based** in exported files. `*.bin` stores little-endian UInt16 IDs;
`tokens_hex.txt` preserves literal spaces and newlines, and `merges.tsv` stores the
ordered left/right/result ID triples. `tokenizer.json` is the reference tokenizer.

To rebuild the prepared data on this computer:

```powershell
C:/Anaconda/python.exe -m pip install --no-deps --only-binary=:all: --target tmp/english_bpe_python_deps tokenizers==0.22.2
C:/Anaconda/python.exe experiments/download_english_basics.py
C:/Anaconda/python.exe experiments/prepare_english_basics.py
```

The small tokenizer dependency is isolated from the existing Python installation.
The supplied prepared files let Julia training and notebook exploration run without Python.
Rebuilding the tokenizer can change tie-breaking/IDs; retain the saved tokenizer to
reuse a checkpoint, and verify its hash. `manifest.json` records the exact prepared artifacts.
