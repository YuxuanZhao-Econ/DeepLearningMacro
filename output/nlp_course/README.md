# Results from the self-contained NLP course

Open `notebooks/Intro_English_GPT.ipynb` with the project's Julia environment and run cells
in order. All model, tokenizer, gradient, training, sampling, and plotting implementations
are present in the notebook. Packages, public datasets, BPE rules, and recorded model arrays
remain project assets. Data sources and preparation instructions are under `reference/datasets/`.

The notebook has 74 cells, including 32 executed Julia cells and 15 figures. It uses the full
IMDb classifier from the first example, tracing one actual 115-token training review through
lookup, coordinatewise mean pooling, prediction, loss, gradients, and one parameter update.
It then derives next-token prediction, Q/K/V attention, and residual connections in the full
5,820,416-parameter GPT. Smaller demonstration models are no longer part of this course.

## Current results

The IMDb classifier has 1,284,225 parameters: 20,000 vocabulary entries, 64 embedding
coordinates, and 64 hidden units. Eight CPU epochs selected epoch 1 by validation loss.
On 24,675 filtered test reviews, accuracy was 87.78%, positive-class F1 was 0.8730, and
binary cross-entropy was 0.2969. Actual-label rows and predicted-label columns, negative
then positive, give `[[11297, 967], [2049, 10362]]`.
The fit takes roughly ten seconds on this machine; startup and data preparation are additional.
Exact duplicate handling and the training/validation/test split are explained in the notebook.

On 4,096 sampled held-out targets, the full GPT at a fresh random initialization has loss 7.7140
and perplexity 2239.55; its trained counterpart has loss 1.8635 and perplexity 6.45.
The tokenizer, architecture, and evaluated windows are identical. This random baseline is not
claimed to be the original stored initialization of the earlier run.
The trained model was fitted earlier from random initialization on this computer;
its original approximately 38-minute GPU run is documented in `output/english_basics/`.
The notebook restores it from plain arrays and reproduces the saved reference logits exactly.
Freshly generated continuations are embedded without editing their text.

On November 4, 2026, the optional GPU training path was checked for two updates on the full
5,820,416-parameter configuration and an RTX 5070, including restoration to CPU arrays.
That check passed. It was not a new full 40-minute training experiment.
CUDA printed a runtime/precompile version warning (13.4 versus 13.2); the computation completed.

## Files and switches

- `sentiment.jls`: selected classifier arrays, random initial arrays, vocabulary, history,
  configuration, and test metrics.
- `small_gpt.jls`: retained result from the previous course version; the current notebook does not use it.
- `english_samples.jls`: current fixed-prompt continuations from the larger GPT.
- `validation.json`: exact-cell execution, structural checks, protected-file hashes, and visual review.
- `gpu_check.txt`: the bounded two-update GPU check.

`TRAIN_SENTIMENT=true` retrains the IMDb model; false loads its saved result.
`TRAIN_LARGE_GPT=false` uses the recorded
larger model; setting it to true starts a fresh fit with a 40-minute budget and saves
`fresh_large_gpt.jls` here. Budgets are checked between updates and are not hard process timeouts.
The GPT's time-based learning-rate schedule can produce slightly different results
when execution speed changes, even with the same random seed.

Validation executed the exact saved Julia cells sequentially in a fresh Julia process and
embedded their actual outputs. This was not an IJulia-kernel execution. The manual classifier
and attention derivatives matched automatic differentiation, and future-token isolation passed.
The inspected head's query gradients also matched gradients of the whole GPT's next-token loss.
The classifier gradient error was zero; the attention error was about 1.19e-7 in Float32.
Each figure and rendered formula sections were visually inspected. The older notebooks and
original English-model checkpoints were preserved.
