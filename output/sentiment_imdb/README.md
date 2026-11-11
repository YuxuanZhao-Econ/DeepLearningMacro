# From-scratch IMDb review classifier

This is the earlier 577,601-parameter baseline. The current self-contained notebook uses
a larger classifier and writes its results to `output/nlp_course/`. The baseline is retained
for reference; the commands below reproduce this earlier experiment.
All model training, tokenization, vocabulary fitting, and evaluation use Julia.
The offline public-data transport helper uses Python/Arrow only to decode Parquet.

The mean-embedding MLP has 577,601 parameters: a random 48 × 12,000 embedding table,
32 ReLU hidden units, and one logit. Eight epochs use Adam, batch size 128,
learning rate 0.003, initialization seed 2718, and split seed 31415.
No pretrained model or embedding is used.

The initial standalone run took approximately 71.3 seconds including Julia startup,
compilation, and data preparation. Its fit function took 13.4 seconds.
This is a measured local run, not a runtime guarantee for other machines.
Subsequent notebook runs record their own fit time.

The best validation-loss epoch was 2. On the 24,675 filtered test reviews:

- Accuracy: 87.4732%.
- Binary cross-entropy: 0.304923.
- Positive-class F1: 0.869694.
- Confusion matrix, actual rows / predicted columns, negative then positive:
  `[[11269, 995], [2096, 10315]]`.

Later epochs reduce training loss but increase validation loss. The notebook keeps this
overfitting behavior visible, uses validation to select the model, and shows fixed-order
successful/failed test examples plus a word-order counterexample.
No hyperparameter tuning on test results was used for this run.

## Run it

```text
julia --startup-file=no --project=. experiments/train_sentiment.jl
```

The earlier notebook used `RUN_SENTIMENT_TRAINING`. The current notebook uses
`TRAIN_SENTIMENT` for its new larger model and `TRAIN_LARGE_GPT` for the optional GPU run.

## Files

- `run.jls`: selected and initial parameter arrays, vocabulary, split metadata/hashes,
  configuration, per-epoch history, test predictions and metrics. No optimizer resume state.
- `metrics.csv`: the latest executed fit's training/validation curve.
- `summary.txt`: its selected epoch, actual fit duration and test metrics.
- `process_time.txt`: standalone trainer process duration; notebook runs do not rewrite it.

Data details and reproducibility: `reference/datasets/imdb/README.md`.
