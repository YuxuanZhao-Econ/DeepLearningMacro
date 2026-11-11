# Local from-scratch joke language model

Start with `notebooks/Intro_Jokes_GPT.ipynb` in the repository's Julia kernel.
The default notebook path loads CPU weights and does not rerun GPU training.

- `initial.jls`: seeded random initialization, before any optimizer update.
- `best.jls`: CPU model selected by validation loss, with vocabulary, configuration,
  full training/validation history, source hashes, and run metadata.
- `metrics.csv`: actual fixed-window losses measured during GPU training.
- `run_summary.txt` / `.jls`: measured elapsed time, selected step, final test score,
  baseline scores, device, and stopping reason.
- `verification.txt` / `verification_probe.jls`: fresh-process loading and numerical
  checks. The saved probe can be compared in another fresh process.

All weights were initialized randomly. This is an 842,496-parameter character GPT,
not GPT-2 pretrained weights, a fine-tune, or a general conversational assistant.
The 192-character context is approximately a short joke, not a long chat history.

To train a fresh model from the repository root:

```text
julia --startup-file=no --project=. experiments/train_jokes_gpt.jl
```

The default maximum is 16,000 steps, with validation-based early stopping and a
45-minute training-function wall-clock budget (package loading precedes that timer).
This command replaces this local experiment's checkpoints and logs. The checkpoint
does not include optimizer state for exact continuation of an interrupted run.
Julia serialization is intended for this recorded Julia/Flux project: load
`src/JokesGPT.jl` before deserialization.

Checkpoint selection uses validation windows. Test windows are evaluated only after
selection. The matched bigram baseline is fit on the training stream and evaluated
on the same fixed windows as the GPT. Test perplexity is per character; it is not
a humor score or comparable to subword-token perplexity.
