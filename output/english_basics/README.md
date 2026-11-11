# From-scratch simple English GPT

Read `notebooks/Intro_English_GPT.ipynb` using the repository's Julia kernel.
The integrated notebook first completes a sentiment-classification lesson and a real IMDb
CPU training project, then derives next-token prediction, Q/K/V attention, and residual
connections in the full 5,820,416-parameter GPT. The notebook uses saved parameter arrays,
recorded metrics, and fresh CPU generation; it does not restart GPU training by default.
The public training text is TinyStories only, including its character dialogue.

All executable tutorial cells use Julia. The complete Julia BPE encoder explains
the merge mechanism. The supplied full-corpus preparation artifacts were produced by
an offline Python/Rust helper; readers do not need to write Python to run the course.
The original `notebooks/Intro_NLP.ipynb` is retained separately.

The expanded principles include a complete review-classification forward/backward pass,
an IMDb training/evaluation project, full projection-matrix attention gradients, and decoding
with temperature and top-k; top-p is explained in prose. The current course keeps its main path concise.
The reference dialogue and extraction verification are in `reference/chat_records/`.

The model has 5,820,416 parameters, width 256, six GPT blocks with eight heads each,
and a 128-BPE-token context. Its 2,048-token vocabulary was learned from the local
training text. Neither model weights nor a pretrained tokenizer were downloaded.

Artifacts produced by training:

- `initial.jls`: the seeded random model before any optimization.
- `best.jls`: the CPU model selected by validation, vocabulary/configuration,
  full history, and provenance hashes.
- `metrics.csv`: fixed-window training/validation losses measured during training.
- `run_summary.txt` / `.jls`: measured duration, selected/last steps, final test loss,
  device, parameter count, and source/tokenizer hashes.
- `samples_step_*.txt` / `.jls`: fixed-prompt, fixed-seed GPU continuations during training.
- `samples.txt` / `.jls`: final GPU samples from the selected checkpoint.
- `verification_probe.jls`: CPU logits for comparison in a fresh process.
- `verification.txt`: fresh CPU loading, tokenizer, provenance, and causality checks.
- `notebook_weights.jls`: plain arrays, configuration, history, and reference logits, so the self-contained notebook can define its own model types.
- `notebook_validation.json`: exact-code execution method, counts, and notebook hashes.
- `curriculum_revision_validation.json`: sentiment-first reorganization checks, executed
  classifier results, Q/K/V gradient checks, figure inspection, and unchanged English-model hashes.
  Earlier `chat_revision_validation.json` records the preceding dialogue-based revision.

The notebook generates CPU samples directly. Small rounding differences between TF32
GPU matrix products and CPU arithmetic can change sampled tokens, even with equal seeds.
The tokenizer and weights remain the same. The notebook labels supplied prompts separately
from generated continuations, with the EOS/length-limit stopping condition shown.

To start again from random parameters:

```text
julia --startup-file=no --project=. experiments/train_english_basics.jl
```

The final experiment has a forty-minute process budget, including startup and compilation,
and reserves two minutes for saving/evaluation. An earlier tokenizer pilot was stopped;
its artifacts remain in `output/english_basics_tokenizer_pilot/`. We conservatively charge
ten minutes for that pilot, keeping the combined training budget below one hour.

Validation selects a checkpoint; test windows are evaluated once afterward. Loss/perplexity
are per BPE token and cannot be directly compared with the earlier character model's values.
The terminal validation between periodic logging intervals is saved in checkpoint history.
Its supplied-run learning-rate field uses zero as an unrecorded-value placeholder;
that placeholder should not be interpreted as an actual optimizer rate.
Fluent samples or a few agreement checks do not establish robust general conversation.
Checkpoints target the recorded Julia/Flux project and omit optimizer state for exact resumption.
Data provenance and reconstruction instructions are in `reference/datasets/english_basics/README.md`.

The current notebook defines all core functions inline. `TRAIN_LARGE_GPT=true` starts a new
40-minute-budget CUDA fit using its compact trainer; new results go to `output/nlp_course/`.
The standalone recipe above remains the recipe associated with the original recorded experiment.
