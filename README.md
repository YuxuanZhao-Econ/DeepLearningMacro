# Deep Learning Macro

This repository contains Julia notebooks for learning deep learning methods used in macroeconomics. `notebooks/Intro_DL.ipynb` introduces the basic neural-network building blocks from scratch in Julia, `notebooks/Intro_NLP.ipynb` builds from tokens and embeddings to training a tiny GPT-style language model, `notebooks/RBC.ipynb` applies the Euler-equation method to a stochastic RBC/Brock-Mirman benchmark, and `notebooks/KS1998.ipynb` extends the method to the Krusell--Smith heterogeneous-agent economy.

## Requirements

The Julia environment is recorded in:

- `Project.toml`
- `Manifest.toml`

From the repository root, activate and instantiate the environment with:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

The project uses:

- `Flux`
- `IJulia`
- Julia standard libraries: `Random`, `LinearAlgebra`, `Statistics`, `Printf`, `Plots`

## Project Structure

- `notebooks/Intro_DL.ipynb`
  - introduces deep learning as function approximation for economic problems
  - builds neural-network components in Julia to learn a toy function

- `notebooks/Intro_NLP.ipynb`
  - introduces tokens, learned embeddings, position information, and causal attention with diagrams and small calculations
  - trains a two-layer, two-head GPT-style model from scratch on an embedded synthetic corpus using Julia and Flux on CPU
  - compares held-out prediction loss with a bigram baseline, generates short sentences, and checks that future tokens cannot affect earlier predictions
  - is self-contained, requires no additional project dependencies, and can be run from the repository root or `notebooks/` with the Julia kernel

- `notebooks/Intro_Jokes_GPT.ipynb`
  - an English introduction from characters, embeddings, and causal attention to a real public-corpus experiment
  - trains an 842,496-parameter GPT-style model from random weights on a small question-and-answer subset of Short Jokes
  - includes saved local GPU training results, diagrams, a held-out bigram comparison, fixed-seed samples, and a causal-mask check
  - loads the supplied CPU checkpoint by default; set `RUN_TRAINING = true` to train afresh on CUDA, with a 45-minute training-function budget
  - uses `src/JokesGPT.jl`, `experiments/train_jokes_gpt.jl`, and the provenance/split files in `reference/datasets/short_jokes/`
  - demonstrates joke completion; it does not establish general conversational ability

- `notebooks/Intro_English_GPT.ipynb`
  - the main English-language Julia NLP course, with a plain numbered layout matching `Intro_DL`, `RBC`, and `KS1998`
  - starts directly with the full IMDb project and traces one real 115-token training review through the same network
  - trains a 1,284,225-parameter IMDb classifier using a 20,000-entry vocabulary and 64-dimensional embeddings
  - defines every symbol and dimension before deriving lookup, coordinatewise mean pooling, neural prediction, loss, and parameter updates
  - derives Q/K/V attention in an actual 32-coordinate head of the full 256-wide GPT, including agreement with full-model backpropagation
  - explains residual connections inside the Transformer, then uses the 5,820,416-parameter English model throughout training and generation
  - defines all tokenizers, models, training loops, and plotting helpers directly in the notebook; only packages, public data, and recorded parameter arrays are external
  - uses `TRAIN_SENTIMENT` and the optional `TRAIN_LARGE_GPT` switches; the GPU switch defaults to false and loads the recorded from-scratch experiment
  - stores current course results in `output/nlp_course/`; earlier training artifacts and the 26-exchange reference dialogue remain available separately
  - authoring: `experiments/build_nlp_self_study.py` → `build_nlp_clean.py`; verification: `experiments/validate_english_notebook.py`

- `notebooks/RBC.ipynb`
  - approximates the consumption policy with a neural network, then solves the RBC benchmark
  - trains the policy by minimizing squared Euler-equation residuals on simulated states

- `notebooks/KS1998.ipynb`
  - approximates the consumption-share and borrowing-multiplier policies with a neural network, then solves the Krusell--Smith heterogeneous-agent economy
  - trains consumption-share and borrowing-multiplier policies with the two-shock AiO Euler objective

- `src/NetworkDiagram.jl`
  - contains the reusable neural-network architecture and training-flow diagrams

- `src/Diagnostics.jl`
  - contains supporting KS evaluation helpers so the teaching notebook can focus on the solution method


- `Project.toml`
  - Julia project dependencies

- `Manifest.toml`
  - pinned Julia dependency versions

- `reference/`
  - `JME2021.pdf`: Maliar, Maliar, and Winant (2021)
  - `Slides_JHU_*.pdf`: lecture slides on deep learning methods for dynamic economic models
  - local copy of the reference TensorFlow implementation
  - external teaching notes from Jesus Fernandez-Villaverde: <https://www.sas.upenn.edu/~jesusfv/teaching.html>

## Methods

### RBC benchmark

The RBC notebook parameterizes the consumption share as a neural-network policy:

```math
\xi_t=\xi(k_t,z_t;\theta).
```

Consumption and next-period capital are constructed to satisfy feasibility:

```math
c_t=\xi(k_t,z_t;\theta)e^{z_t}k_t^\alpha,
\qquad
k_{t+1}=\left[1-\xi(k_t,z_t;\theta)\right]e^{z_t}k_t^\alpha.
```

The training loss is the average squared Euler residual over simulated state-shock draws:

```math
\mathcal L(\theta)=\frac{1}{B}\sum_{b=1}^{B}R_b(\theta)^2.
```

Gradients are computed with `Flux.gradient`, and the parameter vector is updated with Adam.

## References and Resources

- Maliar, L., Maliar, S., and Winant, P. (2021). Deep learning for solving dynamic economic models. *Journal of Monetary Economics*, 122, 76-101.
- Reference TensorFlow implementation: [marcmaliar/deep-learning-euler-method-krusell-smith](https://github.com/marcmaliar/deep-learning-euler-method-krusell-smith).
- Jesus Fernandez-Villaverde teaching notes: <https://www.sas.upenn.edu/~jesusfv/teaching.html>.
