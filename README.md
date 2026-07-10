# Deep Learning Macro

This repository contains Julia notebooks for learning deep learning methods used in macroeconomics. `notebooks/Intro_DL.ipynb` introduces the basic neural-network building blocks from scratch in Julia, while `notebooks/RBC.ipynb` applies the Euler-equation deep learning method of Maliar, Maliar, and Winant (2021) to a stochastic RBC/Brock-Mirman benchmark.

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

- `notebooks/RBC.ipynb`
  - approximates the consumption policy with a neural network, then solves the RBC benchmark
  - trains the policy by minimizing squared Euler-equation residuals on simulated states


- `Project.toml`
  - Julia project dependencies

- `Manifest.toml`
  - pinned Julia dependency versions

- `reference/`
  - `JME2021.pdf`: Maliar, Maliar, and Winant (2021)
  - `Slides_JHU_*.pdf`: lecture slides on deep learning methods for dynamic economic models
  - local copy of the reference TensorFlow implementation
  - external teaching notes from Jesus Fernandez-Villaverde: <https://www.sas.upenn.edu/~jesusfv/teaching.html>

## Method

The notebook parameterizes the consumption share as a neural-network policy:

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
