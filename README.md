# Deep Learning Macro

This repository contains a Julia notebook for learning the Euler-equation deep learning method of Maliar, Maliar, and Winant (2021). The current notebook solves a stochastic RBC/Brock-Mirman benchmark with a neural-network policy rule trained by minimizing simulated Euler-equation residuals.

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
- `Plots`
- Julia standard libraries: `Random`, `LinearAlgebra`, `Statistics`, `Printf`

## Project Structure

- `RBC.ipynb`
  - solves the log-utility, full-depreciation RBC/Brock-Mirman benchmark
  - approximates the consumption policy with a neural network
  - trains the policy by minimizing squared Euler-equation residuals on simulated states
  - compares the learned policy with the analytical policy function

- `Project.toml`
  - Julia project dependencies

- `Manifest.toml`
  - pinned Julia dependency versions

- `reference/`
  - `JME2021.pdf`: Maliar, Maliar, and Winant (2021)
  - local copy of the reference TensorFlow implementation

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
