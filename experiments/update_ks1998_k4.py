from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "KS1998.ipynb"
nb = nbformat.read(NOTEBOOK, as_version=4)

assert "### 3.1 Parameters" in nb.cells[7].source
assert "Base.@kwdef struct KSParams" in nb.cells[8].source
assert "### 5.2 Euler loss" in nb.cells[19].source
assert "function aio_loss_components" in nb.cells[21].source
assert "function train!" in nb.cells[30].source

nb.cells[7].source = nb.cells[7].source.replace(
    "| future branches | 2 per economy | AiO cross product |",
    "| future branches $K$ | 4 per economy | unbiased U-statistic; $K=2$ recovers AiO |",
)

nb.cells[8].source = nb.cells[8].source.replace(
    "    hidden::Int = 32\n",
    "    hidden::Int = 32\n    shock_draws::Int = 4\n",
).replace(
    "par = KSParams(agents=50, hidden=32, fb_convention=:released_code)",
    "par = KSParams(agents=50, hidden=32, shock_draws=4, fb_convention=:released_code)",
)

nb.cells[17].source = nb.cells[17].source.replace(
    '@printf("parameter count: %d\\n", nparams(par))',
    '@printf("parameter count: %d; future shock draws K=%d\\n", nparams(par), par.shock_draws)',
)

nb.cells[19].source = r"""### 5.2 Euler loss and the $K$-shock unbiased estimator

For one future shock vector $\epsilon$, define

$$
R_t^i(\epsilon;\theta)
=\beta
\left(\frac{c_t^i}{c_{t+1}^i(\epsilon)}\right)^\gamma
[1-\delta+r_{t+1}(\epsilon)]
-\mu_t^i.
$$

The model requires

$$
E[R_t^i(\epsilon;\theta)\mid s_t]=0.
$$

Squaring one realized residual would incorrectly penalize shock variance. With
$K\geq2$ independent future shocks, use the unbiased U-statistic

$$
\widehat U_K^i
=\frac{
\left(\sum_{k=1}^K R_k^i\right)^2
-\sum_{k=1}^K(R_k^i)^2
}{K(K-1)}.
$$

Conditional independence gives

$$
E[\widehat U_K^i\mid s_t]
=\left(E[R_t^i\mid s_t]\right)^2.
$$

When $K=2$, this reduces exactly to the original AiO product $R_1^iR_2^i$.
The executable run uses $K=4$ to reduce Monte Carlo variance while preserving
the same population target. The mini-batch loss is

$$
\widehat{\mathcal L}(\theta)
=\frac1{BN}\sum_{b=1}^B\sum_{i=1}^N
\left[(\Phi_b^i)^2+\widehat U_{K,b}^i\right].
$$

The finite U-statistic can still be negative, but its conditional population
target is nonnegative."""

nb.cells[20].source = r"""### 5.3 Implement shocks, prices, future branches, and the loss

The code follows five steps.

1. Evaluate the current policy $(\xi_t,\mu_t,c_t,k_{t+1})$.
2. Draw $K$ conditionally independent future shock vectors.
3. For each branch, construct productivity, aggregate savings, prices, and next cash on hand.
4. Evaluate the same network at every future state and form $R_k^i$.
5. Combine the Fisher--Burmeister MSE with the $K$-shock U-statistic.

The first shock branch advances the endogenous training state. Separate random-number
streams are used for each branch, so branch 1 is unchanged when $K$ changes. This
makes controlled $K=2$ versus $K=4$ comparisons use the same exogenous state path."""

loss_start = nb.cells[21].source.index('"""AiO objective components')
nb.cells[21].source = nb.cells[21].source[:loss_start] + r'''"""AiO objective components for K conditionally independent shock branches."""
function aio_loss_components(theta, state, shocks::AbstractVector,
                             p::KSParams, ss=steady_state(p))
    K = length(shocks)
    K >= 2 || throw(ArgumentError("shock_draws must be at least 2"))
    current = policy(theta, state.w, state.y, state.z, p, ss)
    branches = [_future_branch(theta, current, state.y, state.z, shock, p, ss)
                for shock in shocks]
    residuals = [branch.euler for branch in branches]
    residual_sum = reduce(+, residuals)
    residual_square_sum = reduce(+, (residual .^ 2 for residual in residuals))
    euler = mean((residual_sum .^ 2 .- residual_square_sum) ./ (K * (K - 1)))

    fisher_burmeister = complementarity_residual(current, state, p)
    complementarity = mean(fisher_burmeister .^ 2)
    return (; loss=complementarity + euler, complementarity, euler,
            current, branches)
end

aio_loss(theta, state, shocks::AbstractVector, p::KSParams, ss=steady_state(p)) =
    aio_loss_components(theta, state, shocks, p, ss).loss'''

nb.cells[22].source = r'''rng_loss = MersenneTwister(7)
initial_shocks = [draw_shocks(rng_loss, size(state0.w, 1), par)
                  for _ in 1:par.shock_draws]
initial_loss = aio_loss_components(theta, state0, initial_shocks, par, ss)

@printf("initial total loss = %.6e\n", initial_loss.loss)
@printf("  Fisher--Burmeister term = %.6e\n", initial_loss.complementarity)
@printf("  K=%d Euler U-statistic = %.6e\n", par.shock_draws, initial_loss.euler)'''

nb.cells[23].source = nb.cells[23].source.replace(
    r"\mathcal D_b=\n(s_{t,b},\epsilon_{A,b}^y,\epsilon_{A,b}^z,\n\epsilon_{B,b}^y,\epsilon_{B,b}^z).",
    r"\mathcal D_b=(s_{t,b},\epsilon_{1,b},\ldots,\epsilon_{K,b}).",
).replace(
    "Each economy contains $N$ households and two independent future shock branches.",
    "Each economy contains $N$ households and $K=4$ independent future shock branches.",
)

nb.cells[27].source = r'''# Inspect one simulated training observation before optimization.
preview_state = simulate_training_state(theta, par, ss;
    periods=30, batch_size=4, seed=2025)
preview_rng = MersenneTwister(2026)
preview_shocks = [draw_shocks(preview_rng, 4, par) for _ in 1:par.shock_draws]
preview = aio_loss_components(theta, preview_state, preview_shocks, par, ss)

b, i = 1, 1
@printf("batch shapes: w=%s, y=%s, z=%s\n",
    string(size(preview_state.w)), string(size(preview_state.y)), string(size(preview_state.z)))
@printf("current state: w=%.4f, y=%.4f, z=%.4f\n",
    preview_state.w[b,i], preview_state.y[b,i], preview_state.z[b])
@printf("current policy: xi=%.4f, mu=%.4f, c=%.4f, k'=%.4f\n",
    preview.current.xi[b,i], preview.current.mu[b,i],
    preview.current.c[b,i], preview.current.kp[b,i])
for branch in 1:min(2, par.shock_draws)
    @printf("branch %d: next w=%.4f, Euler R=%.4f\n", branch,
        preview.branches[branch].prices.wealth_next[b,i],
        preview.branches[branch].euler[b,i])
end'''

nb.cells[28].source = r"""## 7) Stochastic Training

At period $n$:

1. draw $K=4$ independent future shock batches;
2. compute the unbiased U-statistic loss;
3. differentiate it with automatic differentiation;
4. update $\theta$ with Adam every second simulated period;
5. advance the state with branch 1;
6. repeat with new shocks.

The NN receives no target consumption data. Increasing $K$ reduces conditional-shock
Monte Carlo variance; it does not eliminate variation from the evolving simulated state."""

nb.cells[29].source = r"""### 7.1 Adam and the complete training loop

`AdamState` stores the first and second gradient moments. The training loop uses one
independent RNG stream per future branch. This ensures that the branch driving the
state transition does not change merely because the number of integration branches
changes. The executable run advances 30,000 simulated periods and performs 15,000
Adam updates.

The recorded U-statistic remains signed and noisy. Convergence is therefore assessed
with rolling summaries and an independently simulated validation-state bank rather
than the instantaneous absolute loss alone."""

nb.cells[30].source = r'''mutable struct AdamState
    m::Vector{Float64}
    v::Vector{Float64}
    t::Int
end

AdamState(n::Int) = AdamState(zeros(n), zeros(n), 0)

function _adam_step!(theta, grad, state::AdamState; lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-7)
    state.t += 1
    state.m .= beta1 .* state.m .+ (1 - beta1) .* grad
    state.v .= beta2 .* state.v .+ (1 - beta2) .* (grad .^ 2)
    mhat = state.m ./ (1 - beta1^state.t)
    vhat = state.v ./ (1 - beta2^state.t)
    theta .-= lr .* mhat ./ (sqrt.(vhat) .+ eps)
    return theta
end

"""Train on evolving simulated states using a K-shock unbiased U-statistic."""
function train!(theta, p::KSParams, ss=steady_state(p);
                periods=30000, batch_size=10, update_every=2, lr=1e-3,
                seed=2026, display_step=1000)
    p.shock_draws >= 2 || throw(ArgumentError("shock_draws must be at least 2"))
    branch_rngs = [MersenneTwister(seed + 100_000 * branch)
                   for branch in 1:p.shock_draws]
    state = initial_state(batch_size, p, ss)
    optimizer = AdamState(length(theta))
    loss_history = Float64[]
    complementarity_history = Float64[]
    euler_history = Float64[]

    for period in 1:periods
        shocks = [draw_shocks(branch_rngs[branch], batch_size, p)
                  for branch in 1:p.shock_draws]
        components = aio_loss_components(theta, state, shocks, p, ss)
        if (period - 1) % update_every == 0
            loss_function = th -> aio_loss(th, state, shocks, p, ss)
            gradient = Flux.gradient(loss_function, theta)[1]
            all(isfinite, gradient) || error("non-finite gradient at period $period")
            _adam_step!(theta, gradient, optimizer; lr=lr)
        end

        push!(loss_history, components.loss)
        push!(complementarity_history, components.complementarity)
        push!(euler_history, components.euler)
        first_branch = components.branches[1]
        state = (; w=first_branch.prices.wealth_next,
                 y=first_branch.y_next, z=first_branch.z_next)

        if period == 1 || period % display_step == 0
            println("period $(lpad(period, 5)) | updates=$(lpad(optimizer.t, 5)) " *
                    "| loss=$(round(components.loss, sigdigits=5)) " *
                    "| FB=$(round(components.complementarity, sigdigits=4)) " *
                    "| K=$(p.shock_draws) Euler U=$(round(components.euler, sigdigits=4))")
        end
    end

    return (; theta, state, updates=optimizer.t, loss_history,
            complementarity_history, euler_history)
end'''

nb.cells[32].source = r"""## 8) Diagnostics

The instantaneous U-statistic is unbiased but remains noisy and may be negative.
The notebook therefore reports:

- rolling absolute magnitudes and the signed rolling Euler estimate;
- conditional Euler errors on an independently simulated state bank;
- Fisher--Burmeister errors;
- an independent aggregate simulation;
- policy-function slices and approximate aggregation.

The validation bank contains 20 economies generated for 500 periods with a seed not
used in training. At each saved final state, the conditional Euler residual is
integrated using 128 additional shocks. This is still simulation-based, but both the
state path and future shocks are held out from training."""

nb.cells[34].source = r'''function rolling_mean(values, window)
    x = Float64.(values)
    prefix = cumsum(vcat(0.0, x))
    return [(prefix[i + 1] - prefix[max(1, i - window + 1)]) /
            (i - max(1, i - window + 1) + 1) for i in eachindex(x)]
end

loss_window = 500
mean_abs_total = rolling_mean(abs.(result.loss_history), loss_window)
mean_fb = rolling_mean(result.complementarity_history, loss_window)
mean_abs_euler = rolling_mean(abs.(result.euler_history), loss_window)
mean_signed_euler = rolling_mean(result.euler_history, loss_window)

p_magnitude = plot(mean_abs_total .+ 1e-14,
    yscale=:log10, lw=2, label="rolling mean |total|",
    xlabel="simulated period", ylabel="rolling magnitude (log scale)")
plot!(p_magnitude, mean_fb .+ 1e-14, lw=2, label="rolling FB MSE")
plot!(p_magnitude, mean_abs_euler .+ 1e-14,
    lw=2, label="rolling mean |Euler U|")
title!(p_magnitude, "K=$(par.shock_draws) objective, window=$loss_window")

p_signed = plot(mean_signed_euler, lw=2, label="signed rolling Euler U",
    xlabel="simulated period", ylabel="signed estimate")
hline!(p_signed, [0.0], ls=:dash, color=:black, label="zero")
title!(p_signed, "Signed estimator reveals cancellation")

plot(p_magnitude, p_signed, layout=(1, 2), size=(1000, 360))'''

nb.cells[35].source = r"""The controlled experiment in `experiments/ks1998_kshock_experiment.jl`
found that $K=4$ reduced the repeated Euler-loss standard deviation by about 43--44%
and increased gradient SNR by roughly 50--60%. The K=2 gradient SNR remained above
six, so the gradient was noisy but not noise-dominated. K=4 improved median through
p95 fixed-bank Euler quantiles in that run, while one extreme residual increased;
the full residual distribution remains important."""

nb.cells[36].source = r'''# Independent validation states and independent future shocks.
validation_state = simulate_training_state(result.theta, par, ss;
    periods=500, batch_size=20, seed=9090)
diagnostics = conditional_residuals(result.theta, validation_state, par, ss;
    n_shocks=128, seed=99)
log_euler_error = log10.(abs.(vec(diagnostics.euler)) .+ 1e-12)
fb_error = abs.(vec(diagnostics.fisher_burmeister))

@printf("validation economies = %d; household states = %d\n",
    size(validation_state.w, 1), length(validation_state.w))
@printf("mean log10 Euler error = %.4f\n", mean(log_euler_error))
@printf("median log10 Euler error = %.4f\n", median(log_euler_error))
@printf("p90 log10 Euler error = %.4f\n", quantile(log_euler_error, 0.90))
@printf("p95 log10 Euler error = %.4f\n", quantile(log_euler_error, 0.95))
@printf("mean absolute FB residual = %.4e\n", mean(fb_error))

p_euler = histogram(log_euler_error, bins=25, label=false,
    xlabel="log10 absolute conditional Euler residual",
    ylabel="held-out household states",
    title="Independent-state Euler errors")
p_euler'''

nbformat.write(nb, NOTEBOOK)
print(f"Updated {NOTEBOOK} for configurable K=4 training")
