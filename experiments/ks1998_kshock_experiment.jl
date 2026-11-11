include(joinpath(@__DIR__, "ks1998_ablation.jl"))

using Serialization

"""K-shock unbiased U-statistic for the squared conditional Euler residual."""
function kshock_loss_components(theta, state, shocks, p::KSParams, ss=steady_state(p))
    K = length(shocks)
    K >= 2 || throw(ArgumentError("K must be at least 2"))
    current = policy(theta, state.w, state.y, state.z, p, ss)
    branches = [_future_branch(theta, current, state.y, state.z, shock, p, ss)
                for shock in shocks]
    residuals = [branch.euler for branch in branches]
    residual_sum = reduce(+, residuals)
    residual_square_sum = reduce(+, (residual .^ 2 for residual in residuals))
    euler_u = mean((residual_sum .^ 2 .- residual_square_sum) ./ (K * (K - 1)))
    fisher_burmeister = complementarity_residual(current, state, p)
    complementarity = mean(fisher_burmeister .^ 2)
    return (; loss=complementarity + euler_u, complementarity,
            euler=euler_u, current, branches)
end

kshock_loss(theta, state, shocks, p::KSParams, ss=steady_state(p)) =
    kshock_loss_components(theta, state, shocks, p, ss).loss

function fb_loss(theta, state, p::KSParams, ss=steady_state(p))
    current = policy(theta, state.w, state.y, state.z, p, ss)
    return mean(complementarity_residual(current, state, p) .^ 2)
end

"""
Train with K independent branch RNG streams. Branch 1 drives the simulated state,
so K=2 and K=4 face identical exogenous state-transition shocks.
"""
function train_kshock!(theta, p::KSParams, ss;
                       K=2, updates=15000, batch_size=10, update_every=2,
                       lr=1e-3, seed=2026, adam_eps=1e-7)
    branch_rngs = [MersenneTwister(seed + 100_000 * branch) for branch in 1:K]
    state = initial_state(batch_size, p, ss)
    optimizer = AdamState(length(theta))
    periods = updates * update_every
    tail_length = min(500, periods)
    tail_abs_loss = Float64[]
    tail_fb = Float64[]
    tail_abs_euler = Float64[]

    for period in 1:periods
        shocks = [draw_shocks(branch_rngs[branch], batch_size, p) for branch in 1:K]
        components = kshock_loss_components(theta, state, shocks, p, ss)
        if (period - 1) % update_every == 0
            objective = th -> kshock_loss(th, state, shocks, p, ss)
            gradient = Flux.gradient(objective, theta)[1]
            all(isfinite, gradient) || error("non-finite K=$K gradient at period $period")
            _adam_step!(theta, gradient, optimizer; lr=lr, eps=adam_eps)
        end

        if period > periods - tail_length
            push!(tail_abs_loss, abs(components.loss))
            push!(tail_fb, components.complementarity)
            push!(tail_abs_euler, abs(components.euler))
        end

        first_branch = components.branches[1]
        state = (; w=first_branch.prices.wealth_next,
                 y=first_branch.y_next, z=first_branch.z_next)

        if period == 1 || period % 5000 == 0
            @printf("K=%d period=%5d updates=%5d | |loss|=%.4e FB=%.4e |Euler U|=%.4e\n",
                K, period, optimizer.t, abs(components.loss),
                components.complementarity, abs(components.euler))
            flush(stdout)
        end
    end

    return (; theta, state, updates=optimizer.t,
            tail_abs_loss=mean(tail_abs_loss),
            tail_fb=mean(tail_fb),
            tail_abs_euler=mean(tail_abs_euler))
end

"""Generate an independent batch of simulated states under a fixed policy."""
function simulated_state_bank(theta, p::KSParams, ss;
                              batch_size=20, burn=500, seed=9001)
    rng = MersenneTwister(seed)
    state = initial_state(batch_size, p, ss)
    for _ in 1:burn
        state = _advance_state(theta, state, draw_shocks(rng, batch_size, p), p, ss)
    end
    return state
end

function combine_state_banks(banks...)
    return (; w=vcat((bank.w for bank in banks)...),
            y=vcat((bank.y for bank in banks)...),
            z=vcat((bank.z for bank in banks)...))
end

"""Conditional residuals on a fixed state bank and common validation shocks."""
function fixed_bank_validation(theta, state_bank, p::KSParams, ss;
                               n_shocks=128, seed=99)
    rng = MersenneTwister(seed)
    current = policy(theta, state_bank.w, state_bank.y, state_bank.z, p, ss)
    residuals = [_future_branch(
        theta, current, state_bank.y, state_bank.z,
        draw_shocks(rng, size(state_bank.w, 1), p), p, ss,
    ).euler for _ in 1:n_shocks]
    conditional_euler = reduce(+, residuals) ./ n_shocks
    absolute_error = abs.(vec(conditional_euler))
    log_error = log10.(absolute_error .+ 1e-12)
    fisher_burmeister = complementarity_residual(current, state_bank, p)
    return (; mean_log10_euler=mean(log_error),
            median_log10_euler=median(log_error),
            p90_log10_euler=quantile(log_error, 0.90),
            p95_log10_euler=quantile(log_error, 0.95),
            p99_log10_euler=quantile(log_error, 0.99),
            max_abs_euler=maximum(absolute_error),
            conditional_mse=mean(conditional_euler .^ 2),
            mean_abs_fb=mean(abs, fisher_burmeister))
end

function policy_metrics(theta, state, p::KSParams, ss)
    shape = policy_shape_metrics(theta, state, p, ss)
    return (; monotonicity_violations=shape.monotonicity_violations,
            concavity_violations=shape.concavity_violations,
            min_policy_slope=shape.min_slope,
            max_consumption=shape.max_consumption)
end

"""
Repeat stochastic Euler gradients at fixed theta and state. Total gradients add
the deterministic FB gradient to each Euler-gradient realization.
"""
function gradient_noise_diagnostic(theta, state, p::KSParams, ss;
                                   K=2, repetitions=32, seed=7000)
    fb_gradient = Flux.gradient(th -> fb_loss(th, state, p, ss), theta)[1]
    euler_gradients = Vector{Vector{Float64}}(undef, repetitions)
    euler_estimates = zeros(repetitions)

    for repetition in 1:repetitions
        rng = MersenneTwister(seed + repetition)
        shocks = [draw_shocks(rng, size(state.w, 1), p) for _ in 1:K]
        euler_objective = th -> kshock_loss_components(th, state, shocks, p, ss).euler
        euler_estimates[repetition] = euler_objective(theta)
        euler_gradients[repetition] = Flux.gradient(euler_objective, theta)[1]
    end

    mean_euler_gradient = reduce(+, euler_gradients) ./ repetitions
    euler_noise = sqrt(mean(sum(abs2, gradient .- mean_euler_gradient)
                            for gradient in euler_gradients))
    euler_signal = norm(mean_euler_gradient)
    total_gradients = [gradient .+ fb_gradient for gradient in euler_gradients]
    mean_total_gradient = reduce(+, total_gradients) ./ repetitions
    total_noise = sqrt(mean(sum(abs2, gradient .- mean_total_gradient)
                            for gradient in total_gradients))
    total_signal = norm(mean_total_gradient)

    return (; K, repetitions,
            mean_euler_estimate=mean(euler_estimates),
            sd_euler_estimate=std(euler_estimates),
            euler_gradient_signal=euler_signal,
            euler_gradient_noise=euler_noise,
            euler_gradient_snr=euler_signal / max(euler_noise, eps()),
            total_gradient_signal=total_signal,
            total_gradient_noise=total_noise,
            total_gradient_snr=total_signal / max(total_noise, eps()))
end

function training_row(K, result, validation, shape, seconds)
    return (; K, updates=result.updates, seconds,
            mean_log10_euler=validation.mean_log10_euler,
            median_log10_euler=validation.median_log10_euler,
            p90_log10_euler=validation.p90_log10_euler,
            p95_log10_euler=validation.p95_log10_euler,
            p99_log10_euler=validation.p99_log10_euler,
            max_abs_euler=validation.max_abs_euler,
            conditional_mse=validation.conditional_mse,
            mean_abs_fb=validation.mean_abs_fb,
            monotonicity_violations=shape.monotonicity_violations,
            concavity_violations=shape.concavity_violations,
            min_policy_slope=shape.min_policy_slope,
            max_consumption=shape.max_consumption,
            tail_abs_loss=result.tail_abs_loss,
            tail_fb_mse=result.tail_fb,
            tail_abs_euler=result.tail_abs_euler)
end

if abspath(PROGRAM_FILE) == @__FILE__
updates = isempty(ARGS) ? 15000 : parse(Int, ARGS[1])
p = KSParams(agents=50, fb_convention=:released_code)
ss = steady_state(p)

println("Training controlled K=2 and K=4 models for $updates updates")
theta2 = init_theta(MersenneTwister(1234), p; style=:reference)
seconds2 = @elapsed result2 = train_kshock!(theta2, p, ss; K=2, updates=updates)
theta4 = init_theta(MersenneTwister(1234), p; style=:reference)
seconds4 = @elapsed result4 = train_kshock!(theta4, p, ss; K=4, updates=updates)

bank2 = simulated_state_bank(result2.theta, p, ss; batch_size=20, burn=500, seed=9001)
bank4 = simulated_state_bank(result4.theta, p, ss; batch_size=20, burn=500, seed=9002)
fixed_bank = combine_state_banks(bank2, bank4)

validation2 = fixed_bank_validation(result2.theta, fixed_bank, p, ss;
                                    n_shocks=128, seed=99)
validation4 = fixed_bank_validation(result4.theta, fixed_bank, p, ss;
                                    n_shocks=128, seed=99)
shape2 = policy_metrics(result2.theta, fixed_bank, p, ss)
shape4 = policy_metrics(result4.theta, fixed_bank, p, ss)

training_rows = [
    training_row(2, result2, validation2, shape2, seconds2),
    training_row(4, result4, validation4, shape4, seconds4),
]

println("Measuring repeated-gradient noise at the trained K=2 solution")
noise_rows = [
    gradient_noise_diagnostic(result2.theta, fixed_bank, p, ss; K=2),
    gradient_noise_diagnostic(result2.theta, fixed_bank, p, ss; K=4),
]

results_dir = joinpath(PROJECT_ROOT, "experiments", "results")
mkpath(results_dir)
suffix = updates == 15000 ? "full" : string(updates)
training_path = joinpath(results_dir, "ks1998_kshock_training_$(suffix).csv")
noise_path = joinpath(results_dir, "ks1998_kshock_gradient_noise_$(suffix).csv")
models_path = joinpath(results_dir, "ks1998_kshock_models_$(suffix).jls")
write_results(training_path, training_rows)
write_results(noise_path, noise_rows)
serialize(models_path, (; p, ss, result2, result4, fixed_bank))

println("Training comparison:")
for row in training_rows
    @printf("K=%d | median=%7.4f p90=%7.4f condMSE=%.4e | mono=%d | %.1fs\n",
        row.K, row.median_log10_euler, row.p90_log10_euler,
        row.conditional_mse, row.monotonicity_violations, row.seconds)
end
println("Gradient-noise comparison at fixed theta/state:")
for row in noise_rows
    @printf("K=%d | Euler loss sd=%.4e | Euler grad SNR=%.4f | total grad SNR=%.4f\n",
        row.K, row.sd_euler_estimate, row.euler_gradient_snr,
        row.total_gradient_snr)
end
println("Saved $training_path")
println("Saved $noise_path")
println("Saved $models_path")
end
