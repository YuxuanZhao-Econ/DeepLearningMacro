using Pkg

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(PROJECT_ROOT)

using Flux
using LinearAlgebra
using Printf
using Random
using Statistics

Base.@kwdef struct KSParams
    alpha::Float64 = 0.36
    delta::Float64 = 0.08
    A::Float64 = 1.0
    beta::Float64 = 0.96
    gamma::Float64 = 1.0
    rho_z::Float64 = 0.90
    sigma_z::Float64 = 0.01
    rho_y::Float64 = 0.95
    sigma_y::Float64 = 0.20
    agents::Int = 50
    hidden::Int = 32
    xi_min::Float64 = 0.0
    xi_max::Float64 = 1.0
    wealth_cap_multiple::Float64 = 4.0
    productivity_sd_bound::Float64 = 2.0
    mu_log_bound::Float64 = Inf
    fb_convention::Symbol = :released_code
end

function steady_state(p::KSParams)
    K = ((1 / p.beta - (1 - p.delta)) / (p.alpha * p.A))^(1 / (p.alpha - 1))
    Y = p.A * K^p.alpha
    rental = p.A * p.alpha * K^(p.alpha - 1)
    wage = p.A * (1 - p.alpha) * K^p.alpha
    W = (1 - p.delta + rental) * K + wage
    C = Y - p.delta * K
    xi = C / W
    return (; K, Y, C, W, xi, rental, wage)
end

logit(x) = log(x / (1 - x))

function sigmoid_stable(x)
    if x >= 0
        q = exp(-x)
        return inv(1 + q)
    end
    q = exp(x)
    return q / (1 + q)
end

input_dimension(p::KSParams) = 2 * p.agents + 3

function nparams(p::KSParams)
    D, H = input_dimension(p), p.hidden
    return H * D + H + H * H + H + 2 * H + 2
end

function _unpack(theta, p::KSParams)
    D, H = input_dimension(p), p.hidden
    i = 1
    W1 = reshape(@view(theta[i:(i + H * D - 1)]), H, D)
    i += H * D
    b1 = @view(theta[i:(i + H - 1)])
    i += H
    W2 = reshape(@view(theta[i:(i + H * H - 1)]), H, H)
    i += H * H
    b2 = @view(theta[i:(i + H - 1)])
    i += H
    W3 = reshape(@view(theta[i:(i + 2 * H - 1)]), 2, H)
    i += 2 * H
    b3 = @view(theta[i:(i + 1)])
    return W1, b1, W2, b2, W3, b3
end

function _truncated_standard_normal(rng::AbstractRNG; cutoff=2.0)
    value = randn(rng)
    while abs(value) > cutoff
        value = randn(rng)
    end
    return value
end

function init_theta(rng::AbstractRNG, p::KSParams; style=:reference)
    if style === :old
        theta = 0.02 .* randn(rng, nparams(p))
        _, _, _, _, _, b3 = _unpack(theta, p)
        fill!(b3, 0.0)
        return theta
    elseif style !== :reference
        throw(ArgumentError("initialization style must be :reference or :old"))
    end

    theta = zeros(Float64, nparams(p))
    W1, b1, W2, b2, W3, b3 = _unpack(theta, p)
    for weights in (W1, W2, W3), index in eachindex(weights)
        weights[index] = 0.01 * _truncated_standard_normal(rng)
    end
    fill!(b1, 0.0)
    fill!(b2, 0.0)
    fill!(b3, 0.0)
    return theta
end

function initial_state(batch_size::Int, p::KSParams, ss=steady_state(p))
    return (; w=fill(ss.W, batch_size, p.agents),
            y=ones(batch_size, p.agents), z=ones(batch_size))
end

function _state_bounds(p::KSParams, ss)
    y_log_sd = p.sigma_y / sqrt(1 - p.rho_y^2)
    z_log_sd = p.sigma_z / sqrt(1 - p.rho_z^2)
    return (; y_log_sd, z_log_sd, w_max=p.wealth_cap_multiple * ss.W)
end

function _normalize_state(w, y, z, p::KSParams, ss)
    bounds = _state_bounds(p, ss)
    normw = 2 .* w ./ bounds.w_max .- 1
    normy = log.(max.(y, 1e-12)) ./ max(p.productivity_sd_bound * bounds.y_log_sd, 1e-8)
    normz = log.(max.(z, 1e-12)) ./ max(p.productivity_sd_bound * bounds.z_log_sd, 1e-8)
    return normw, normy, normz
end

function _network_input(w, y, z, p::KSParams, ss)
    B, N = size(w)
    normw, normy, normz = _normalize_state(w, y, z, p, ss)
    full_w = reshape(
        permutedims(repeat(reshape(normw, B, N, 1), 1, 1, N), (2, 3, 1)),
        N, N * B,
    )
    full_y = reshape(
        permutedims(repeat(reshape(normy, B, N, 1), 1, 1, N), (2, 3, 1)),
        N, N * B,
    )
    aggregate_z = reshape(vec(repeat(reshape(normz, 1, B), N, 1)), 1, N * B)
    own_w = reshape(vec(permutedims(normw, (2, 1))), 1, N * B)
    own_y = reshape(vec(permutedims(normy, (2, 1))), 1, N * B)
    return vcat(full_w, full_y, aggregate_z, own_w, own_y)
end

function _network_raw(theta, x, p::KSParams)
    W1, b1, W2, b2, W3, b3 = _unpack(theta, p)
    h1 = sigmoid_stable.(W1 * x .+ reshape(b1, :, 1))
    h2 = sigmoid_stable.(W2 * h1 .+ reshape(b2, :, 1))
    return W3 * h2 .+ reshape(b3, :, 1)
end

function policy(theta, w, y, z, p::KSParams, ss=steady_state(p))
    B, N = size(w)
    raw = _network_raw(theta, _network_input(w, y, z, p, ss), p)
    qss = (ss.xi - p.xi_min) / (p.xi_max - p.xi_min)
    xi_vector = p.xi_min .+ (p.xi_max - p.xi_min) .*
                sigmoid_stable.(raw[1, :] .+ logit(qss))
    mu_vector = exp.(clamp.(raw[2, :], -p.mu_log_bound, p.mu_log_bound))
    xi = permutedims(reshape(xi_vector, N, B), (2, 1))
    mu = permutedims(reshape(mu_vector, N, B), (2, 1))
    kp = min.((1 .- xi) .* w, p.wealth_cap_multiple * ss.W)
    c = max.(w .- kp, 1e-10)
    return (; xi, mu, c, kp)
end

function draw_shocks(rng::AbstractRNG, batch_size::Int, p::KSParams)
    return (; eps_y=randn(rng, batch_size, p.agents),
            eps_z=randn(rng, batch_size))
end

function transition_productivity(y, z, shocks, p::KSParams)
    bounds = _state_bounds(p, steady_state(p))
    y_bound = p.productivity_sd_bound * bounds.y_log_sd
    z_bound = p.productivity_sd_bound * bounds.z_log_sd
    log_y_next = p.rho_y .* log.(max.(y, 1e-12)) .+
                 p.sigma_y .* shocks.eps_y .-
                 0.5 * p.sigma_y^2 / (1 + p.rho_y)
    y_next_raw = exp.(clamp.(log_y_next, -y_bound, y_bound))
    y_next = y_next_raw ./ mean(y_next_raw; dims=2)
    log_z_next = p.rho_z .* log.(max.(z, 1e-12)) .+
                 p.sigma_z .* shocks.eps_z .-
                 0.5 * p.sigma_z^2 / (1 + p.rho_z)
    z_next = exp.(clamp.(log_z_next, -z_bound, z_bound))
    return y_next, z_next
end

function next_period(kp, y_next, z_next, p::KSParams)
    B = size(kp, 1)
    aggregate_k = vec(mean(kp; dims=2))
    rental = p.A .* z_next .* p.alpha .* aggregate_k .^ (p.alpha - 1)
    wage = p.A .* z_next .* (1 - p.alpha) .* aggregate_k .^ p.alpha
    gross_return = 1 - p.delta .+ rental
    wealth_next = reshape(gross_return, B, 1) .* kp .+
                  reshape(wage, B, 1) .* y_next
    return (; aggregate_k, rental, wage, gross_return, wealth_next)
end

function _future_branch(theta, current_policy, y, z, shocks, p::KSParams, ss)
    y_next, z_next = transition_productivity(y, z, shocks, p)
    prices = next_period(current_policy.kp, y_next, z_next, p)
    next_policy = policy(theta, prices.wealth_next, y_next, z_next, p, ss)
    euler = p.beta .* (current_policy.c ./ next_policy.c) .^ p.gamma .*
            reshape(prices.gross_return, :, 1) .- current_policy.mu
    return (; y_next, z_next, prices, next_policy, euler)
end

function complementarity_residual(current, state, p::KSParams)
    if p.fb_convention === :released_code
        a = current.mu .- 1
        b = state.w ./ max.(current.c, 1e-12) .- 1
    elseif p.fb_convention === :paper
        a = 1 .- current.c ./ max.(state.w, 1e-12)
        b = 1 .- current.mu
    else
        throw(ArgumentError("unknown FB convention"))
    end
    return a .+ b .- sqrt.(a .^ 2 .+ b .^ 2)
end

function aio_loss_components(theta, state, shocks_a, shocks_b, p::KSParams,
                             ss=steady_state(p))
    current = policy(theta, state.w, state.y, state.z, p, ss)
    branch_a = _future_branch(theta, current, state.y, state.z, shocks_a, p, ss)
    branch_b = _future_branch(theta, current, state.y, state.z, shocks_b, p, ss)
    fisher_burmeister = complementarity_residual(current, state, p)
    complementarity = mean(fisher_burmeister .^ 2)
    euler = mean(branch_a.euler .* branch_b.euler)
    return (; loss=complementarity + euler, complementarity, euler,
            current, branch_a, branch_b)
end

aio_loss(theta, state, shocks_a, shocks_b, p::KSParams, ss=steady_state(p)) =
    aio_loss_components(theta, state, shocks_a, shocks_b, p, ss).loss

function _advance_state(theta, state, shocks, p::KSParams, ss)
    current = policy(theta, state.w, state.y, state.z, p, ss)
    y_next, z_next = transition_productivity(state.y, state.z, shocks, p)
    prices = next_period(current.kp, y_next, z_next, p)
    return (; w=prices.wealth_next, y=y_next, z=z_next)
end

mutable struct AdamState
    m::Vector{Float64}
    v::Vector{Float64}
    t::Int
end

AdamState(n::Int) = AdamState(zeros(n), zeros(n), 0)

function _adam_step!(theta, grad, state::AdamState; lr=1e-3,
                     beta1=0.9, beta2=0.999, eps=1e-7)
    state.t += 1
    state.m .= beta1 .* state.m .+ (1 - beta1) .* grad
    state.v .= beta2 .* state.v .+ (1 - beta2) .* (grad .^ 2)
    mhat = state.m ./ (1 - beta1^state.t)
    vhat = state.v ./ (1 - beta2^state.t)
    theta .-= lr .* mhat ./ (sqrt.(vhat) .+ eps)
    return theta
end

function train_experiment!(theta, p::KSParams, ss;
                           updates=2000, batch_size=10, update_every=2,
                           lr=1e-3, seed=2026, adam_eps=1e-7,
                           advance_style=:pre)
    rng = MersenneTwister(seed)
    state = initial_state(batch_size, p, ss)
    optimizer = AdamState(length(theta))
    tail_loss = Float64[]
    tail_fb = Float64[]
    tail_euler = Float64[]
    tail_length = min(500, updates * update_every)
    periods = updates * update_every

    for period in 1:periods
        shocks_a = draw_shocks(rng, batch_size, p)
        shocks_b = draw_shocks(rng, batch_size, p)
        components = aio_loss_components(theta, state, shocks_a, shocks_b, p, ss)
        is_update = (period - 1) % update_every == 0
        if is_update
            loss_function = th -> aio_loss(th, state, shocks_a, shocks_b, p, ss)
            gradient = Flux.gradient(loss_function, theta)[1]
            all(isfinite, gradient) || error("non-finite gradient at period $period")
            _adam_step!(theta, gradient, optimizer; lr=lr, eps=adam_eps)
        end

        if period > periods - tail_length
            push!(tail_loss, abs(components.loss))
            push!(tail_fb, components.complementarity)
            push!(tail_euler, abs(components.euler))
        end

        if advance_style === :pre
            state = (; w=components.branch_a.prices.wealth_next,
                     y=components.branch_a.y_next, z=components.branch_a.z_next)
        elseif advance_style === :post
            state = _advance_state(theta, state, shocks_a, p, ss)
        else
            throw(ArgumentError("advance_style must be :pre or :post"))
        end
    end

    return (; theta, state, updates=optimizer.t,
            tail_abs_loss=mean(tail_loss), tail_fb=mean(tail_fb),
            tail_abs_euler=mean(tail_euler))
end

include(joinpath(PROJECT_ROOT, "src", "Diagnostics.jl"))

function policy_shape_metrics(theta, state, p, ss)
    slice = policy_slice(theta, state, p, ss;
        agent=1, points=100, productivity_points=7, wealth_min_multiple=0.0)
    slopes = diff(slice.consumption; dims=1)
    second_differences = diff(slice.consumption; dims=1) |> x -> diff(x; dims=1)
    scale = maximum(abs, slice.consumption)
    monotonicity_tolerance = 1e-8 * max(scale, 1.0)
    curvature_tolerance = 1e-6 * max(scale, 1.0)
    monotonicity_violations = count(<(-monotonicity_tolerance), slopes)
    concavity_violations = count(>(curvature_tolerance), second_differences)
    return (; monotonicity_violations, concavity_violations,
            min_slope=minimum(slopes), max_consumption=maximum(slice.consumption))
end

Base.@kwdef struct ExperimentConfig
    name::String
    agents::Int = 50
    batch_size::Int = 10
    updates::Int = 2000
    update_every::Int = 2
    init_style::Symbol = :reference
    fb_convention::Symbol = :released_code
    advance_style::Symbol = :pre
    adam_eps::Float64 = 1e-7
    old_bounds::Bool = false
    seed::Int = 2026
end

function run_experiment(config::ExperimentConfig)
    p = KSParams(
        agents=config.agents,
        xi_min=config.old_bounds ? 1e-4 : 0.0,
        xi_max=config.old_bounds ? 1 - 1e-4 : 1.0,
        mu_log_bound=config.old_bounds ? 8.0 : Inf,
        fb_convention=config.fb_convention,
    )
    ss = steady_state(p)
    theta = init_theta(MersenneTwister(1234), p; style=config.init_style)
    elapsed = @elapsed result = train_experiment!(theta, p, ss;
        updates=config.updates,
        batch_size=config.batch_size,
        update_every=config.update_every,
        seed=config.seed,
        adam_eps=config.adam_eps,
        advance_style=config.advance_style,
    )
    diagnostics = conditional_residuals(result.theta, result.state, p, ss;
        n_shocks=64, seed=99)
    log_euler = log10.(abs.(vec(diagnostics.euler)) .+ 1e-12)
    abs_fb = abs.(vec(diagnostics.fisher_burmeister))
    shape = policy_shape_metrics(result.theta, result.state, p, ss)
    simulation = simulate_economy(result.theta, p, ss;
        periods=500, burn=0, seed=404, start_state=result.state)
    aggregation = approximate_aggregation(simulation)

    row = (;
        name=config.name,
        agents=config.agents,
        batch_size=config.batch_size,
        updates=config.updates,
        seed=config.seed,
        init_style=String(config.init_style),
        fb_convention=String(config.fb_convention),
        advance_style=String(config.advance_style),
        adam_eps=config.adam_eps,
        old_bounds=config.old_bounds,
        parameters=nparams(p),
        seconds=elapsed,
        mean_log10_euler=mean(log_euler),
        median_log10_euler=median(log_euler),
        p90_log10_euler=quantile(log_euler, 0.90),
        mean_abs_fb=mean(abs_fb),
        monotonicity_violations=shape.monotonicity_violations,
        concavity_violations=shape.concavity_violations,
        min_policy_slope=shape.min_slope,
        max_consumption=shape.max_consumption,
        tail_abs_loss=result.tail_abs_loss,
        tail_fb_mse=result.tail_fb,
        tail_abs_aio_euler=result.tail_abs_euler,
        mean_aggregate_k=mean(simulation.aggregate_k),
        aggregation_r2=aggregation.r2,
    )
    @printf("%-22s N=%3d B=%2d updates=%5d | med Euler=%7.4f p90=%7.4f | mono=%3d conc=%3d | %.1fs\n",
        row.name, row.agents, row.batch_size, row.updates,
        row.median_log10_euler, row.p90_log10_euler,
        row.monotonicity_violations, row.concavity_violations, row.seconds)
    flush(stdout)
    return row
end

function experiment_configs(mode)
    if mode == "ablation"
        return [
            ExperimentConfig(name="reference_short"),
            ExperimentConfig(name="paper_fb", fb_convention=:paper),
            ExperimentConfig(name="old_initialization", init_style=:old),
            ExperimentConfig(name="post_update_advance", advance_style=:post),
            ExperimentConfig(name="adam_eps_1e-8", adam_eps=1e-8),
            ExperimentConfig(name="old_output_bounds", old_bounds=true),
        ]
    elseif mode == "scaling"
        return [
            ExperimentConfig(name="N50_B10", agents=50, batch_size=10),
            ExperimentConfig(name="N50_B20", agents=50, batch_size=20),
            ExperimentConfig(name="N50_B40", agents=50, batch_size=40),
            ExperimentConfig(name="N100_B10", agents=100, batch_size=10),
            ExperimentConfig(name="N100_B20", agents=100, batch_size=20),
        ]
    elseif mode == "long"
        return [
            ExperimentConfig(name="reference_15k", updates=15000),
            ExperimentConfig(name="N50_B20_15k", batch_size=20, updates=15000),
            ExperimentConfig(name="N100_B10_15k", agents=100, updates=15000),
        ]
    elseif mode == "long_seed_2027"
        return [
            ExperimentConfig(name="N50_B10_15k_seed2027", updates=15000, seed=2027),
            ExperimentConfig(name="N50_B20_15k_seed2027", batch_size=20,
                             updates=15000, seed=2027),
        ]
    elseif mode == "old_comparison"
        return [
            ExperimentConfig(name="reference_every_period", update_every=1),
            ExperimentConfig(name="old_bundle_N50_B10", update_every=1,
                             init_style=:old, fb_convention=:paper,
                             advance_style=:post, adam_eps=1e-8, old_bounds=true),
            ExperimentConfig(name="old_like_N200_B20", agents=200, batch_size=20,
                             update_every=1, init_style=:old,
                             fb_convention=:paper, advance_style=:post,
                             adam_eps=1e-8, old_bounds=true),
        ]
    end
    throw(ArgumentError("unknown experiment mode"))
end

function csv_value(value)
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_results(path, rows)
    mkpath(dirname(path))
    names = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(string.(names), ','))
        for row in rows
            println(io, join((csv_value(getproperty(row, name)) for name in names), ','))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? "ablation" : ARGS[1]
    configs = experiment_configs(mode)
    println("Running $(length(configs)) KS1998 experiments in mode=$mode")
    rows = [run_experiment(config) for config in configs]
    output_path = joinpath(PROJECT_ROOT, "experiments", "results", "ks1998_$(mode).csv")
    write_results(output_path, rows)
    println("Saved results to $output_path")
end
