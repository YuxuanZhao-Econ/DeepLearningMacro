module KSReplayExperiment

using Flux
using LinearAlgebra
using Random
using Statistics

export KSParams,
       ReplayConfig,
       steady_state,
       init_theta,
       initial_state,
       prepare_initial_replay,
       train_online,
       train_prioritized_replay,
       collect_state_panel,
       concatenate_states,
       conditional_errors,
       validation_summary,
       feasible_policy_slice,
       rectangular_policy_slice

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
    xi_min::Float64 = 1e-4
    xi_max::Float64 = 1.0 - 1e-4
    wealth_scale_multiple::Float64 = 4.0
    productivity_sd_bound::Float64 = 2.0
    mu_log_bound::Float64 = 8.0
end

Base.@kwdef struct ReplayConfig
    capacity::Int = 300
    replay_fraction::Float64 = 0.30
    priority_alpha::Float64 = 0.70
    focus_agents::Int = 8
    refresh_every::Int = 5
    priority_shocks::Int = 2
    priority_floor::Float64 = 1e-4
end

"""Deterministic steady state with unit labor and productivity."""
function steady_state(p::KSParams)
    K = ((1 / p.beta - (1 - p.delta)) / (p.alpha * p.A))^(1 / (p.alpha - 1))
    Y = p.A * K^p.alpha
    rental = p.A * p.alpha * K^(p.alpha - 1)
    wage = p.A * (1 - p.alpha) * K^p.alpha
    gross_return = 1 - p.delta + rental
    W = gross_return * K + wage
    C = Y - p.delta * K
    xi = C / W
    return (; K, Y, C, W, xi, rental, wage, gross_return)
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

function init_theta(rng::AbstractRNG, p::KSParams; scale=0.02)
    theta = scale .* randn(rng, nparams(p))
    D, H = input_dimension(p), p.hidden
    b3_start = H * D + H + H * H + H + 2 * H + 1
    theta[b3_start:(b3_start + 1)] .= 0.0
    return theta
end

function initial_state(batch_size::Int, p::KSParams, ss=steady_state(p))
    return (;
        w=fill(ss.W, batch_size, p.agents),
        y=ones(batch_size, p.agents),
        z=ones(batch_size),
    )
end

copy_state(state) = (; w=copy(state.w), y=copy(state.y), z=copy(state.z))

function _state_bounds(p::KSParams, ss)
    y_log_sd = p.sigma_y / sqrt(1 - p.rho_y^2)
    z_log_sd = p.sigma_z / sqrt(1 - p.rho_z^2)
    wealth_scale = p.wealth_scale_multiple * ss.W
    return (; y_log_sd, z_log_sd, wealth_scale)
end

function _normalize_state(w, y, z, p::KSParams, ss)
    bounds = _state_bounds(p, ss)
    normw = 2 .* w ./ bounds.wealth_scale .- 1
    normy = log.(max.(y, 1e-12)) ./ max(p.productivity_sd_bound * bounds.y_log_sd, 1e-8)
    normz = log.(max.(z, 1e-12)) ./ max(p.productivity_sd_bound * bounds.z_log_sd, 1e-8)
    return normw, normy, normz
end

function _network_input(w, y, z, p::KSParams, ss)
    B, N = size(w)
    N == p.agents || throw(DimensionMismatch("state has $N agents; expected $(p.agents)"))
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

"""
Policy map without an artificial upper bound on capital. The bounded consumption
share enforces c>0, k'>=0, and the budget identity by construction.
"""
function policy(theta, w, y, z, p::KSParams, ss=steady_state(p))
    B, N = size(w)
    raw = _network_raw(theta, _network_input(w, y, z, p, ss), p)
    qss = (ss.xi - p.xi_min) / (p.xi_max - p.xi_min)
    xi_vector = p.xi_min .+ (p.xi_max - p.xi_min) .*
                sigmoid_stable.(raw[1, :] .+ logit(qss))
    mu_vector = exp.(clamp.(raw[2, :], -p.mu_log_bound, p.mu_log_bound))
    xi = permutedims(reshape(xi_vector, N, B), (2, 1))
    mu = permutedims(reshape(mu_vector, N, B), (2, 1))
    c = xi .* w
    kp = (1 .- xi) .* w
    return (; xi, mu, c, kp)
end

function draw_shocks(rng::AbstractRNG, batch_size::Int, p::KSParams)
    return (; eps_y=randn(rng, batch_size, p.agents), eps_z=randn(rng, batch_size))
end

function transition_productivity(y, z, shocks, p::KSParams)
    bounds = _state_bounds(p, steady_state(p))
    y_bound = p.productivity_sd_bound * bounds.y_log_sd
    z_bound = p.productivity_sd_bound * bounds.z_log_sd

    log_y_next = p.rho_y .* log.(max.(y, 1e-12)) .+
                 p.sigma_y .* shocks.eps_y .-
                 0.5 * p.sigma_y^2 / (1 + p.rho_y)
    log_y_next = clamp.(log_y_next, -y_bound, y_bound)
    y_next_raw = exp.(log_y_next)
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

function future_branch(theta, current, y, z, shocks, p::KSParams, ss)
    y_next, z_next = transition_productivity(y, z, shocks, p)
    prices = next_period(current.kp, y_next, z_next, p)
    next_policy = policy(theta, prices.wealth_next, y_next, z_next, p, ss)
    euler = p.beta .* (current.c ./ next_policy.c) .^ p.gamma .*
            reshape(prices.gross_return, :, 1) .- current.mu
    return (; y_next, z_next, prices, next_policy, euler)
end

function residual_components(theta, state, shocks_a, shocks_b, p::KSParams, ss)
    current = policy(theta, state.w, state.y, state.z, p, ss)
    branch_a = future_branch(theta, current, state.y, state.z, shocks_a, p, ss)
    branch_b = future_branch(theta, current, state.y, state.z, shocks_b, p, ss)
    a = current.kp ./ max.(state.w, 1e-12)
    b = 1 .- current.mu
    fb = a .+ b .- sqrt.(a .^ 2 .+ b .^ 2)
    per_household = fb .^ 2 .+ branch_a.euler .* branch_b.euler
    return (; per_household, fb, current, branch_a, branch_b)
end

function weighted_aio_loss(theta, state, shocks_a, shocks_b, weights, p::KSParams, ss)
    components = residual_components(theta, state, shocks_a, shocks_b, p, ss)
    return sum(weights .* components.per_household)
end

function advance_state(theta, state, shocks, p::KSParams, ss)
    current = policy(theta, state.w, state.y, state.z, p, ss)
    y_next, z_next = transition_productivity(state.y, state.z, shocks, p)
    prices = next_period(current.kp, y_next, z_next, p)
    return (; w=prices.wealth_next, y=y_next, z=z_next)
end

function select_rows(state, rows)
    return (; w=state.w[rows, :], y=state.y[rows, :], z=state.z[rows])
end

function concatenate_states(states...)
    return (;
        w=vcat((state.w for state in states)...),
        y=vcat((state.y for state in states)...),
        z=vcat((state.z for state in states)...),
    )
end

mutable struct AdamState
    m::Vector{Float64}
    v::Vector{Float64}
    t::Int
end

AdamState(n::Int) = AdamState(zeros(n), zeros(n), 0)

function adam_step!(theta, grad, state::AdamState;
                    lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)
    state.t += 1
    state.m .= beta1 .* state.m .+ (1 - beta1) .* grad
    state.v .= beta2 .* state.v .+ (1 - beta2) .* (grad .^ 2)
    mhat = state.m ./ (1 - beta1^state.t)
    vhat = state.v ./ (1 - beta2^state.t)
    theta .-= lr .* mhat ./ (sqrt.(vhat) .+ eps)
    return theta
end

mutable struct ReplayBuffer
    w::Matrix{Float64}
    y::Matrix{Float64}
    z::Vector{Float64}
    priority::Matrix{Float64}
    size::Int
    next::Int
end

function ReplayBuffer(capacity::Int, p::KSParams)
    return ReplayBuffer(
        zeros(capacity, p.agents),
        zeros(capacity, p.agents),
        zeros(capacity),
        ones(capacity, p.agents),
        0,
        1,
    )
end

function push_states!(buffer::ReplayBuffer, state, priority)
    B = size(state.w, 1)
    for b in 1:B
        row = buffer.next
        buffer.w[row, :] .= state.w[b, :]
        buffer.y[row, :] .= state.y[b, :]
        buffer.z[row] = state.z[b]
        buffer.priority[row, :] .= priority[b, :]
        buffer.size = min(buffer.size + 1, size(buffer.w, 1))
        buffer.next = row == size(buffer.w, 1) ? 1 : row + 1
    end
    return buffer
end

function sample_weighted(rng::AbstractRNG, weights, n::Int)
    clean = max.(Float64.(weights), 0.0)
    total = sum(clean)
    if !isfinite(total) || total <= 0
        return rand(rng, 1:length(clean), n)
    end
    cdf = cumsum(clean)
    return [searchsortedfirst(cdf, rand(rng) * total) for _ in 1:n]
end

function estimate_priority(theta, state, p::KSParams, ss, rng::AbstractRNG;
                           n_shocks=2, floor=1e-4)
    current = policy(theta, state.w, state.y, state.z, p, ss)
    conditional = zeros(size(state.w))
    for _ in 1:n_shocks
        shocks = draw_shocks(rng, size(state.w, 1), p)
        branch = future_branch(theta, current, state.y, state.z, shocks, p, ss)
        conditional .+= branch.euler
    end
    conditional ./= n_shocks
    a = current.kp ./ max.(state.w, 1e-12)
    b = 1 .- current.mu
    fb = a .+ b .- sqrt.(a .^ 2 .+ b .^ 2)
    return floor .+ abs.(conditional) .+ abs.(fb)
end

function sample_replay(buffer::ReplayBuffer, rng::AbstractRNG, rows::Int,
                       focus_agents::Int; alpha=0.7)
    buffer.size > 0 || error("cannot sample from an empty replay buffer")
    active_priority = @view buffer.priority[1:buffer.size, :]
    row_score = vec(mean(active_priority .^ alpha; dims=2))
    sampled_rows = sample_weighted(rng, row_score, rows)
    replay_state = (;
        w=copy(buffer.w[sampled_rows, :]),
        y=copy(buffer.y[sampled_rows, :]),
        z=copy(buffer.z[sampled_rows]),
    )
    focus = Matrix{Int}(undef, rows, focus_agents)
    for b in 1:rows
        if alpha == 0.0 && focus_agents == size(buffer.priority, 2)
            # Exact uniform replay over every household in the saved economy.
            # This arm isolates state replay from residual-based prioritization.
            focus[b, :] .= 1:focus_agents
        else
            household_score = max.(buffer.priority[sampled_rows[b], :], 0.0) .^ alpha
            focus[b, :] .= sample_weighted(rng, household_score, focus_agents)
        end
    end
    return replay_state, focus, sampled_rows
end

function prepare_initial_replay(theta, p::KSParams, ss=steady_state(p);
                                batch_size=10, warmup=60, insert_every=2,
                                config=ReplayConfig(), seed=2027)
    transition_rng = MersenneTwister(seed)
    priority_rng = MersenneTwister(seed + 1)
    state = initial_state(batch_size, p, ss)
    buffer = ReplayBuffer(config.capacity, p)
    for t in 1:warmup
        state = advance_state(theta, state, draw_shocks(transition_rng, batch_size, p), p, ss)
        if t % insert_every == 0
            priority = estimate_priority(
                theta, state, p, ss, priority_rng;
                n_shocks=config.priority_shocks,
                floor=config.priority_floor,
            )
            push_states!(buffer, state, priority)
        end
    end
    return copy_state(state), buffer
end

function training_weights(batch_size::Int, agents::Int)
    return fill(inv(batch_size * agents), batch_size, agents)
end

function mixed_weights(recent_rows::Int, replay_rows::Int, agents::Int,
                       focus, replay_fraction::Float64)
    weights = zeros(recent_rows + replay_rows, agents)
    weights[1:recent_rows, :] .= (1 - replay_fraction) / (recent_rows * agents)
    each_focus_weight = replay_fraction / length(focus)
    for b in 1:replay_rows, j in axes(focus, 2)
        weights[recent_rows + b, focus[b, j]] += each_focus_weight
    end
    return weights
end

function record_loss(theta, state, weights, p, ss, rng)
    B = size(state.w, 1)
    components = residual_components(
        theta, state,
        draw_shocks(rng, B, p), draw_shocks(rng, B, p), p, ss,
    )
    return (
        total=sum(weights .* components.per_household),
        fb=sum(weights .* (components.fb .^ 2)),
        euler=sum(weights .* (components.branch_a.euler .* components.branch_b.euler)),
    )
end

function train_online(initial_theta, initial_live, p::KSParams, ss=steady_state(p);
                      steps=800, lr=1e-3, seed=2030, record_every=20)
    theta = copy(initial_theta)
    live = copy_state(initial_live)
    batch_size = size(live.w, 1)
    optimizer = AdamState(length(theta))
    loss_rng = MersenneTwister(seed)
    transition_rng = MersenneTwister(seed + 1)
    record_rng = MersenneTwister(seed + 2)
    history = NamedTuple[]
    weights = training_weights(batch_size, p.agents)

    for step in 1:steps
        shocks_a = draw_shocks(loss_rng, batch_size, p)
        shocks_b = draw_shocks(loss_rng, batch_size, p)
        objective = th -> weighted_aio_loss(th, live, shocks_a, shocks_b, weights, p, ss)
        gradient = Flux.gradient(objective, theta)[1]
        all(isfinite, gradient) || error("non-finite baseline gradient at step $step")
        adam_step!(theta, gradient, optimizer; lr=lr)
        live = advance_state(
            theta, live, draw_shocks(transition_rng, batch_size, p), p, ss,
        )
        if step == 1 || step % record_every == 0
            values = record_loss(theta, live, weights, p, ss, record_rng)
            push!(history, (; step, values...))
        end
    end
    return (; theta, live, history)
end

function train_prioritized_replay(initial_theta, initial_live, initial_buffer,
                                  p::KSParams, ss=steady_state(p);
                                  steps=800, lr=1e-3, seed=2030,
                                  record_every=20, config=ReplayConfig())
    theta = copy(initial_theta)
    live = copy_state(initial_live)
    buffer = deepcopy(initial_buffer)
    batch_size = size(live.w, 1)
    replay_rows = clamp(round(Int, config.replay_fraction * batch_size), 1, batch_size - 1)
    recent_rows = batch_size - replay_rows
    optimizer = AdamState(length(theta))
    loss_rng = MersenneTwister(seed)
    transition_rng = MersenneTwister(seed + 1)
    record_rng = MersenneTwister(seed + 2)
    replay_rng = MersenneTwister(seed + 3)
    priority_rng = MersenneTwister(seed + 4)
    history = NamedTuple[]

    for step in 1:steps
        recent_indices = randperm(replay_rng, batch_size)[1:recent_rows]
        recent = select_rows(live, recent_indices)
        replay, focus, _ = sample_replay(
            buffer, replay_rng, replay_rows, config.focus_agents;
            alpha=config.priority_alpha,
        )
        training_state = concatenate_states(recent, replay)
        weights = mixed_weights(
            recent_rows, replay_rows, p.agents, focus, config.replay_fraction,
        )
        shocks_a = draw_shocks(loss_rng, batch_size, p)
        shocks_b = draw_shocks(loss_rng, batch_size, p)
        objective = th -> weighted_aio_loss(
            th, training_state, shocks_a, shocks_b, weights, p, ss,
        )
        gradient = Flux.gradient(objective, theta)[1]
        all(isfinite, gradient) || error("non-finite replay gradient at step $step")
        adam_step!(theta, gradient, optimizer; lr=lr)

        live = advance_state(
            theta, live, draw_shocks(transition_rng, batch_size, p), p, ss,
        )
        if step % config.refresh_every == 0
            priority = estimate_priority(
                theta, live, p, ss, priority_rng;
                n_shocks=config.priority_shocks,
                floor=config.priority_floor,
            )
            push_states!(buffer, live, priority)
        end
        if step == 1 || step % record_every == 0
            values = record_loss(theta, training_state, weights, p, ss, record_rng)
            focused_w = mean([
                replay.w[b, focus[b, j]]
                for b in 1:replay_rows, j in axes(focus, 2)
            ])
            focused_y = mean([
                replay.y[b, focus[b, j]]
                for b in 1:replay_rows, j in axes(focus, 2)
            ])
            push!(history, (; step, values..., focused_w, focused_y))
        end
    end
    return (; theta, live, history, buffer)
end

function collect_state_panel(theta, p::KSParams, ss=steady_state(p);
                             paths=5, burn=120, snapshots=10, stride=8, seed=3000)
    rng = MersenneTwister(seed)
    state = initial_state(paths, p, ss)
    for _ in 1:burn
        state = advance_state(theta, state, draw_shocks(rng, paths, p), p, ss)
    end
    w = Matrix{Float64}(undef, paths * snapshots, p.agents)
    y = similar(w)
    z = Vector{Float64}(undef, paths * snapshots)
    stored = 0
    for _ in 1:snapshots
        for _ in 1:stride
            state = advance_state(theta, state, draw_shocks(rng, paths, p), p, ss)
        end
        rows = (stored + 1):(stored + paths)
        w[rows, :] .= state.w
        y[rows, :] .= state.y
        z[rows] .= state.z
        stored += paths
    end
    return (; w, y, z)
end

function conditional_errors(theta, states, p::KSParams, ss=steady_state(p);
                            n_shocks=24, chunk_size=10, seed=4000)
    rng = MersenneTwister(seed)
    B, N = size(states.w)
    euler = Matrix{Float64}(undef, B, N)
    fb_all = similar(euler)
    for first_row in 1:chunk_size:B
        rows = first_row:min(first_row + chunk_size - 1, B)
        state = select_rows(states, rows)
        current = policy(theta, state.w, state.y, state.z, p, ss)
        conditional = zeros(size(state.w))
        for _ in 1:n_shocks
            branch = future_branch(
                theta, current, state.y, state.z,
                draw_shocks(rng, length(rows), p), p, ss,
            )
            conditional .+= branch.euler
        end
        conditional ./= n_shocks
        a = current.kp ./ max.(state.w, 1e-12)
        b = 1 .- current.mu
        fb = a .+ b .- sqrt.(a .^ 2 .+ b .^ 2)
        euler[rows, :] .= conditional
        fb_all[rows, :] .= fb
    end
    return (; euler, fb=fb_all)
end

function _metrics(values)
    x = abs.(Float64.(values))
    return (;
        count=length(x),
        median=median(x),
        p90=quantile(x, 0.90),
        rmse=sqrt(mean(x .^ 2)),
    )
end

function validation_summary(errors, states; wealth_cut, productivity_cut)
    w = vec(states.w)
    y = vec(states.y)
    residual = vec(errors.euler)
    all_mask = trues(length(w))
    high_w = w .>= wealth_cut
    high_y = y .>= productivity_cut
    joint = high_w .& high_y
    return (;
        all=_metrics(residual[all_mask]),
        high_wealth=_metrics(residual[high_w]),
        high_productivity=_metrics(residual[high_y]),
        joint_tail=_metrics(residual[joint]),
        mean_abs_fb=mean(abs.(errors.fb)),
    )
end

function feasible_policy_slice(theta, base_state, p::KSParams, ss=steady_state(p);
                               agent=1, points=80,
                               productivity_levels=[0.5, 1.0, 2.0])
    implied_assets = max.((base_state.w .- ss.wage .* base_state.y) ./ ss.gross_return, 0.0)
    k_max = max(quantile(vec(implied_assets), 0.99), ss.K)
    capital_grid = collect(range(0.0, k_max, length=points))
    consumption = zeros(points, length(productivity_levels))
    wealth = similar(consumption)
    base_w = vec(base_state.w[1, :])
    base_y = vec(base_state.y[1, :])
    base_z = base_state.z[1]
    for (j, y_value) in enumerate(productivity_levels), (i, k_value) in enumerate(capital_grid)
        w_value = ss.gross_return * k_value + ss.wage * y_value
        w = reshape(copy(base_w), 1, :)
        y = reshape(copy(base_y), 1, :)
        w[1, agent] = w_value
        y[1, agent] = y_value
        result = policy(theta, w, y, [base_z], p, ss)
        wealth[i, j] = w_value
        consumption[i, j] = result.c[1, agent]
    end
    monotonicity_violations = [
        count(<(-1e-5), diff(consumption[:, j]))
        for j in axes(consumption, 2)
    ]
    return (; capital_grid, wealth, consumption, productivity_levels,
            monotonicity_violations)
end

"""Consumption and savings slices on the main notebook's common rectangular (w,y) grid."""
function rectangular_policy_slice(theta, base_state, p::KSParams, ss=steady_state(p);
                                  agent=1, points=100, productivity_points=7)
    1 <= agent <= p.agents || throw(ArgumentError("agent index out of range"))
    bounds = _state_bounds(p, ss)
    wealth_grid = collect(range(0.05 * ss.W,
                                p.wealth_scale_multiple * ss.W,
                                length=points))
    log_y_grid = collect(range(-p.productivity_sd_bound * bounds.y_log_sd,
                               p.productivity_sd_bound * bounds.y_log_sd,
                               length=productivity_points))
    productivity_grid = exp.(log_y_grid)
    consumption = zeros(points, productivity_points)
    savings = zeros(points, productivity_points)

    base_w = vec(base_state.w[1, :])
    base_y = vec(base_state.y[1, :])
    base_z = base_state.z[1]
    for (j, y_value) in enumerate(productivity_grid),
        (i, w_value) in enumerate(wealth_grid)
        w = reshape(copy(base_w), 1, :)
        y = reshape(copy(base_y), 1, :)
        w[1, agent] = w_value
        y[1, agent] = y_value
        result = policy(theta, w, y, [base_z], p, ss)
        consumption[i, j] = result.c[1, agent]
        savings[i, j] = result.kp[1, agent]
    end
    return (; wealth_grid, productivity_grid, consumption, savings)
end

end
