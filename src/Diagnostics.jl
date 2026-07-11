"""Out-of-sample conditional Euler and complementarity residuals."""
function conditional_residuals(theta, state, p::KSParams, ss=steady_state(p);
                               n_shocks=64, seed=99)
    rng = MersenneTwister(seed)
    current = policy(theta, state.w, state.y, state.z, p, ss)
    euler_sum = zeros(size(state.w))

    for _ in 1:n_shocks
        shocks = draw_shocks(rng, size(state.w, 1), p)
        branch = _future_branch(theta, current, state.y, state.z, shocks, p, ss)
        euler_sum .+= branch.euler
    end

    a = current.mu .- 1
    b = state.w ./ current.c .- 1
    fisher_burmeister = a .+ b .- sqrt.(a .^ 2 .+ b .^ 2)
    return (; euler=euler_sum ./ n_shocks, fisher_burmeister, current)
end

"""Gini coefficient for a nonnegative finite vector."""
function gini_coefficient(values)
    x = sort(max.(collect(values), 0.0))
    total = sum(x)
    total <= 0 && return 0.0
    n = length(x)
    return 2 * sum((1:n) .* x) / (n * total) - (n + 1) / n
end

"""Simulate one economy under a fixed trained policy."""
function simulate_economy(theta, p::KSParams, ss=steady_state(p);
                          periods=600, burn=100, seed=404)
    rng = MersenneTwister(seed)
    state = initial_state(1, p, ss)
    total_periods = periods + burn
    aggregate_k = zeros(periods)
    aggregate_c = zeros(periods)
    aggregate_z = zeros(periods)
    capital_gini = zeros(periods)
    capital_panel = zeros(periods, p.agents)

    saved = 0
    for t in 1:total_periods
        current = policy(theta, state.w, state.y, state.z, p, ss)
        if t > burn
            saved += 1
            aggregate_k[saved] = mean(current.kp)
            aggregate_c[saved] = mean(current.c)
            aggregate_z[saved] = state.z[1]
            capital_gini[saved] = gini_coefficient(vec(current.kp))
            capital_panel[saved, :] .= vec(current.kp)
        end
        state = _advance_state(theta, state, draw_shocks(rng, 1, p), p, ss)
    end

    return (; aggregate_k, aggregate_c, aggregate_z, capital_gini, capital_panel,
            final_state=state)
end

"""Krusell--Smith approximate-aggregation regression and R-squared."""
function approximate_aggregation(simulation)
    K, z = simulation.aggregate_k, simulation.aggregate_z
    length(K) >= 3 || error("simulation needs at least three periods")
    X = hcat(ones(length(K) - 1), log.(K[1:(end - 1)]), log.(z[1:(end - 1)]))
    target = log.(K[2:end])
    coefficients = X \ target
    fitted = X * coefficients
    ss_residual = sum((target .- fitted) .^ 2)
    ss_total = sum((target .- mean(target)) .^ 2)
    r2 = ss_total > 0 ? 1 - ss_residual / ss_total : 1.0
    return (; coefficients, r2, fitted, target)
end

"""Consumption and savings slices holding the rest of one cross section fixed."""
function policy_slice(theta, base_state, p::KSParams, ss=steady_state(p);
                      agent=1, points=100, productivity_points=7)
    1 <= agent <= p.agents || throw(ArgumentError("agent index out of range"))
    bounds = _state_bounds(p, ss)
    wealth_grid = collect(range(0.05 * ss.W, p.wealth_cap_multiple * ss.W, length=points))
    log_y_grid = collect(range(-p.productivity_sd_bound * bounds.y_log_sd,
                               p.productivity_sd_bound * bounds.y_log_sd,
                               length=productivity_points))
    productivity_grid = exp.(log_y_grid)
    consumption = zeros(points, productivity_points)
    savings = zeros(points, productivity_points)

    base_w = vec(base_state.w[1, :])
    base_y = vec(base_state.y[1, :])
    base_z = base_state.z[1]
    for (j, y_value) in enumerate(productivity_grid), (i, w_value) in enumerate(wealth_grid)
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
