include(joinpath(@__DIR__, "ks1998_kshock_experiment.jl"))

models_path = joinpath(PROJECT_ROOT, "experiments", "results",
                       "ks1998_kshock_models_full.jls")
models = deserialize(models_path)
theta, p, ss = models.result4.theta, models.p, models.ss

"""Simulate correctly aligned (K_t, z_t, distribution_t, K_{t+1}) observations."""
function simulate_plm_data(theta, p::KSParams, ss;
                           periods=20_000, burn=2_000, seed=4404)
    rng = MersenneTwister(seed)
    state = initial_state(1, p, ss)
    current_capital = fill(ss.K, 1, p.agents)
    total_periods = periods + burn

    log_k = zeros(periods)
    log_k_prime = zeros(periods)
    log_z = zeros(periods)
    capital_gini = zeros(periods)
    capital_cv = zeros(periods)
    capital_spread = zeros(periods)
    constrained_share = zeros(periods)
    wealth_gini = zeros(periods)
    wealth_cv = zeros(periods)
    wealth_spread = zeros(periods)
    productivity_cv = zeros(periods)
    capital_productivity_correlation = zeros(periods)

    saved = 0
    for period in 1:total_periods
        current_policy = policy(theta, state.w, state.y, state.z, p, ss)
        if period > burn
            saved += 1
            capital_vector = vec(current_capital)
            mean_capital = mean(capital_vector)
            next_mean_capital = mean(current_policy.kp)
            log_k[saved] = log(mean_capital)
            log_k_prime[saved] = log(next_mean_capital)
            log_z[saved] = log(state.z[1])
            capital_gini[saved] = gini_coefficient(capital_vector)
            capital_cv[saved] = std(capital_vector) / mean_capital
            capital_spread[saved] = (
                quantile(capital_vector, 0.90) - quantile(capital_vector, 0.10)
            ) / mean_capital
            constrained_share[saved] = mean(capital_vector .< 1e-8)
            wealth_vector = vec(state.w)
            productivity_vector = vec(state.y)
            mean_wealth = mean(wealth_vector)
            wealth_gini[saved] = gini_coefficient(wealth_vector)
            wealth_cv[saved] = std(wealth_vector) / mean_wealth
            wealth_spread[saved] = (
                quantile(wealth_vector, 0.90) - quantile(wealth_vector, 0.10)
            ) / mean_wealth
            productivity_cv[saved] = std(productivity_vector) / mean(productivity_vector)
            capital_productivity_correlation[saved] = cor(capital_vector, productivity_vector)
        end

        shocks = draw_shocks(rng, 1, p)
        state = _advance_state(theta, state, shocks, p, ss)
        current_capital = current_policy.kp
    end

    return (; log_k, log_k_prime, log_z, capital_gini, capital_cv,
            capital_spread, constrained_share, wealth_gini, wealth_cv,
            wealth_spread, productivity_cv, capital_productivity_correlation,
            seed=fill(seed, periods), time=collect(1:periods))
end

function stack_data(paths)
    fields = (:log_k, :log_k_prime, :log_z, :capital_gini, :capital_cv,
              :capital_spread, :constrained_share, :wealth_gini, :wealth_cv,
              :wealth_spread, :productivity_cv,
              :capital_productivity_correlation, :seed, :time)
    return NamedTuple{fields}(Tuple(vcat((getproperty(path, field)
                                         for path in paths)...)
                                    for field in fields))
end

function subset(data, mask)
    fields = propertynames(data)
    return NamedTuple{fields}(Tuple(getproperty(data, field)[mask] for field in fields))
end

function design_matrix(data, specification)
    one = ones(length(data.log_k))
    if specification === :k_only
        return hcat(one, data.log_k)
    elseif specification === :k_z
        return hcat(one, data.log_k, data.log_z)
    elseif specification === :ks_regimes
        good = Float64.(data.log_z .>= 0)
        return hcat(one, data.log_k, good, good .* data.log_k)
    elseif specification === :k_z_flexible
        return hcat(one, data.log_k, data.log_z, data.log_z .^ 2,
                    data.log_k .* data.log_z)
    elseif specification === :k_z_distribution
        return hcat(one, data.log_k, data.log_z, data.log_z .^ 2,
                    data.log_k .* data.log_z, data.capital_gini,
                    data.capital_cv, data.capital_spread,
                    data.constrained_share, data.wealth_gini, data.wealth_cv,
                    data.wealth_spread, data.productivity_cv,
                    data.capital_productivity_correlation)
    end
    throw(ArgumentError("unknown specification $specification"))
end

function r2_score(actual, fitted)
    sse = sum((actual .- fitted) .^ 2)
    sst = sum((actual .- mean(actual)) .^ 2)
    return 1 - sse / sst
end

function regression_row(name, specification, train, test)
    x_train = design_matrix(train, specification)
    x_test = design_matrix(test, specification)
    coefficients = x_train \ train.log_k_prime
    fitted_train = x_train * coefficients
    fitted_test = x_test * coefficients
    level_percent_error = 100 .* (exp.(fitted_test) ./ exp.(test.log_k_prime) .- 1)
    n, q = size(x_train)
    r2_train = r2_score(train.log_k_prime, fitted_train)
    return (; name, predictors=q - 1,
            train_r2=r2_train,
            adjusted_train_r2=1 - (1 - r2_train) * (n - 1) / (n - q),
            test_r2=r2_score(test.log_k_prime, fitted_test),
            test_rmse_log=sqrt(mean((test.log_k_prime .- fitted_test) .^ 2)),
            test_rmse_percent=sqrt(mean(level_percent_error .^ 2)),
            test_p99_abs_percent=quantile(abs.(level_percent_error), 0.99),
            intercept=coefficients[1],
            log_k_coefficient=coefficients[2],
            coefficient_3=q >= 3 ? coefficients[3] : NaN)
end

function identity_row(test)
    fitted = test.log_k
    level_percent_error = 100 .* (exp.(fitted) ./ exp.(test.log_k_prime) .- 1)
    return (; name="identity K'=K", predictors=0,
            train_r2=NaN, adjusted_train_r2=NaN,
            test_r2=r2_score(test.log_k_prime, fitted),
            test_rmse_log=sqrt(mean((test.log_k_prime .- fitted) .^ 2)),
            test_rmse_percent=sqrt(mean(level_percent_error .^ 2)),
            test_p99_abs_percent=quantile(abs.(level_percent_error), 0.99),
            intercept=0.0, log_k_coefficient=1.0, coefficient_3=NaN)
end

seeds = 4404:4408
println("Simulating $(length(seeds)) independent PLM paths")
paths = [simulate_plm_data(theta, p, ss; periods=20_000, burn=2_000, seed=seed)
         for seed in seeds]
data = stack_data(paths)

# Preserve time ordering within every path: first 70% estimation, final 30% test.
train_mask = data.time .<= 14_000
test_mask = data.time .> 14_000
train = subset(data, train_mask)
test = subset(data, test_mask)

specifications = [
    ("log K only", :k_only),
    ("log K + log z", :k_z),
    ("two z regimes", :ks_regimes),
    ("flexible aggregate", :k_z_flexible),
    ("aggregate + distribution", :k_z_distribution),
]
rows = [identity_row(test)]
append!(rows, [regression_row(name, specification, train, test)
               for (name, specification) in specifications])

# Original-KS-style separate laws for z below/above one.
regime_rows = NamedTuple[]
for (regime_name, predicate) in [
    ("z below one", x -> x < 0),
    ("z at or above one", x -> x >= 0),
]
    train_regime = subset(train, predicate.(train.log_z))
    test_regime = subset(test, predicate.(test.log_z))
    row = regression_row(regime_name, :k_only, train_regime, test_regime)
    push!(regime_rows, merge(row, (;
        train_observations=length(train_regime.log_k),
        test_observations=length(test_regime.log_k))))
end

# Per-seed stability for the headline log-K-only regression.
seed_rows = NamedTuple[]
for seed in seeds
    seed_train = subset(train, train.seed .== seed)
    seed_test = subset(test, test.seed .== seed)
    row = regression_row("seed $seed", :k_only, seed_train, seed_test)
    push!(seed_rows, merge(row, (; seed)))
end

results_dir = joinpath(PROJECT_ROOT, "experiments", "results")
mkpath(results_dir)
main_path = joinpath(results_dir, "ks1998_plm_regression.csv")
regime_path = joinpath(results_dir, "ks1998_plm_regimes.csv")
seed_path = joinpath(results_dir, "ks1998_plm_seed_stability.csv")
write_results(main_path, rows)
write_results(regime_path, regime_rows)
write_results(seed_path, seed_rows)

println("\nPooled PLM results (70,000 train; 30,000 time-ordered test observations):")
for row in rows
    @printf("%-26s train R2=%9s test R2=%.8f RMSE=%7.4f%% p99=%7.4f%%\n",
        row.name,
        isnan(row.train_r2) ? "--" : @sprintf("%.8f", row.train_r2),
        row.test_r2, row.test_rmse_percent, row.test_p99_abs_percent)
end

println("\nSeparate original-KS-style regime laws:")
for row in regime_rows
    @printf("%-22s test R2=%.8f RMSE=%7.4f%% n_test=%d\n",
        row.name, row.test_r2, row.test_rmse_percent, row.test_observations)
end

println("\nHeadline data range:")
@printf("mean K=%.4f; sd(log K)=%.5f; min K=%.4f; max K=%.4f\n",
    mean(exp.(data.log_k)), std(data.log_k),
    minimum(exp.(data.log_k)), maximum(exp.(data.log_k)))
@printf("mean z=%.5f; sd(log z)=%.5f\n",
    mean(exp.(data.log_z)), std(data.log_z))

println("Saved $main_path")
println("Saved $regime_path")
println("Saved $seed_path")
