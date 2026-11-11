using Serialization

include(joinpath(project_root, "src", "Diagnostics.jl"))

par.agents == 400 || error("This diagnostic expects N=400, got $(par.agents)")

results_dir = joinpath(project_root, "experiments", "results")
output_dir = joinpath(project_root, "output")
mkpath(results_dir)
mkpath(output_dir)

println("THREE_TESTS_TRAINING_BEGIN")
theta0 = init_theta(MersenneTwister(1234), par, ss)
result = train!(theta0, par, ss;
    periods=30000,
    batch_size=10,
    update_every=2,
    lr=1e-3,
    seed=2026,
    display_step=5000,
    device=:gpu,
)
println("THREE_TESTS_TRAINING_END elapsed_seconds=$(result.elapsed_seconds)")

checkpoint_path = joinpath(output_dir,
    "KS1998_N400_H64_K4_P30000_seed2026_gpu_three_tests.jls")
serialize(checkpoint_path, (; par, ss, result))

function evaluate_slice(theta, base_state, p, ss, wealth_grid, productivity_grid;
                        agent=1, chunk_size=10)
    points = length(wealth_grid)
    consumption = zeros(points, length(productivity_grid))
    savings = zeros(points, length(productivity_grid))
    base_w = vec(base_state.w[1, :])
    base_y = vec(base_state.y[1, :])
    base_z = base_state.z[1]

    for (j, y_value) in enumerate(productivity_grid)
        for first_index in 1:chunk_size:points
            last_index = min(first_index + chunk_size - 1, points)
            indices = first_index:last_index
            batch = length(indices)
            w = repeat(reshape(base_w, 1, :), batch, 1)
            y = repeat(reshape(base_y, 1, :), batch, 1)
            w[:, agent] .= wealth_grid[indices]
            y[:, agent] .= y_value
            evaluated = policy(theta, w, y, fill(base_z, batch), p, ss)
            consumption[indices, j] .= evaluated.c[:, agent]
            savings[indices, j] .= evaluated.kp[:, agent]
        end
    end
    return (; wealth_grid, productivity_grid, consumption, savings)
end

function shape_metrics(slice; tolerance=1e-7)
    dc = diff(slice.consumption; dims=1)
    dk = diff(slice.savings; dims=1)
    d2c = diff(slice.consumption; dims=1)
    d2c = diff(d2c; dims=1)
    return (;
        consumption_monotonicity_violation_rate=mean(dc .< -tolerance),
        savings_monotonicity_violation_rate=mean(dk .< -tolerance),
        consumption_concavity_violation_rate=mean(d2c .> tolerance),
        minimum_consumption_increment=minimum(dc),
        minimum_savings_increment=minimum(dk),
        maximum_positive_consumption_second_difference=maximum(d2c),
    )
end

# Test 1: compare the original broad slice with the empirical 1--99% support.
wealth_observations = vec(result.state.w)
wealth_q01 = quantile(wealth_observations, 0.01)
wealth_q99 = quantile(wealth_observations, 0.99)
bounds = _state_bounds(par, ss)
productivity_grid = exp.(collect(range(
    -par.productivity_sd_bound * bounds.y_log_sd,
     par.productivity_sd_bound * bounds.y_log_sd,
    length=7,
)))
full_grid = collect(range(0.0, par.wealth_cap_multiple * ss.W, length=100))
support_grid = collect(range(wealth_q01, wealth_q99, length=100))
full_slice = evaluate_slice(result.theta, result.state, par, ss,
    full_grid, productivity_grid)
support_slice = evaluate_slice(result.theta, result.state, par, ss,
    support_grid, productivity_grid)
full_shape = shape_metrics(full_slice)
support_shape = shape_metrics(support_slice)

println("TEST1_SUPPORT wealth_q01=$(wealth_q01) wealth_q99=$(wealth_q99)")
println("TEST1_FULL $(full_shape)")
println("TEST1_INSUPPORT $(support_shape)")

blue = :steelblue
orange = :darkorange
grey = :grey45
selected_productivity = [1, 4, 7]

p_full_c = plot(xlabel="cash on hand w", ylabel="consumption c",
    title="Broad wealth range", legend=:topleft)
p_full_k = plot(xlabel="cash on hand w", ylabel="next capital k'",
    title="Broad wealth range", legend=:topleft)
p_support_c = plot(xlabel="cash on hand w", ylabel="consumption c",
    title="Simulated 1%-99% wealth support", legend=:topleft)
p_support_k = plot(xlabel="cash on hand w", ylabel="next capital k'",
    title="Simulated 1%-99% wealth support", legend=:topleft)
palette = [blue, grey, orange]
for (color_index, j) in enumerate(selected_productivity)
    label = "y=$(round(productivity_grid[j], digits=2))"
    plot!(p_full_c, full_grid, full_slice.consumption[:, j],
        lw=2, color=palette[color_index], label=label)
    plot!(p_full_k, full_grid, full_slice.savings[:, j],
        lw=2, color=palette[color_index], label=label)
    plot!(p_support_c, support_grid, support_slice.consumption[:, j],
        lw=2, color=palette[color_index], label=label)
    plot!(p_support_k, support_grid, support_slice.savings[:, j],
        lw=2, color=palette[color_index], label=label)
end
policy_figure = plot(p_full_c, p_full_k, p_support_c, p_support_k,
    layout=(2, 2), size=(1100, 720),
    plot_title="N=400 policy slices: broad range versus simulated support")
policy_path = joinpath(results_dir, "ks1998_n400_policy_support_test.png")
savefig(policy_figure, policy_path)

# Test 2: jointly permute household columns and undo the permutation on outputs.
permutation_rng = MersenneTwister(7711)
permutation_state = result.state
baseline_policy = policy(result.theta, permutation_state.w,
    permutation_state.y, permutation_state.z, par, ss)
consumption_errors = Float64[]
savings_errors = Float64[]
for _ in 1:30
    permutation = randperm(permutation_rng, par.agents)
    inverse_permutation = invperm(permutation)
    permuted_policy = policy(result.theta,
        permutation_state.w[:, permutation],
        permutation_state.y[:, permutation],
        permutation_state.z, par, ss)
    append!(consumption_errors,
        vec(permuted_policy.c[:, inverse_permutation] .- baseline_policy.c))
    append!(savings_errors,
        vec(permuted_policy.kp[:, inverse_permutation] .- baseline_policy.kp))
end

permutation_metrics = (;
    consumption_rmse=sqrt(mean(consumption_errors .^ 2)),
    consumption_relative_rmse=sqrt(mean(consumption_errors .^ 2)) /
        sqrt(mean(vec(baseline_policy.c) .^ 2)),
    consumption_p95_absolute=quantile(abs.(consumption_errors), 0.95),
    consumption_max_absolute=maximum(abs.(consumption_errors)),
    savings_rmse=sqrt(mean(savings_errors .^ 2)),
    savings_relative_rmse=sqrt(mean(savings_errors .^ 2)) /
        sqrt(mean(vec(baseline_policy.kp) .^ 2)),
    savings_p95_absolute=quantile(abs.(savings_errors), 0.95),
    savings_max_absolute=maximum(abs.(savings_errors)),
)
println("TEST2_PERMUTATION $(permutation_metrics)")

p_permutation = histogram(log10.(abs.(consumption_errors) ./
        sqrt(mean(vec(baseline_policy.c) .^ 2)) .+ 1e-16),
    bins=45, normalize=:pdf, alpha=0.55, color=blue,
    label="consumption", xlabel="log10 absolute change / policy RMS",
    ylabel="density", title="Household permutation sensitivity")
histogram!(p_permutation, log10.(abs.(savings_errors) ./
        sqrt(mean(vec(baseline_policy.kp) .^ 2)) .+ 1e-16),
    bins=45, normalize=:pdf, alpha=0.45, color=orange, label="savings")
permutation_path = joinpath(results_dir,
    "ks1998_n400_permutation_test.png")
savefig(p_permutation, permutation_path)

# Test 3: long independent simulation and nested PLM specifications.
simulation = simulate_economy(result.theta, par, ss;
    periods=10000, burn=1000, seed=404, start_state=result.state)
log_k = log.(simulation.aggregate_k)
log_z = log.(simulation.aggregate_z)
target = log_k[2:end]
xk = log_k[1:(end - 1)]
xz = log_z[1:(end - 1)]

panel = simulation.capital_panel[1:(end - 1), :]
panel_mean = vec(mean(panel; dims=2))
panel_std = vec(std(panel; dims=2, corrected=false))
panel_cv = panel_std ./ max.(panel_mean, 1e-12)
panel_gini = simulation.capital_gini[1:(end - 1)]
panel_q10 = [quantile(@view(panel[t, :]), 0.10) for t in axes(panel, 1)]
panel_q50 = [quantile(@view(panel[t, :]), 0.50) for t in axes(panel, 1)]
panel_q90 = [quantile(@view(panel[t, :]), 0.90) for t in axes(panel, 1)]
q10_share = panel_q10 ./ max.(panel_mean, 1e-12)
q50_share = panel_q50 ./ max.(panel_mean, 1e-12)
q90_share = panel_q90 ./ max.(panel_mean, 1e-12)

X1 = hcat(ones(length(target)), xk, xz)
X2 = hcat(X1, xk .^ 2, xz .^ 2, xk .* xz)
X3 = hcat(X2, panel_gini, panel_cv, q10_share, q50_share, q90_share)

function regression_metrics(X, y, train_last)
    train_indices = 1:train_last
    test_indices = (train_last + 1):length(y)
    coefficients = X[train_indices, :] \ y[train_indices]
    fitted_train = X[train_indices, :] * coefficients
    fitted_test = X[test_indices, :] * coefficients
    train_sse = sum((y[train_indices] .- fitted_train) .^ 2)
    train_sst = sum((y[train_indices] .- mean(y[train_indices])) .^ 2)
    test_sse = sum((y[test_indices] .- fitted_test) .^ 2)
    test_sst = sum((y[test_indices] .- mean(y[test_indices])) .^ 2)
    return (; coefficients,
        train_r2=1 - train_sse / train_sst,
        test_r2=1 - test_sse / test_sst,
        test_rmse=sqrt(mean((y[test_indices] .- fitted_test) .^ 2)))
end

train_last = floor(Int, 0.70 * length(target))
plm1 = regression_metrics(X1, target, train_last)
plm2 = regression_metrics(X2, target, train_last)
plm3 = regression_metrics(X3, target, train_last)
println("TEST3_PLM_M1 train_r2=$(plm1.train_r2) test_r2=$(plm1.test_r2) test_rmse=$(plm1.test_rmse)")
println("TEST3_PLM_M2 train_r2=$(plm2.train_r2) test_r2=$(plm2.test_r2) test_rmse=$(plm2.test_rmse)")
println("TEST3_PLM_M3 train_r2=$(plm3.train_r2) test_r2=$(plm3.test_r2) test_rmse=$(plm3.test_rmse)")

models = ["M1 linear", "M2 nonlinear", "M3 + distribution"]
train_r2_values = [plm1.train_r2, plm2.train_r2, plm3.train_r2]
test_r2_values = [plm1.test_r2, plm2.test_r2, plm3.test_r2]
p_plm = bar((1:3) .- 0.19, train_r2_values,
    bar_width=0.36, ylim=(minimum(test_r2_values) - 0.005, 1.0),
    color=blue, label="train R²",
    ylabel="R²", title="PLM regression comparison (10,000 periods)",
    legend=:bottomright, xticks=(1:3, models))
bar!(p_plm, (1:3) .+ 0.19, test_r2_values,
    bar_width=0.36, color=orange, label="test R²")
plm_path = joinpath(results_dir, "ks1998_n400_plm_test.png")
savefig(p_plm, plm_path)

summary_path = joinpath(results_dir, "ks1998_n400_three_tests_summary.csv")
open(summary_path, "w") do io
    println(io, "test,metric,value")
    println(io, "policy_support,wealth_q01,$wealth_q01")
    println(io, "policy_support,wealth_q99,$wealth_q99")
    for key in keys(full_shape)
        println(io, "policy_full,$key,$(getproperty(full_shape, key))")
        println(io, "policy_in_support,$key,$(getproperty(support_shape, key))")
    end
    for key in keys(permutation_metrics)
        println(io, "permutation,$key,$(getproperty(permutation_metrics, key))")
    end
    for (name, metrics) in (("M1", plm1), ("M2", plm2), ("M3", plm3))
        println(io, "plm_$name,train_r2,$(metrics.train_r2)")
        println(io, "plm_$name,test_r2,$(metrics.test_r2)")
        println(io, "plm_$name,test_rmse,$(metrics.test_rmse)")
    end
end

println("THREE_TESTS_COMPLETE")
println("checkpoint_path=$checkpoint_path")
println("summary_path=$summary_path")
println("policy_path=$policy_path")
println("permutation_path=$permutation_path")
println("plm_path=$plm_path")
