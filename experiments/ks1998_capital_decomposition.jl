include(joinpath(@__DIR__, "ks1998_kshock_experiment.jl"))

models_path = joinpath(PROJECT_ROOT, "experiments", "results",
                       "ks1998_kshock_models_full.jls")
models = deserialize(models_path)
theta = models.result4.theta

configs = [
    (name="no_shocks", sigma_y=0.0, sigma_z=0.0),
    (name="idiosyncratic_only", sigma_y=0.20, sigma_z=0.0),
    (name="aggregate_only", sigma_y=0.0, sigma_z=0.01),
    (name="full_risk", sigma_y=0.20, sigma_z=0.01),
]

rows = NamedTuple[]
for config in configs
    p = KSParams(agents=50, sigma_y=config.sigma_y, sigma_z=config.sigma_z,
                 fb_convention=:released_code)
    ss = steady_state(p)
    simulation = simulate_economy(theta, p, ss;
        periods=5000, burn=1000, seed=4404)
    capital = simulation.aggregate_k
    push!(rows, (; name=config.name, sigma_y=config.sigma_y,
        sigma_z=config.sigma_z, deterministic_k=ss.K,
        mean_k=mean(capital), median_k=median(capital),
        p10_k=quantile(capital, 0.10), p90_k=quantile(capital, 0.90),
        fraction_above_deterministic=mean(capital .> ss.K),
        mean_gap_percent=100 * (mean(capital) / ss.K - 1)))
end

output_path = joinpath(PROJECT_ROOT, "experiments", "results",
                       "ks1998_capital_decomposition.csv")
write_results(output_path, rows)
for row in rows
    @printf("%-20s mean K=%.4f gap=%6.2f%% above Kss=%5.1f%% p10=%.4f p90=%.4f\n",
        row.name, row.mean_k, row.mean_gap_percent,
        100 * row.fraction_above_deterministic, row.p10_k, row.p90_k)
end
println("Saved $output_path")

# Check the risky-economy policy at the deterministic symmetric state. This is
# more informative than treating a no-realized-shock path as a re-solved
# deterministic economy: the policy still anticipates the full shock process.
p_full = KSParams(agents=50, sigma_y=0.20, sigma_z=0.01,
                  fb_convention=:released_code)
ss_full = steady_state(p_full)
symmetric_state = initial_state(1, p_full, ss_full)
symmetric_policy = policy(theta, symmetric_state.w, symmetric_state.y,
                          symmetric_state.z, p_full, ss_full)
symmetric_diagnostics = conditional_residuals(
    theta, symmetric_state, p_full, ss_full; n_shocks=2048, seed=5511)
symmetric_euler = vec(symmetric_diagnostics.euler)

@printf("\nRisky policy evaluated at deterministic symmetric state:\n")
@printf("K_ss=%.4f; chosen mean k'=%.4f; gap=%+.2f%%\n",
    ss_full.K, mean(symmetric_policy.kp),
    100 * (mean(symmetric_policy.kp) / ss_full.K - 1))
@printf("mean xi=%.4f; mean mu=%.4f\n",
    mean(symmetric_policy.xi), mean(symmetric_policy.mu))
@printf("2048-shock conditional Euler: mean=%+.4e mean|.|=%.4e max|.|=%.4e\n",
    mean(symmetric_euler), mean(abs, symmetric_euler), maximum(abs, symmetric_euler))
