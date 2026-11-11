include(joinpath(@__DIR__, "ks1998_kshock_experiment.jl"))

models_path = joinpath(PROJECT_ROOT, "experiments", "results",
                       "ks1998_kshock_models_full.jls")
models = deserialize(models_path)
p, ss = models.p, models.ss

validation_rows = [
    merge((; K_trained=2), fixed_bank_validation(
        models.result2.theta, models.fixed_bank, p, ss; n_shocks=128, seed=99)),
    merge((; K_trained=4), fixed_bank_validation(
        models.result4.theta, models.fixed_bank, p, ss; n_shocks=128, seed=99)),
]

println("Measuring estimator noise at the trained K=4 solution")
noise_at_k4 = [
    merge((; theta_trained_with_K=4), gradient_noise_diagnostic(
        models.result4.theta, models.fixed_bank, p, ss; K=2)),
    merge((; theta_trained_with_K=4), gradient_noise_diagnostic(
        models.result4.theta, models.fixed_bank, p, ss; K=4)),
]

results_dir = joinpath(PROJECT_ROOT, "experiments", "results")
validation_path = joinpath(results_dir, "ks1998_kshock_validation_tail.csv")
noise_path = joinpath(results_dir, "ks1998_kshock_gradient_noise_at_k4.csv")
write_results(validation_path, validation_rows)
write_results(noise_path, noise_at_k4)

println("Fixed-bank validation tails:")
for row in validation_rows
    @printf("trained K=%d | median=%7.4f p90=%7.4f p95=%7.4f p99=%7.4f max=%.4f MSE=%.4e\n",
        row.K_trained, row.median_log10_euler, row.p90_log10_euler,
        row.p95_log10_euler, row.p99_log10_euler,
        row.max_abs_euler, row.conditional_mse)
end
println("Gradient noise at trained K=4 theta:")
for row in noise_at_k4
    @printf("estimator K=%d | loss sd=%.4e Euler grad SNR=%.4f total grad SNR=%.4f\n",
        row.K, row.sd_euler_estimate, row.euler_gradient_snr,
        row.total_gradient_snr)
end
println("Saved $validation_path")
println("Saved $noise_path")
