include(joinpath(@__DIR__, "ks1998_acceleration_benchmark.jl"))

using Dates

function train_short(theta_initial, p, ss; mode=:baseline, periods=60,
                     batch_size=10, update_every=2, seed=2026, lr=1e-3)
    theta = copy(theta_initial)
    branch_rngs = [MersenneTwister(seed + 100_000 * branch) for branch in 1:4]
    state = initial_state(batch_size, p, ss)
    optimizer = AdamState(length(theta))
    losses = Float64[]

    for period in 1:periods
        shocks = [draw_shocks(branch_rngs[branch], batch_size, p)
                  for branch in 1:4]
        is_update = (period - 1) % update_every == 0

        if mode === :baseline
            components = kshock_loss_components(theta, state, shocks, p, ss)
            if is_update
                gradient = Flux.gradient(
                    th -> kshock_loss(th, state, shocks, p, ss), theta,
                )[1]
                _adam_step!(theta, gradient, optimizer; lr)
            end
            first_branch = components.branches[1]
        elseif mode === :withgradient_serial
            if is_update
                value_gradient = Flux.withgradient(theta) do th
                    candidate = kshock_loss_components(th, state, shocks, p, ss)
                    candidate.loss, candidate
                end
                components = value_gradient.val[2]
                gradient = value_gradient.grad[1]
                _adam_step!(theta, gradient, optimizer; lr)
            else
                components = kshock_loss_components(theta, state, shocks, p, ss)
            end
            first_branch = components.branches[1]
        elseif mode === :withgradient_stacked
            stacked_shocks = stack_branch_shocks(shocks)
            if is_update
                value_gradient = Flux.withgradient(theta) do th
                    candidate = kshock_loss_components_stacked(
                        th, state, stacked_shocks, 4, p, ss,
                    )
                    candidate.loss, candidate
                end
                components = value_gradient.val[2]
                gradient = value_gradient.grad[1]
                _adam_step!(theta, gradient, optimizer; lr)
            else
                components = kshock_loss_components_stacked(
                    theta, state, stacked_shocks, 4, p, ss,
                )
            end
            first_branch = components.first_branch
        else
            throw(ArgumentError("unknown mode $mode"))
        end

        push!(losses, components.loss)
        state = (; w=first_branch.prices.wealth_next,
                 y=first_branch.y_next, z=first_branch.z_next)
    end
    return (; theta, state, losses, updates=optimizer.t)
end

function timed_training(theta, p, ss; mode, periods, repetitions=3)
    train_short(theta, p, ss; mode, periods=4) # compile
    times = Float64[]
    final_result = nothing
    for _ in 1:repetitions
        GC.gc()
        started = time_ns()
        final_result = train_short(theta, p, ss; mode, periods)
        push!(times, (time_ns() - started) / 1e9)
    end
    return (; seconds=median(times), minimum_seconds=minimum(times),
            maximum_seconds=maximum(times), repetitions, result=final_result)
end

function comparison_metrics(reference, candidate)
    theta_relative_l2 = norm(reference.theta .- candidate.theta) /
                        max(norm(reference.theta), eps(Float64))
    return (;
        theta_relative_l2,
        theta_max_abs=maximum(abs.(reference.theta .- candidate.theta)),
        state_w_max_abs=maximum(abs.(reference.state.w .- candidate.state.w)),
        state_y_max_abs=maximum(abs.(reference.state.y .- candidate.state.y)),
        state_z_max_abs=maximum(abs.(reference.state.z .- candidate.state.z)),
        loss_max_abs=maximum(abs.(reference.losses .- candidate.losses)),
    )
end

function main_training_benchmark()
    p = KSParams(agents=200, hidden=64, fb_convention=:released_code)
    ss = steady_state(p)
    theta = init_theta(MersenneTwister(1234), p; style=:reference)
    periods = isempty(ARGS) ? 60 : parse(Int, ARGS[1])
    rows = NamedTuple[]

    configurations = [
        (mode=:baseline, blas_threads=6),
        (mode=:baseline, blas_threads=12),
        (mode=:withgradient_serial, blas_threads=4),
        (mode=:withgradient_serial, blas_threads=6),
        (mode=:withgradient_stacked, blas_threads=6),
        (mode=:withgradient_stacked, blas_threads=12),
    ]
    timings = Dict{Tuple{Symbol,Int},Any}()
    for configuration in configurations
        BLAS.set_num_threads(configuration.blas_threads)
        println("training mode=$(configuration.mode) BLAS=$(configuration.blas_threads)")
        timing = timed_training(theta, p, ss;
                                mode=configuration.mode, periods,
                                repetitions=3)
        timings[(configuration.mode, configuration.blas_threads)] = timing
        push!(rows, (;
            mode=string(configuration.mode),
            blas_threads=configuration.blas_threads,
            julia_threads=Threads.nthreads(), periods,
            updates=timing.result.updates,
            seconds=timing.seconds,
            seconds_per_period=timing.seconds / periods,
            minimum_seconds=timing.minimum_seconds,
            maximum_seconds=timing.maximum_seconds,
            repetitions=timing.repetitions,
        ))
        @printf("  %.3fs total, %.4fs/period [%.3f, %.3f]\n",
                timing.seconds, timing.seconds / periods,
                timing.minimum_seconds, timing.maximum_seconds)
    end

    reference = timings[(:baseline, 6)].result
    validation_rows = NamedTuple[]
    for configuration in configurations[2:end]
        candidate = timings[(configuration.mode, configuration.blas_threads)].result
        metrics = comparison_metrics(reference, candidate)
        push!(validation_rows, (;
            mode=string(configuration.mode),
            blas_threads=configuration.blas_threads,
            metrics...,
        ))
        @printf("validation %-22s BLAS=%2d theta_rel=%.3e loss_max=%.3e state_w=%.3e\n",
                string(configuration.mode), configuration.blas_threads,
                metrics.theta_relative_l2, metrics.loss_max_abs,
                metrics.state_w_max_abs)
    end

    results_dir = joinpath(PROJECT_ROOT, "experiments", "results")
    mkpath(results_dir)
    timing_path = joinpath(results_dir, "ks1998_acceleration_training.csv")
    validation_path = joinpath(results_dir,
                               "ks1998_acceleration_training_validation.csv")
    write_results(timing_path, rows)
    write_results(validation_path, validation_rows)
    println("saved $timing_path")
    println("saved $validation_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_training_benchmark()
end
