include(joinpath(@__DIR__, "ks1998_kshock_experiment.jl"))

using Dates

"""Prepare shock arrays outside the differentiated computation."""
function stack_branch_shocks(shocks)
    return (;
        eps_y=vcat(map(shock -> shock.eps_y, shocks)...),
        eps_z=vcat(map(shock -> shock.eps_z, shocks)...),
    )
end

"""Stack K independent future branches along the economy-batch dimension."""
function kshock_loss_components_stacked(theta, state, stacked_shocks, K::Int,
                                        p::KSParams, ss=steady_state(p))
    K >= 2 || throw(ArgumentError("K must be at least 2"))
    B, N = size(state.w)
    current = policy(theta, state.w, state.y, state.z, p, ss)

    repeated_y = repeat(state.y, K, 1)
    repeated_z = repeat(state.z, K)
    repeated_kp = repeat(current.kp, K, 1)
    repeated_c = repeat(current.c, K, 1)
    repeated_mu = repeat(current.mu, K, 1)

    y_next, z_next = transition_productivity(repeated_y, repeated_z,
                                              stacked_shocks, p)
    prices = next_period(repeated_kp, y_next, z_next, p)
    next_policy = policy(theta, prices.wealth_next, y_next, z_next, p, ss)
    euler_flat = p.beta .* (repeated_c ./ next_policy.c) .^ p.gamma .*
                 reshape(prices.gross_return, :, 1) .- repeated_mu
    residual_cube = permutedims(reshape(euler_flat, B, K, N), (2, 1, 3))
    residual_sum = dropdims(sum(residual_cube; dims=1); dims=1)
    residual_square_sum = dropdims(sum(residual_cube .^ 2; dims=1); dims=1)
    euler_u = mean((residual_sum .^ 2 .- residual_square_sum) ./ (K * (K - 1)))
    fisher_burmeister = complementarity_residual(current, state, p)
    complementarity = mean(fisher_burmeister .^ 2)

    first_rows = 1:B
    first_prices = (;
        aggregate_k=prices.aggregate_k[first_rows],
        rental=prices.rental[first_rows],
        wage=prices.wage[first_rows],
        gross_return=prices.gross_return[first_rows],
        wealth_next=prices.wealth_next[first_rows, :],
    )
    first_branch = (;
        y_next=y_next[first_rows, :],
        z_next=z_next[first_rows],
        prices=first_prices,
        euler=euler_flat[first_rows, :],
    )
    return (; loss=complementarity + euler_u, complementarity,
            euler=euler_u, current, first_branch)
end

kshock_loss_stacked(theta, state, stacked_shocks, K::Int,
                    p::KSParams, ss=steady_state(p)) =
    kshock_loss_components_stacked(theta, state, stacked_shocks, K, p, ss).loss

"""Threaded forward evaluation of independent branches; tested for AD support."""
function kshock_loss_components_threaded(theta, state, shocks,
                                         p::KSParams, ss=steady_state(p))
    K = length(shocks)
    current = policy(theta, state.w, state.y, state.z, p, ss)
    tasks = [Threads.@spawn _future_branch(theta, current, state.y, state.z,
                                           shock, p, ss)
             for shock in shocks]
    branches = fetch.(tasks)
    residuals = [branch.euler for branch in branches]
    residual_sum = reduce(+, residuals)
    residual_square_sum = reduce(+, (residual .^ 2 for residual in residuals))
    euler_u = mean((residual_sum .^ 2 .- residual_square_sum) ./ (K * (K - 1)))
    fisher_burmeister = complementarity_residual(current, state, p)
    complementarity = mean(fisher_burmeister .^ 2)
    return (; loss=complementarity + euler_u, complementarity,
            euler=euler_u, current, branches)
end

kshock_loss_threaded(theta, state, shocks, p::KSParams, ss=steady_state(p)) =
    kshock_loss_components_threaded(theta, state, shocks, p, ss).loss

function median_elapsed(f; repetitions=5)
    f() # compile and warm caches
    times = Float64[]
    for _ in 1:repetitions
        GC.gc()
        started = time_ns()
        f()
        push!(times, (time_ns() - started) / 1e9)
    end
    return (; median=median(times), minimum=minimum(times), maximum=maximum(times),
            repetitions)
end

function benchmark_row(name, implementation, operation, blas_threads, timing;
                       valid=true, note="")
    return (; name, implementation, operation, blas_threads,
            julia_threads=Threads.nthreads(), seconds=timing.median,
            minimum_seconds=timing.minimum, maximum_seconds=timing.maximum,
            repetitions=timing.repetitions, valid, note)
end

function main()
    p = KSParams(agents=200, hidden=64, fb_convention=:released_code)
    ss = steady_state(p)
    theta = init_theta(MersenneTwister(1234), p; style=:reference)
    state = initial_state(10, p, ss)
    state_rng = MersenneTwister(8080)
    for _ in 1:8
        state = _advance_state(theta, state, draw_shocks(state_rng, 10, p), p, ss)
    end
    branch_rngs = [MersenneTwister(2026 + 100_000 * branch) for branch in 1:4]
    shocks = [draw_shocks(branch_rngs[branch], 10, p) for branch in 1:4]
    stacked_shocks = stack_branch_shocks(shocks)

    serial_components = kshock_loss_components(theta, state, shocks, p, ss)
    stacked_components = kshock_loss_components_stacked(
        theta, state, stacked_shocks, length(shocks), p, ss,
    )
    serial_gradient = Flux.gradient(
        th -> kshock_loss(th, state, shocks, p, ss), theta,
    )[1]
    stacked_gradient = Flux.gradient(
        th -> kshock_loss_stacked(th, state, stacked_shocks,
                                  length(shocks), p, ss), theta,
    )[1]
    loss_abs_difference = abs(serial_components.loss - stacked_components.loss)
    gradient_max_abs_difference = maximum(abs.(serial_gradient .- stacked_gradient))
    gradient_relative_l2_difference = (
        norm(serial_gradient .- stacked_gradient) /
        max(norm(serial_gradient), eps(Float64))
    )
    branch1_wealth_max_abs_difference = maximum(abs.(
        serial_components.branches[1].prices.wealth_next .-
        stacked_components.first_branch.prices.wealth_next
    ))
    stacked_valid = loss_abs_difference < 1e-12 &&
                    gradient_relative_l2_difference < 1e-10 &&
                    branch1_wealth_max_abs_difference < 1e-12

    @printf("configuration N=%d H=%d B=%d K=%d parameters=%d\n",
            p.agents, p.hidden, size(state.w, 1), length(shocks), length(theta))
    @printf("stacked equivalence: |dloss|=%.3e max|dgrad|=%.3e relL2=%.3e branch1=%.3e valid=%s\n",
            loss_abs_difference, gradient_max_abs_difference,
            gradient_relative_l2_difference,
            branch1_wealth_max_abs_difference, string(stacked_valid))

    rows = NamedTuple[]
    serial_forward = () -> kshock_loss_components(theta, state, shocks, p, ss)
    stacked_forward = () -> kshock_loss_components_stacked(
        theta, state, stacked_shocks, length(shocks), p, ss,
    )
    serial_gradient_only = () -> Flux.gradient(
        th -> kshock_loss(th, state, shocks, p, ss), theta,
    )[1]
    stacked_gradient_only = () -> Flux.gradient(
        th -> kshock_loss_stacked(th, state, stacked_shocks,
                                  length(shocks), p, ss), theta,
    )[1]
    current_update = () -> begin
        kshock_loss_components(theta, state, shocks, p, ss)
        Flux.gradient(th -> kshock_loss(th, state, shocks, p, ss), theta)[1]
    end
    withgradient_serial = () -> Flux.withgradient(theta) do th
        components = kshock_loss_components(th, state, shocks, p, ss)
        components.loss, components
    end
    withgradient_stacked = () -> Flux.withgradient(theta) do th
        components = kshock_loss_components_stacked(
            th, state, stacked_shocks, length(shocks), p, ss,
        )
        components.loss, components
    end

    for blas_threads in (1, 2, 4, 6, 12)
        BLAS.set_num_threads(blas_threads)
        println("benchmarking BLAS threads=$blas_threads")
        for (name, implementation, operation, f, valid) in [
            ("serial forward", "serial", "forward", serial_forward, true),
            ("stacked forward", "stacked", "forward", stacked_forward, stacked_valid),
            ("serial gradient", "serial", "gradient", serial_gradient_only, true),
            ("stacked gradient", "stacked", "gradient", stacked_gradient_only, stacked_valid),
            ("current update path", "serial", "forward+gradient", current_update, true),
            ("withgradient serial", "serial_withgradient", "forward+gradient", withgradient_serial, true),
            ("withgradient stacked", "stacked_withgradient", "forward+gradient", withgradient_stacked, stacked_valid),
        ]
            timing = median_elapsed(f; repetitions=5)
            push!(rows, benchmark_row(name, implementation, operation,
                                      blas_threads, timing; valid))
            @printf("  %-24s %.4fs [%.4f, %.4f]\n",
                    name, timing.median, timing.minimum, timing.maximum)
        end
    end

    threaded_gradient_valid = false
    threaded_note = ""
    threaded_loss_difference = NaN
    threaded_gradient_relative_difference = NaN
    try
        BLAS.set_num_threads(1)
        threaded_components = kshock_loss_components_threaded(
            theta, state, shocks, p, ss,
        )
        threaded_gradient = Flux.gradient(
            th -> kshock_loss_threaded(th, state, shocks, p, ss), theta,
        )[1]
        threaded_loss_difference = abs(threaded_components.loss - serial_components.loss)
        threaded_gradient_relative_difference = (
            norm(threaded_gradient .- serial_gradient) /
            max(norm(serial_gradient), eps(Float64))
        )
        threaded_gradient_valid = threaded_loss_difference < 1e-12 &&
                                  threaded_gradient_relative_difference < 1e-10
        threaded_note = @sprintf("|dloss|=%.3e relL2grad=%.3e",
                                 threaded_loss_difference,
                                 threaded_gradient_relative_difference)
        threaded_forward = () -> kshock_loss_components_threaded(
            theta, state, shocks, p, ss,
        )
        threaded_gradient_only = () -> Flux.gradient(
            th -> kshock_loss_threaded(th, state, shocks, p, ss), theta,
        )[1]
        for (name, operation, f) in [
            ("threaded forward", "forward", threaded_forward),
            ("threaded gradient", "gradient", threaded_gradient_only),
        ]
            timing = median_elapsed(f; repetitions=5)
            push!(rows, benchmark_row(name, "threaded", operation, 1, timing;
                                      valid=threaded_gradient_valid,
                                      note=threaded_note))
        end
    catch error
        threaded_note = sprint(showerror, error)
        println("threaded gradient rejected: $threaded_note")
    end

    results_dir = joinpath(PROJECT_ROOT, "experiments", "results")
    mkpath(results_dir)
    timing_path = joinpath(results_dir, "ks1998_acceleration_cpu.csv")
    validation_path = joinpath(results_dir, "ks1998_acceleration_validation.csv")
    write_results(timing_path, rows)
    validation = [(;
        timestamp=string(now()), agents=p.agents, hidden=p.hidden,
        batch_size=size(state.w, 1), branches=length(shocks),
        parameters=length(theta), loss_abs_difference,
        gradient_max_abs_difference, gradient_relative_l2_difference,
        branch1_wealth_max_abs_difference, stacked_valid,
        threaded_gradient_valid, threaded_loss_difference,
        threaded_gradient_relative_difference, threaded_note,
    )]
    write_results(validation_path, validation)
    println("saved $timing_path")
    println("saved $validation_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
