include(joinpath(@__DIR__, "ks1998_acceleration_benchmark.jl"))

pushfirst!(LOAD_PATH, joinpath(@__DIR__, "cuda_benchmark_env"))
using CUDA

CUDA.allowscalar(false)

to_precision(x::AbstractArray, ::Type{T}) where {T} = T.(x)
state_to_precision(state, ::Type{T}) where {T} =
    (; w=T.(state.w), y=T.(state.y), z=T.(state.z))
shocks_to_precision(shocks, ::Type{T}) where {T} =
    (; eps_y=T.(shocks.eps_y), eps_z=T.(shocks.eps_z))

to_gpu(x::AbstractArray) = CuArray(x)
state_to_gpu(state) = (; w=CuArray(state.w), y=CuArray(state.y), z=CuArray(state.z))
shocks_to_gpu(shocks) =
    (; eps_y=CuArray(shocks.eps_y), eps_z=CuArray(shocks.eps_z))

function gpu_timing(f; repetitions=10)
    f()
    CUDA.synchronize()
    times = Float64[]
    for _ in 1:repetitions
        GC.gc(false)
        CUDA.reclaim()
        started = time_ns()
        f()
        CUDA.synchronize()
        push!(times, (time_ns() - started) / 1e9)
    end
    return (; median=median(times), minimum=minimum(times), maximum=maximum(times),
            repetitions)
end

function cpu_timing(f; repetitions=10)
    f()
    times = Float64[]
    for _ in 1:repetitions
        GC.gc(false)
        started = time_ns()
        f()
        push!(times, (time_ns() - started) / 1e9)
    end
    return (; median=median(times), minimum=minimum(times), maximum=maximum(times),
            repetitions)
end

mutable struct GenericAdamState{A}
    m::A
    v::A
    t::Int
end

GenericAdamState(theta) = GenericAdamState(similar(theta) .= 0,
                                            similar(theta) .= 0, 0)

function generic_adam_step!(theta, gradient, optimizer::GenericAdamState;
                            lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-7)
    optimizer.t += 1
    optimizer.m .= beta1 .* optimizer.m .+ (1 - beta1) .* gradient
    optimizer.v .= beta2 .* optimizer.v .+ (1 - beta2) .* gradient .^ 2
    theta .-= lr .* (optimizer.m ./ (1 - beta1^optimizer.t)) ./
              (sqrt.(optimizer.v ./ (1 - beta2^optimizer.t)) .+ eps)
    return theta
end

function train_stacked_device(theta_initial, p, ss; device=:cpu,
                              periods=100, batch_size=10, update_every=2,
                              seed=2026, lr=1e-3)
    theta = device === :gpu ? CuArray(theta_initial) : copy(theta_initial)
    initial = state_to_precision(initial_state(batch_size, p, ss),
                                 eltype(theta_initial))
    state = device === :gpu ? state_to_gpu(initial) : initial
    optimizer = GenericAdamState(theta)
    branch_rngs = [MersenneTwister(seed + 100_000 * branch) for branch in 1:4]
    losses = Float64[]

    for period in 1:periods
        shocks = [draw_shocks(branch_rngs[branch], batch_size, p)
                  for branch in 1:4]
        stacked = shocks_to_precision(stack_branch_shocks(shocks),
                                      eltype(theta_initial))
        stacked_device = device === :gpu ? shocks_to_gpu(stacked) : stacked
        is_update = (period - 1) % update_every == 0

        if is_update
            value_gradient = Flux.withgradient(theta) do th
                candidate = kshock_loss_components_stacked(
                    th, state, stacked_device, 4, p, ss,
                )
                candidate.loss, candidate
            end
            components = value_gradient.val[2]
            generic_adam_step!(theta, value_gradient.grad[1], optimizer; lr)
        else
            components = kshock_loss_components_stacked(
                theta, state, stacked_device, 4, p, ss,
            )
        end
        push!(losses, Float64(components.loss))
        branch = components.first_branch
        state = (; w=branch.prices.wealth_next, y=branch.y_next, z=branch.z_next)
    end
    device === :gpu && CUDA.synchronize()
    return (; theta=Array(theta),
            state=(; w=Array(state.w), y=Array(state.y), z=Array(state.z)),
            losses, updates=optimizer.t)
end

function timed_device_training(theta, p, ss; device, periods, repetitions=3)
    train_stacked_device(theta, p, ss; device, periods=4)
    device === :gpu && CUDA.synchronize()
    times = Float64[]
    result = nothing
    for _ in 1:repetitions
        GC.gc()
        device === :gpu && CUDA.reclaim()
        started = time_ns()
        result = train_stacked_device(theta, p, ss; device, periods)
        device === :gpu && CUDA.synchronize()
        push!(times, (time_ns() - started) / 1e9)
    end
    return (; median=median(times), minimum=minimum(times), maximum=maximum(times),
            repetitions, result)
end

function trajectory_differences(reference, candidate)
    theta_relative_l2 = norm(Float64.(reference.theta) .-
                             Float64.(candidate.theta)) /
                        max(norm(Float64.(reference.theta)), eps(Float64))
    return (;
        theta_relative_l2,
        theta_max_abs=maximum(abs.(Float64.(reference.theta) .-
                                   Float64.(candidate.theta))),
        state_w_max_abs=maximum(abs.(Float64.(reference.state.w) .-
                                     Float64.(candidate.state.w))),
        state_y_max_abs=maximum(abs.(Float64.(reference.state.y) .-
                                     Float64.(candidate.state.y))),
        state_z_max_abs=maximum(abs.(Float64.(reference.state.z) .-
                                     Float64.(candidate.state.z))),
        loss_max_abs=maximum(abs.(reference.losses .- candidate.losses)),
    )
end

function gpu_main()
    CUDA.functional() || error("CUDA is not functional on this machine")
    device = CUDA.device()
    @printf("CUDA device: %s, capability %s\n", CUDA.name(device),
            string(CUDA.capability(device)))

    p = KSParams(agents=200, hidden=64, fb_convention=:released_code)
    ss = steady_state(p)
    theta64 = init_theta(MersenneTwister(1234), p; style=:reference)
    state64 = initial_state(10, p, ss)
    state_rng = MersenneTwister(8080)
    for _ in 1:8
        state64 = _advance_state(theta64, state64,
                                 draw_shocks(state_rng, 10, p), p, ss)
    end
    branch_rngs = [MersenneTwister(2026 + 100_000 * branch) for branch in 1:4]
    shocks64 = [draw_shocks(branch_rngs[branch], 10, p) for branch in 1:4]
    stacked64 = stack_branch_shocks(shocks64)

    rows = NamedTuple[]
    validations = NamedTuple[]
    BLAS.set_num_threads(12)

    for T in (Float64, Float32)
        println("testing precision $T")
        theta_cpu = T.(theta64)
        state_cpu = state_to_precision(state64, T)
        stacked_cpu = shocks_to_precision(stacked64, T)
        theta_gpu = CuArray(theta_cpu)
        state_gpu = state_to_gpu(state_cpu)
        stacked_gpu = shocks_to_gpu(stacked_cpu)

        cpu_fun = () -> Flux.withgradient(theta_cpu) do th
            components = kshock_loss_components_stacked(
                th, state_cpu, stacked_cpu, 4, p, ss,
            )
            components.loss, components
        end
        gpu_fun = () -> Flux.withgradient(theta_gpu) do th
            components = kshock_loss_components_stacked(
                th, state_gpu, stacked_gpu, 4, p, ss,
            )
            components.loss, components
        end

        cpu_result = cpu_fun()
        gpu_result = gpu_fun()
        CUDA.synchronize()
        cpu_loss = Float64(cpu_result[1][1])
        gpu_loss = Float64(gpu_result[1][1])
        cpu_gradient = cpu_result[2][1]
        gpu_gradient = Array(gpu_result[2][1])
        loss_abs_difference = abs(cpu_loss - gpu_loss)
        gradient_relative_l2_difference = (
            norm(Float64.(cpu_gradient) .- Float64.(gpu_gradient)) /
            max(norm(Float64.(cpu_gradient)), eps(Float64))
        )
        gradient_max_abs_difference = maximum(abs.(
            Float64.(cpu_gradient) .- Float64.(gpu_gradient)
        ))
        valid = T === Float64 ?
            (loss_abs_difference < 1e-10 && gradient_relative_l2_difference < 1e-9) :
            (loss_abs_difference < 1e-6 && gradient_relative_l2_difference < 1e-4)
        push!(validations, (;
            precision=string(T), cpu_loss, gpu_loss, loss_abs_difference,
            gradient_max_abs_difference, gradient_relative_l2_difference, valid,
        ))
        @printf("  equivalence |dloss|=%.3e max|dgrad|=%.3e relL2=%.3e valid=%s\n",
                loss_abs_difference, gradient_max_abs_difference,
                gradient_relative_l2_difference, string(valid))

        cpu_time = cpu_timing(cpu_fun; repetitions=10)
        gpu_time = gpu_timing(gpu_fun; repetitions=10)
        transfer_fun = () -> begin
            theta_tmp = CuArray(theta_cpu)
            state_tmp = state_to_gpu(state_cpu)
            shocks_tmp = shocks_to_gpu(stacked_cpu)
            Flux.withgradient(theta_tmp) do th
                components = kshock_loss_components_stacked(
                    th, state_tmp, shocks_tmp, 4, p, ss,
                )
                components.loss, components
            end
        end
        transfer_time = gpu_timing(transfer_fun; repetitions=10)

        for (device_name, includes_transfer, timing) in [
            ("cpu", false, cpu_time),
            ("gpu_resident", false, gpu_time),
            ("gpu_with_host_transfer", true, transfer_time),
        ]
            push!(rows, (;
                device=device_name, precision=string(T), includes_transfer,
                seconds=timing.median, minimum_seconds=timing.minimum,
                maximum_seconds=timing.maximum,
                repetitions=timing.repetitions, valid,
            ))
            @printf("  %-23s %.4fs [%.4f, %.4f]\n", device_name,
                    timing.median, timing.minimum, timing.maximum)
        end
    end

    println("end-to-end trajectory benchmark (100 periods, 3 repetitions)")
    training_rows = NamedTuple[]
    training_validations = NamedTuple[]
    for T in (Float64, Float32)
        theta_cpu = T.(theta64)
        println("  training CPU $T")
        cpu_training = timed_device_training(
            theta_cpu, p, ss; device=:cpu, periods=100, repetitions=3,
        )
        println("  training GPU $T")
        gpu_training = timed_device_training(
            theta_cpu, p, ss; device=:gpu, periods=100, repetitions=3,
        )
        differences = trajectory_differences(cpu_training.result,
                                             gpu_training.result)
        valid = T === Float64 ? differences.theta_relative_l2 < 1e-9 :
                                differences.theta_relative_l2 < 1e-4
        push!(training_validations, (;
            precision=string(T), differences..., valid,
        ))
        for (device_name, timing) in [
            ("cpu", cpu_training), ("gpu", gpu_training),
        ]
            push!(training_rows, (;
                device=device_name, precision=string(T), periods=100,
                seconds=timing.median, seconds_per_period=timing.median / 100,
                minimum_seconds=timing.minimum, maximum_seconds=timing.maximum,
                repetitions=timing.repetitions, valid,
            ))
        end
        @printf("    CPU %.3fs, GPU %.3fs, speedup %.2fx, theta relL2 %.3e, valid=%s\n",
                cpu_training.median, gpu_training.median,
                cpu_training.median / gpu_training.median,
                differences.theta_relative_l2, string(valid))
    end

    results_dir = joinpath(PROJECT_ROOT, "experiments", "results")
    mkpath(results_dir)
    timing_path = joinpath(results_dir, "ks1998_acceleration_gpu.csv")
    validation_path = joinpath(results_dir, "ks1998_acceleration_gpu_validation.csv")
    training_path = joinpath(results_dir, "ks1998_acceleration_gpu_training.csv")
    training_validation_path = joinpath(
        results_dir, "ks1998_acceleration_gpu_training_validation.csv",
    )
    write_results(timing_path, rows)
    write_results(validation_path, validations)
    write_results(training_path, training_rows)
    write_results(training_validation_path, training_validations)
    println("saved $timing_path")
    println("saved $validation_path")
    println("saved $training_path")
    println("saved $training_validation_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    gpu_main()
end
